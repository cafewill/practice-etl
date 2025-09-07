#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-지역적재.py — DiningCode 리스트 JSON에서 'addr' 3뎁스(L1/L2/L3) 분해 →
regions 테이블에 계층 업서트 (Papago 번역 연동/번역 캐시/진행률 로그/용어사전)

주요 변경점
- ✅ 구글 라이브러리 번역 제거, Papago NMT API 사용
- ✅ config.json 에서 Papago 키 로드 (X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY)
- ✅ 번역 캐시(메모리+파일) 사용: --cache-file 지정 시 로드/세이브
- ✅ 상세 로그:
   [CFG] Papago 활성/비활성, 타임아웃, max 호출수, key tail
   [STEP] 현재 처리 순번/총량(+ Papago 호출 누계)
   [TX] papago 호출·캐시·스킵·동일언어 처리
   [PROG] 진행률 주기 출력
- ✅ Papago 400(N2MT05: source=target)시 원문 반환 후 계속 진행
- ✅ 중국어 간/번체 옵션: --zh-variant cn|tw

예시 실행
python3 dc-지역적재.py \
  --files "20250823-merged-list-whole.json" \
  --config config.json --profile local \
  --translate auto --translate-timeout 2.0 --translate-max 0 \
  --zh-variant cn \
  --log-every 100 \
  --cache-file .cache/trans-cache.json \
  --load-db true \
  --show true --show-levels all --show-limit 30 \
  --show-categories true --category-limit 30 \
  --export-preview out/region-preview.json \
  --export-categories out/categories-from-addr-load.json
"""

import argparse, glob, hashlib, json, os, re, sys, time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pymysql

# ---------------- 공통 유틸 ----------------

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ").strip())

def now_local_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z")

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def _compact_json(obj: Dict[str, Any]) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",",":"))
    except Exception:
        return str(obj)

def mask_tail(s: str, keep: int = 5) -> str:
    if not s:
        return "NONE"
    tail = s[-keep:] if len(s) >= keep else s
    return f"{'*'*4}{tail}"

# 제주 표준화: '제주특별자치도'→'제주도', 단독 토큰 '제주'→'제주도' (제주시/서귀포시는 보호)
def normalize_jeju(text: Optional[str]) -> str:
    if not text:
        return ""
    s = _clean_ws(text)
    s = re.sub(r"(?<![가-힣])제주특별자치도(?![가-힣])", "제주도", s)
    s = re.sub(r"(?<![가-힣])제주(?![가-힣])", "제주도", s)
    s = re.sub(r"(제주도)(\s+\1)+", r"\1", s)
    return _clean_ws(s)

def to_float(v) -> Optional[float]:
    if v is None: return None
    try:
        return float(v)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        return float(m.group(0)) if m else None

# ---- parent_id 강제 규칙 ----
def coerce_parent_id_by_level(level: int, parent_id):
    """level=1 & parent_id 미지정이면 0으로 강제."""
    if level == 1:
        if parent_id is None:
            return 0
        if isinstance(parent_id, str) and parent_id.strip().lower() in ("", "null", "none"):
            return 0
    return parent_id

# ---------------- 리스트 탐색 ----------------

def get_by_path(doc: Any, path: str) -> Optional[Any]:
    cur = doc
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur

def _looks_like_item_list(v: Any) -> bool:
    return isinstance(v, list) and v and isinstance(v[0], dict) and (
        "nm" in v[0] or "v_rid" in v[0] or "addr" in v[0]
    )

def find_items_auto(doc: Any, dot_path: Optional[str] = None) -> List[Dict[str, Any]]:
    if dot_path:
        v = get_by_path(doc, dot_path)
        if _looks_like_item_list(v):
            return v  # type: ignore
    for path in ("result_data.poi_section.list", "list", "result.list", "data.list"):
        v = get_by_path(doc, path)
        if _looks_like_item_list(v):
            return v  # type: ignore
    found: List[Dict[str, Any]] = []
    def dfs(x: Any):
        nonlocal found
        if found: return
        if isinstance(x, dict):
            for k, vv in x.items():
                if k in ("list","items","pois") and _looks_like_item_list(vv):
                    found = vv  # type: ignore
                    return
                dfs(vv)
        elif isinstance(x, list):
            for it in x:
                dfs(it)
    dfs(doc)
    return found

# ---------------- 주소 3뎁스 파서 ----------------

RE_L2 = re.compile(r"(제주시|서귀포시)$")
RE_L3 = re.compile(r".+(읍|면|동)$")
RE_L3_ALT = re.compile(r".+리$")

def split_addr_3(addr: str) -> Tuple[str,str,str]:
    """addr → (L1,L2,L3) 추출. L1은 normalize_jeju()에 의해 '제주도'로 정규화."""
    s = normalize_jeju(addr)
    if not s: return ("","","")
    toks = s.split()

    # L1
    l1 = ""
    for t in toks[:3]:
        if t == "제주도" or (t.endswith("도") and "제주" in t):
            l1 = "제주도"; break
    # L2
    l2 = ""
    for t in toks:
        if RE_L2.search(t):
            l2 = t; break
    # L3
    l3 = ""
    start = toks.index(l2)+1 if l2 in toks else 1
    for t in toks[start:start+6]:
        if RE_L3.search(t):
            l3 = t; break
    if not l3:
        for t in toks[start:start+6]:
            if RE_L3_ALT.search(t): l3 = t; break

    if not l1 and len(toks)>=1: l1 = toks[0]
    if not l2 and len(toks)>=2: l2 = toks[1]
    if not l3 and len(toks)>=3: l3 = toks[2]
    return (_clean_ws(l1), _clean_ws(l2), _clean_ws(l3))

# ---------------- 로마자 & 간단 감지 ----------------

_RE_HANGUL = re.compile(r"[가-힣]")
_RE_HAN = re.compile(r"[\u4E00-\u9FFF]")

_L = ["g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"]
_V = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo","u","wo","we","wi","yu","eu","ui","i"]
_T = ["","k","k","ks","n","nj","nh","t","l","lk","lm","lb","ls","lt","lp","lh","m","p","ps","t","t","ng","t","t","k","t","p","t"]

_roman_cache: Dict[str, str] = {}
_trans_cache: Dict[Tuple[str, str], str] = {}   # (dest_norm, norm_text) -> translated

def _norm_key(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip()).lower()

def has_hangul(s: str) -> bool:
    return bool(_RE_HANGUL.search(s or ""))

def detect_lang_simple(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "unknown"
    if _RE_HANGUL.search(s): return "ko"
    if _RE_HAN.search(s): return "zh"
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    alpha_cnt = sum(1 for ch in s if ch.isalpha())
    if ascii_cnt >= max(1, int(len(s) * 0.9)) and alpha_cnt > 0:
        return "en"
    return "unknown"

def romanize_korean(text: str) -> str:
    key = _norm_key(text)
    if key in _roman_cache:
        return _roman_cache[key]
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            s_index = code - 0xAC00
            l = s_index // 588
            v = (s_index % 588) // 28
            t = s_index % 28
            out.append(_L[l] + _V[v] + _T[t])
        else:
            out.append(ch)
    s = "".join(out)
    res = " ".join(w.capitalize() for w in re.split(r"\s+", s) if w)
    _roman_cache[key] = res
    return res

def titlecase_en(s: str) -> str:
    if not s: return s
    parts = re.split(r"([/\-\s])", s)  # 구분자 보존
    def cap(tok: str) -> str:
        if not tok or tok in "/- ": return tok
        if tok.isupper() and len(tok) <= 4: return tok
        return tok[:1].upper() + tok[1:]
    return "".join(cap(p) for p in parts)

def norm_dest_lang(dest: str) -> str:
    d = (dest or "").lower().replace("_","-")
    if d in ("en","english"): return "en"
    if d in ("cn","zh","zh-cn","zh-hans"): return "zh-CN"
    if d in ("tw","zh-tw","zh-hant"): return "zh-TW"
    return dest

# ---------------- Papago HTTP ----------------

def _requests_session(timeout: float = 8.0) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.request_timeout = timeout
    return s

PAPAGO_ERRORS: List[Dict[str, Any]] = []
PAPAGO_STATS = {"ok": 0, "err": 0, "codes": Counter(), "ecodes": Counter()}

def _log_papago_error(event: Dict[str, Any], trace: bool):
    PAPAGO_ERRORS.append(event)
    PAPAGO_STATS["err"] += 1
    if event.get("status") is not None:
        PAPAGO_STATS["codes"][str(event["status"])] += 1
    ecode = event.get("errorCode")
    if ecode:
        PAPAGO_STATS["ecodes"][str(ecode)] += 1
    if trace:
        status = event.get("status")
        ecode = event.get("errorCode")
        emsg  = event.get("errorMessage")
        remain = event.get("headers_snippet", {}).get("x-ratelimit-remaining") or \
                 event.get("headers_snippet", {}).get("x-ratelimit-remaining-minute")
        print(f"[TX][papago][ERR] status={status} ecode={ecode} remain={remain} msg={emsg}")

def _log_papago_ok():
    PAPAGO_STATS["ok"] += 1

def _print_papago_summary():
    total = PAPAGO_STATS["ok"] + PAPAGO_STATS["err"]
    if total == 0: return
    print("\n[TX][papago] 요약")
    print(f"  - 성공 OK: {PAPAGO_STATS['ok']}")
    print(f"  - 실패 ERR: {PAPAGO_STATS['err']}")
    if PAPAGO_STATS["codes"]:
        print("  - HTTP별:", dict(PAPAGO_STATS["codes"].most_common()))
    if PAPAGO_STATS["ecodes"]:
        print("  - 에러코드별:", dict(PAPAGO_STATS["ecodes"].most_common()))

def _save_papago_errors(path: Optional[str]):
    if not path: return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(PAPAGO_ERRORS, f, ensure_ascii=False, indent=2)
        print(f"[TX][papago] 에러 로그 저장 → {path} ({len(PAPAGO_ERRORS)}건)")
    except Exception as e:
        print(f"[TX][papago] 에러 로그 저장 실패: {e}", file=sys.stderr)

class PapagoClient:
    PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"

    def __init__(self, config_path: str = "config.json", timeout: float = 8.0, trace: bool = False):
        self.timeout = timeout
        self.trace = trace
        self.session = _requests_session(timeout=timeout)
        self.headers, self._client_id = self._load_headers(config_path)

    @staticmethod
    def _load_headers(config_path: str) -> Tuple[Dict[str, str], str]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        client_id = cfg.get("PAPAGO_CLIENT_ID") or cfg.get("X-NCP-APIGW-API-KEY-ID") or ""
        client_secret = cfg.get("PAPAGO_CLIENT_SECRET") or cfg.get("X-NCP-APIGW-API-KEY") or ""
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/json",
        }
        return headers, client_id

    def key_tail(self) -> str:
        return mask_tail(self._client_id, keep=5)

    @staticmethod
    def _normalize_lang(code: str) -> str:
        if not code:
            return "en"
        c = code.replace("_", "-").lower()
        mapping = {
            "cn": "zh-CN", "zh": "zh-CN", "zh-cn": "zh-CN", "zh-hans": "zh-CN",
            "tw": "zh-TW", "zh-tw": "zh-TW", "zh-hant": "zh-TW"
        }
        out = mapping.get(c, code)
        out = out.replace("_", "-")
        if out.lower() == "zh-cn": return "zh-CN"
        if out.lower() == "zh-tw": return "zh-TW"
        return out

    def _parse_error_payload(self, obj: Any) -> Tuple[Optional[str], Optional[str]]:
        code = msg = None
        if isinstance(obj, dict):
            if "error" in obj and isinstance(obj["error"], dict):
                code = obj["error"].get("errorCode")
                msg = obj["error"].get("message") or obj["error"].get("errorMessage")
            code = code or obj.get("errorCode")
            msg = msg or obj.get("errorMessage")
        return code, msg

    def translate(self, text: str, target: str, source: str = "auto", honorific: bool = True) -> str:
        if text is None:
            return ""
        payload = {
            "source": source or "auto",
            "target": self._normalize_lang(target),
            "text": str(text),
            "honorific": "true" if honorific else "false",
        }
        try:
            r = self.session.post(self.PAPAGO_URL, headers=self.headers, json=payload, timeout=self.timeout)
            status = r.status_code

            if status == 200:
                try:
                    data = r.json()
                    out = (data.get("message", {}).get("result", {}).get("translatedText") or "").strip()
                except Exception:
                    out = ""
                if out:
                    _log_papago_ok()
                    if self.trace:
                        print(f"[TX][papago][OK] len={len(text)} → {payload['target']}")
                    return out
                event = {
                    "ts": now_local_str(), "status": status, "errorCode": "EMPTY_RESULT",
                    "errorMessage": "translatedText empty", "target": payload["target"],
                    "src": payload["source"], "len": len(text),
                    "headers_snippet": {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-")}
                }
                _log_papago_error(event, self.trace)
                return ""

            # 실패: 400 예외처리(N2MT05)
            err_json, err_text = {}, ""
            try:
                err_json = r.json()
            except Exception:
                err_text = r.text

            ecode, emsg = self._parse_error_payload(err_json)

            RAW_MAX = 4096
            raw_body = None
            if status == 400:
                raw_body = (r.text or "")[:RAW_MAX]
                preview = raw_body[:600].replace("\n", " ")
                print(f"[TX][papago][400] target={payload['target']} len={len(text)} RAW({len(raw_body)}B) = {preview}")
                blob = (preview or "").lower()
                if (ecode == "N2MT05") or ("source and target must be different" in blob):
                    if self.trace:
                        print(f"[TX][papago][IDENTITY] source==target; return original (len={len(text)})")
                    _log_papago_ok()
                    return text

            event = {
                "ts": now_local_str(), "status": status,
                "errorCode": ecode,
                "errorMessage": emsg if emsg else (err_text[:400] if err_text else None),
                "target": payload["target"], "src": payload["source"], "len": len(text),
                "headers_snippet": {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-")}
            }
            if raw_body is not None:
                event["raw_body"] = raw_body
            _log_papago_error(event, self.trace)
            return ""

        except Exception as e:
            event = {
                "ts": now_local_str(), "status": None, "errorCode": "EXCEPTION",
                "errorMessage": str(e), "target": payload["target"], "src": source or "auto", "len": len(text),
                "headers_snippet": {}
            }
            _log_papago_error(event, self.trace)
            return ""

# ---------------- 번역 컨트롤러 (Papago 고정) ----------------

class TransCtl:
    """
    Papago 고정. translate_max<=0 → 무제한.
    trace=True면 호출/캐시/스킵/identity 전부 로그.
    """
    def __init__(self, timeout: float, max_calls: Optional[int],
                 zh_variant: str, trace: bool, config_path: str,
                 papago_honorific: bool = True):
        self.provider = "papago"
        self.timeout = timeout
        self.max_calls = max_calls  # None이면 무제한
        self.calls_used = 0
        self.zh_variant = "zh-TW" if zh_variant == "tw" else "zh-CN"
        self.trace = trace
        self.papago_honorific = papago_honorific
        self.papago = PapagoClient(config_path=config_path, timeout=timeout, trace=trace)

    def _can_call(self) -> bool:
        return (self.max_calls is None) or (self.calls_used < self.max_calls)

    def translate(self, text: str, dest: str) -> Optional[str]:
        dest_norm = norm_dest_lang(dest)
        key = (dest_norm, _norm_key(text))
        # 캐시 HIT
        if key in _trans_cache:
            if self.trace:
                print(f"[TX][papago][OK] len={len(text)} → {dest_norm} [CACHE]")
            return _trans_cache[key]

        # 입력언어 = 목적언어 → 스킵(원문 사용)
        lang = detect_lang_simple(text)
        if (dest_norm == "en" and lang == "en") or \
           (dest_norm in ("zh-CN","zh-TW") and lang == "zh"):
            if self.trace:
                print(f"[TX][papago][OK] len={len(text)} → {dest_norm} [SKIP]")
            _log_papago_ok()
            _trans_cache[key] = text
            return text

        if not self._can_call():
            if self.trace:
                print(f"[TX] translate_max reached ({self.max_calls}); keep original.")
            return None

        out: Optional[str] = None
        self.calls_used += 1
        if self.trace:
            print(f"[TX] papago → {dest_norm}: len={len(text)}")
        try:
            out = self.papago.translate(text, target=dest_norm, source="auto", honorific=self.papago_honorific)
        except Exception as e:
            _log_papago_error({
                "ts": now_local_str(), "status": None, "errorCode": "EXCEPTION",
                "errorMessage": f"TransCtl: {e}", "target": dest_norm, "src": "auto", "len": len(text),
                "headers_snippet": {}
            }, trace=self.trace)
            out = None

        if out:
            _trans_cache[key] = out
        return out

# ---- 캐시 I/O ------------------------------------------------

def load_trans_cache(path: Optional[str]):
    if not path: return
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cnt = 0
        if isinstance(d, dict):
            for dest, m in d.items():
                if not isinstance(m, dict): continue
                dest_norm = norm_dest_lang(dest)
                for norm_text, val in m.items():
                    _trans_cache[(dest_norm, norm_text)] = val
                    cnt += 1
        print(f"[CACHE] 번역 캐시 로드: {cnt}건 from {path}")
    except FileNotFoundError:
        print(f"[CACHE] 파일 없음(새로 생성 예정): {path}")
    except Exception as e:
        print(f"[CACHE] 로드 실패({e}) — 무시하고 진행")

def save_trans_cache(path: Optional[str]):
    if not path: return
    out: Dict[str, Dict[str, str]] = {}
    for (dest_norm, norm_text), val in _trans_cache.items():
        out.setdefault(dest_norm, {})[norm_text] = val
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[CACHE] 번역 캐시 저장: {len(_trans_cache)}건 -> {path}")

# ---- 간↔번(중국어) 변환 --------------------------------------------------------

CN_TO_TW_TABLE = str.maketrans({
    "济": "濟","岛":"島","归":"歸","旧":"舊","静":"靜","馆":"館","条":"條","术":"術",
    "汉":"漢","汤":"湯","面":"麵","酱":"醬","团":"團","凤":"鳳","点":"點","里":"裡",
    "市":"市","道":"道","郡":"郡","邑":"邑","洞":"洞",
})

def to_tw_if_needed(text_cn: str, zh_variant: str) -> str:
    return text_cn.translate(CN_TO_TW_TABLE) if zh_variant == "tw" else text_cn

# ---------------- 제주 지명 중국어 사전 ----------------

JEJU_ZH_GLOSSARY_FULL_CN: Dict[str, str] = {
    "제주도": "济州岛",
    "제주특별자치도": "济州特别自治道",
    "제주시": "济州市",
    "서귀포시": "西归浦市",
    "중문동": "中文洞",
    "성산읍": "城山邑",
    "애월읍": "涯月邑",
    "구좌읍": "旧左邑",
    "조천읍": "朝天邑",
    "한림읍": "翰林邑",
    "남원읍": "南元邑",
    "대정읍": "大静邑",
    "표선면": "表善面",
    "안덕면": "安德面",
    "우도": "牛岛",
    "우도면": "牛岛面",
}
JEJU_ZH_GLOSSARY_ROOT_CN: Dict[str, str] = {
    "제주": "济州", "서귀포": "西归浦", "중문": "中文", "성산": "城山",
    "애월": "涯月", "구좌": "旧左", "조천": "朝天", "한림": "翰林",
    "남원": "南元", "대정": "大静", "표선": "表善", "안덕": "安德", "우도": "牛岛",
}
ZH_SUFFIX_MAP = {"도":"道","시":"市","군":"郡","읍":"邑","면":"面","동":"洞","리":"里"}

def jeju_ko_to_cn(ko: str, zh_variant: str, transctl: Optional[TransCtl]) -> Optional[str]:
    src = (ko or "").strip()
    if not src or not has_hangul(src):
        return None
    hit = JEJU_ZH_GLOSSARY_FULL_CN.get(src)
    if hit:
        return to_tw_if_needed(hit, zh_variant)
    for suf in ("특별자치도","제주도","도","시","군","읍","면","동","리"):
        if src.endswith(suf):
            root = src[:-len(suf)]
            suffix_cn = ZH_SUFFIX_MAP.get(suf, "")
            mapped_root = JEJU_ZH_GLOSSARY_ROOT_CN.get(root)
            if mapped_root:
                return to_tw_if_needed(mapped_root + suffix_cn, zh_variant)
            break
    if transctl:
        dest = "zh-TW" if zh_variant == "tw" else "zh-CN"
        t = transctl.translate(src, dest)
        if t: return t.strip()
    return None

# ---------------- 카테고리 용어사전 ----------------

CATEGORY_GLOSSARY_EN: Dict[str, str] = {
    "디저트": "Dessert","치킨": "Chicken","커피": "Coffee","카페": "Cafe",
    "국밥": "Gukbap","해장국": "Haejang-guk","김밥": "Gimbap","떡볶이": "Tteokbokki",
    "만두": "Dumplings","라멘": "Ramen","라면": "Ramen (Korean Style)","우동": "Udon",
    "국수": "Noodles","칼국수": "Kalguksu","메밀국수": "Buckwheat Noodles",
    "우동/라멘": "Udon/Ramen","초밥": "Sushi","스시": "Sushi","오마카세": "Omakase",
    "텐동": "Tendon","규동": "Gyudon","파스타": "Pasta","피자": "Pizza",
    "수제버거": "Handmade Burger","햄버거": "Burger","스테이크": "Steak",
    "비빔밥": "Bibimbap","불고기": "Bulgogi","삼겹살": "Samgyeopsal","오겹살": "Ogeopsal",
    "돼지갈비": "Pork Ribs","근고기": "Thick-cut Pork","양념갈비": "Marinated Ribs",
    "우대갈비": "Premium Beef Ribs","소갈비살": "Beef Rib Finger","정육식당": "Butcher's BBQ",
    "돔베고기": "Dombae Pork","물회": "Mulhoe","해물탕": "Seafood Hot Pot",
    "해물뚝배기": "Seafood Earthen Pot","전복죽": "Abalone Porridge","전복돌솥밥": "Abalone Hot-pot Rice",
    "전복솥밥": "Abalone Pot Rice","전복구이": "Grilled Abalone","갈치구이": "Grilled Hairtail",
    "고등어구이": "Grilled Mackerel","옥돔구이": "Grilled Tilefish","갈치조림": "Braised Hairtail",
    "갈치국": "Hairtail Soup","딱새우": "Sweet Shrimp","딱새우회": "Sweet Shrimp Sashimi",
    "성게비빔밥": "Sea Urchin Bibimbap","성게미역국": "Sea Urchin Seaweed Soup",
    "보말죽": "Top Shell Porridge","보말칼국수": "Top Shell Kalguksu","마라탕": "Mala Soup",
    "짬뽕": "Jjamppong","짜장면": "Jjajangmyeon","탕수육": "Sweet & Sour Pork",
    "브런치": "Brunch","베이커리": "Bakery","빙수": "Bingsu","아이스크림": "Ice Cream",
    "젤라또": "Gelato","도넛": "Donut","케이크": "Cake","소금빵": "Salt Bread",
    "크로플": "Croffle","북카페": "Book Cafe","루프탑": "Rooftop Cafe","와인바": "Wine Bar",
    "맥주": "Beer","하이볼": "Highball","포차": "Pocha","펍": "Pub","술집": "Bar","요리주점": "Gastro Pub",
}
CATEGORY_GLOSSARY_CN: Dict[str, str] = {
    "디저트": "甜点","치킨": "炸鸡","커피": "咖啡","카페": "咖啡店","국밥": "汤饭","해장국": "解酒汤",
    "김밥": "紫菜包饭","떡볶이": "炒年糕","만두": "饺子","라멘": "拉面","라면": "泡面","우동": "乌冬面",
    "국수": "面","칼국수": "刀削面","메밀국수": "荞麦面","초밥": "寿司","스시": "寿司","오마카세": "主厨精选",
    "텐동": "天丼","규동": "牛肉盖饭","파스타": "意面","피자": "披萨","수제버거": "手工汉堡","햄버거": "汉堡",
    "스테이크": "牛排","비빔밥": "拌饭","불고기": "烤肉（韩式）","삼겹살": "五花肉","오겹살": "五花三层肉",
    "돼지갈비": "猪排骨","근고기": "厚切猪肉","양념갈비": "腌制排骨","우대갈비": "厚切牛排骨","소갈비살": "牛肋条",
    "정육식당": "肉铺直烤","돔베고기": "切片猪肉","물회": "凉拌生鱼汤","해물탕": "海鲜汤","해물뚝배기": "海鲜砂锅",
    "전복죽": "鲍鱼粥","전복돌솥밥": "鲍鱼石锅拌饭","전복솥밥": "鲍鱼砂锅饭","전복구이": "烤鲍鱼",
    "갈치구이": "烤带鱼","고등어구이": "烤青花鱼","옥돔구이": "烤条石鲷","갈치조림": "炖带鱼","갈치국": "带鱼汤",
    "딱새우": "甜虾","딱새우회": "甜虾生鱼片","성게비빔밥": "海胆拌饭","성게미역국": "海胆海带汤",
    "보말죽": "法螺粥","보말칼국수": "法螺刀削面","마라탕": "麻辣烫","짬뽕": "什锦海鲜面","짜장면": "炸酱面",
    "탕수육": "糖醋肉","브런치": "早午餐","베이커리": "烘焙店","빙수": "刨冰","아이스크림": "冰淇淋",
    "젤라또": "意式冰淇淋","도넛": "甜甜圈","케이크": "蛋糕","소금빵": "盐面包","크로플": "可颂华夫",
    "북카페": "书店咖啡","루프탑": "露台咖啡","와인바": "葡萄酒吧","맥주": "啤酒","하이볼": "高球酒",
    "포차": "路边摊酒馆","펍": "啤酒屋","술집": "小酒吧","요리주점": "料理酒馆",
}

def cat_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if not ko: return (None,None)
    en = CATEGORY_GLOSSARY_EN.get(ko)
    cn = CATEGORY_GLOSSARY_CN.get(ko)
    if cn: cn = to_tw_if_needed(cn, zh_variant)
    return (titlecase_en(en) if en else None, cn)

def cat_lookup_slashed(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if "/" not in ko: return (None, None)
    parts = [p.strip() for p in ko.split("/")]
    en_parts, cn_parts = [], []
    for p in parts:
        en_p, cn_p = cat_lookup(p, zh_variant)
        en_parts.append(en_p or p)
        cn_parts.append(cn_p or p)
    en = " / ".join(titlecase_en(x) for x in en_parts)
    cn = " / ".join(cn_parts)
    return (en, to_tw_if_needed(cn, zh_variant))

# ---------------- 지역/카테고리 name_json 빌더 ----------------

def build_region_name_obj(ko: str, translate_mode: str, transctl: Optional[TransCtl], zh_variant: str) -> Dict[str,str]:
    ko = (ko or "").strip()
    if not ko:
        return {"ko":"", "en":"", "cn":""}

    # EN
    if translate_mode == "off":
        en = ""
    elif translate_mode == "romanize" or transctl is None:
        en = romanize_korean(ko) if has_hangul(ko) else ko
    else:
        if has_hangul(ko):
            t_en = transctl.translate(ko, "en")
            en = (t_en or "").strip() or romanize_korean(ko)
        else:
            en = ko
    en = titlecase_en(en)

    # CN: 제주 사전 우선
    if translate_mode == "off":
        cn = ""
    else:
        jeju_cn = jeju_ko_to_cn(ko, zh_variant, transctl if translate_mode == "auto" else None)
        if jeju_cn:
            cn = jeju_cn
        else:
            if translate_mode == "romanize" or transctl is None:
                cn = en if en else (romanize_korean(ko) if has_hangul(ko) else ko)
            else:
                dest = "zh-TW" if zh_variant == "tw" else "zh-CN"
                t_zh = transctl.translate(ko, dest) if has_hangul(ko) else None
                cn = (t_zh or (en if en else romanize_korean(ko))).strip()
                cn = to_tw_if_needed(cn, zh_variant)

    return {"ko": ko, "en": en, "cn": cn}

def build_cat_name_json(ko: str, translate_mode: str, transctl: Optional[TransCtl], zh_variant: str) -> Dict[str,str]:
    ko = (ko or "").strip()
    if not ko: return {"ko":"", "en":"", "cn":""}

    en, cn = cat_lookup(ko, zh_variant)
    if en or cn:
        return {"ko": ko, "en": titlecase_en(en or ""), "cn": cn or ""}

    if "/" in ko:
        en_s, cn_s = cat_lookup_slashed(ko, zh_variant)
        if (en_s and not has_hangul(en_s)) or (cn_s and not has_hangul(cn_s)):
            return {"ko": ko, "en": titlecase_en(en_s or ""), "cn": cn_s or ""}

    # fallback
    if translate_mode == "off":
        return {"ko": ko, "en": "", "cn": ""}
    if translate_mode == "romanize" or transctl is None:
        en_fb = romanize_korean(ko) if has_hangul(ko) else ko
        return {"ko": ko, "en": titlecase_en(en_fb), "cn": en_fb}
    # auto
    t_en = transctl.translate(ko, "en") if has_hangul(ko) else None
    en_fb = (t_en or (romanize_korean(ko) if has_hangul(ko) else ko)).strip()
    dest = "zh-TW" if zh_variant == "tw" else "zh-CN"
    t_zh = transctl.translate(ko, dest) if has_hangul(ko) else None
    cn_fb = to_tw_if_needed((t_zh or en_fb).strip(), zh_variant)
    return {"ko": ko, "en": titlecase_en(en_fb), "cn": cn_fb}

# ---------------- 집계 (addr 전용 + 카테고리 카운트) ----------------

class Stats:
    __slots__ = ("count","sum_lat","sum_lng")
    def __init__(self):
        self.count = 0
        self.sum_lat = 0.0
        self.sum_lng = 0.0
    def add(self, lat: Optional[float], lng: Optional[float]):
        self.count += 1
        if lat is not None and lng is not None:
            self.sum_lat += float(lat)
            self.sum_lng += float(lng)
    def centroid(self) -> Tuple[Optional[float], Optional[float]]:
        if self.count <= 0 or (self.sum_lat == 0.0 and self.sum_lng == 0.0):
            return (None, None)
        return (self.sum_lat / self.count, self.sum_lng / self.count)

def split_categories(cat: Optional[str]) -> List[str]:
    if not cat: return []
    toks = [t.strip() for t in str(cat).split(",")]
    toks = [t for t in toks if t]
    seen=set(); out=[]
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def gather_stats_addr_and_categories(files: List[str], dot_path: Optional[str]) -> Tuple[
    Dict[str, Stats],
    Dict[Tuple[str,str], Stats],
    Dict[Tuple[str,str,str], Stats],
    Counter
]:
    lv1: Dict[str, Stats] = defaultdict(Stats)
    lv2: Dict[Tuple[str,str], Stats] = defaultdict(Stats)
    lv3: Dict[Tuple[str,str,str], Stats] = defaultdict(Stats)
    cat_counter: Counter = Counter()

    total_files = len(files)
    print(f"[INFO] 입력 파일 수: {total_files}")

    for fidx, fp in enumerate(files, 1):
        print(f"[STEP] 파일 {fidx}/{total_files}: {fp}")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[WARN] JSON 로드 실패: {fp} ({e})", file=sys.stderr)
            continue
        items = find_items_auto(doc, dot_path)
        if not items:
            continue

        for it in items:
            if not isinstance(it, dict): continue
            addr_raw = it.get("addr") or it.get("address")
            addr = normalize_jeju(addr_raw)
            l1,l2,l3 = split_addr_3(addr)
            if not l1:  # 최소 l1 필요
                continue
            lat = to_float(it.get("lat"))
            lng = to_float(it.get("lng"))
            lv1[l1].add(lat, lng)
            if l2:
                lv2[(l1,l2)].add(lat, lng)
            if l2 and l3:
                lv3[(l1,l2,l3)].add(lat, lng)

            # 카테고리 집계
            for tok in split_categories(it.get("category")):
                cat_counter[tok] += 1

    return lv1, lv2, lv3, cat_counter

# ---------------- 설정/DB ----------------

DEFAULT_MYSQL = {"host":"127.0.0.1","port":3306,"user":None,"password":None,"db":"DEMO","charset":"utf8mb4"}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_json_config(path: Optional[str], profile: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "profiles" in doc:
        prof = profile or doc.get("default_profile") or next(iter(doc["profiles"].keys()))
        mysql_cfg = doc["profiles"].get(prof, {}).get("mysql", {})
    else:
        mysql_cfg = doc.get("mysql", {})
    return {"mysql": mysql_cfg}

def load_env_override() -> Dict[str, Any]:
    env_map = {
        "host": os.getenv("MYSQL_HOST"),
        "port": os.getenv("MYSQL_PORT"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "db": os.getenv("MYSQL_DB"),
        "charset": os.getenv("MYSQL_CHARSET"),
    }
    if env_map["port"] is not None:
        try: env_map["port"] = int(env_map["port"])
        except ValueError: env_map["port"] = None
    env_clean = {k:v for k,v in env_map.items() if v is not None}
    return {"mysql": env_clean} if env_clean else {}

def build_effective_mysql_config(cfg_json: Dict[str, Any], cli_over: Dict[str, Any]) -> Dict[str, Any]:
    eff = {"mysql": dict(DEFAULT_MYSQL)}
    eff = deep_merge(eff, cfg_json)
    eff = deep_merge(eff, load_env_override())
    cli_clean = {k:v for k,v in cli_over.items() if v is not None}
    eff = deep_merge(eff, {"mysql": cli_clean})
    return eff["mysql"]

def connect_mysql(mysql_cfg: Dict[str, Any]):
    return pymysql.connect(
        host=mysql_cfg.get("host", DEFAULT_MYSQL["host"]),
        port=int(mysql_cfg.get("port", DEFAULT_MYSQL["port"])),
        user=mysql_cfg.get("user"),
        password=mysql_cfg.get("password"),
        database=mysql_cfg.get("db", DEFAULT_MYSQL["db"]),
        charset=mysql_cfg.get("charset", DEFAULT_MYSQL["charset"]),
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )

# ---------------- 안전장치: id=0 루트 보장 ----------------

def ensure_root_zero_row(cur, table="regions"):
    """id=0 루트 더미 보장."""
    cur.execute(f"SELECT 1 FROM {table} WHERE id=0 LIMIT 1")
    if cur.fetchone():
        return
    cur.execute("SET @OLD_SQL_MODE := @@SESSION.SQL_MODE")
    cur.execute("SET SESSION SQL_MODE = CONCAT_WS(',', @@SESSION.SQL_MODE, 'NO_AUTO_VALUE_ON_ZERO')")
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    root_name_json = json.dumps({"ko":"ROOT","en":"ROOT","cn":"ROOT"}, ensure_ascii=False)
    cur.execute(
        f"""INSERT IGNORE INTO {table}
            (id, parent_id, name, level, code, center_latitude, center_longitude, place_count, updated_at)
            VALUES (0, 0, %s, 0, 'root-0', NULL, NULL, 0, CURRENT_TIMESTAMP)""",
        (root_name_json,)
    )
    cur.execute("SET FOREIGN_KEY_CHECKS=1")
    cur.execute("SET SESSION SQL_MODE = @OLD_SQL_MODE")

# ---------------- DB 업서트 & 진행률 로그 ----------------

def make_code(scope: str, level: int, l1: str, l2: str = "", l3: str = "") -> str:
    key = f"{scope}|{level}|{l1}|{l2}|{l3}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]

def log_progress(prefix: str, done: int, total: int, calls_used: Optional[int], t0: float):
    elapsed = time.perf_counter() - t0
    pct = (done / total * 100.0) if total else 100.0
    extra = f", tx_calls={calls_used}" if calls_used is not None else ""
    print(f"[PROG] {prefix}: {done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s{extra}")

def upsert_region(
    cur,
    table: str,
    *,
    parent_id: Any,
    name_ko: str,
    level: int,
    code: str,
    centroid: Tuple[Optional[float], Optional[float]],
    place_count: int,
    translate_mode: str,
    transctl: Optional[TransCtl],
    zh_variant: str,
) -> int:
    """code(해시)로 UPSERT — name은 항상 {"ko","en","cn"} JSON 문자열로 저장."""
    parent_id = coerce_parent_id_by_level(level, parent_id)

    # ✅ 루트 레벨은 무조건 NULL, 0/''도 NULL로 정규화
    if level == 1:
        parent_id = None
    else:
        parent_id = None if (parent_id in (0, '0', '', None)) else int(parent_id)

    cur.execute(f"SELECT id FROM {table} WHERE code=%s", (code,))
    row = cur.fetchone()
    lat, lng = centroid
    name_obj = build_region_name_obj(name_ko, translate_mode, transctl, zh_variant)
    name_json = json.dumps(name_obj, ensure_ascii=False)
    if row:
        rid = row["id"]
        cur.execute(
            f"""UPDATE {table}
                SET parent_id=%s, name=%s, level=%s,
                    center_latitude=%s, center_longitude=%s,
                    place_count=%s, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s""",
            (parent_id, name_json, level, lat, lng, place_count, rid)
        )
        return rid
    else:
        cur.execute(
            f"""INSERT INTO {table}
                (parent_id, name, level, code, center_latitude, center_longitude, place_count)
                VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (parent_id, name_json, level, code, lat, lng, place_count)
        )
        return cur.lastrowid

def load_addr_stats_to_db(
    a1: Dict[str, Stats],
    a2: Dict[Tuple[str,str], Stats],
    a3: Dict[Tuple[str,str,str], Stats],
    mysql_cfg: Dict[str, Any],
    translate_mode: str,
    transctl: Optional[TransCtl],
    zh_variant: str,
    log_every: int,
    cache_file: Optional[str],
):
    """집계 결과를 regions 테이블에 적재."""
    if translate_mode == "auto":
        load_trans_cache(cache_file)
    conn = connect_mysql(mysql_cfg)
    try:
        with conn.cursor() as cur:
            ensure_root_zero_row(cur, "regions")

            table = "regions"
            print("[LOAD] regions (지번 주소) DB 적재 시작…")

            # lv1
            id_l1: Dict[str, int] = {}
            total1, total2, total3 = len(a1), len(a2), len(a3)
            t0 = time.perf_counter()
            for i, (l1, st) in enumerate(sorted(a1.items(), key=lambda x: x[0]), 1):
                code = make_code("addr", 1, l1)
                rid = upsert_region(cur, table,
                                    parent_id=None, name_ko=l1, level=1, code=code,
                                    centroid=st.centroid(), place_count=st.count,
                                    translate_mode=translate_mode, transctl=transctl, zh_variant=zh_variant)
                id_l1[l1] = rid
                print(f"[STEP] LV1 {i}/{total1} '{l1}' tx_calls={getattr(transctl,'calls_used',None)}")
                if (i % log_every == 0) or (i == total1):
                    log_progress(f"{table} lv1", i, total1, getattr(transctl, "calls_used", None), t0)

            # lv2
            id_l2: Dict[Tuple[str,str], int] = {}
            t1 = time.perf_counter()
            for j, ((l1,l2), st) in enumerate(sorted(a2.items(), key=lambda x: (x[0][0], x[0][1])), 1):
                p_id = id_l1.get(l1)
                if not p_id:
                    continue
                code = make_code("addr", 2, l1, l2)
                rid = upsert_region(cur, table,
                                    parent_id=p_id, name_ko=l2, level=2, code=code,
                                    centroid=st.centroid(), place_count=st.count,
                                    translate_mode=translate_mode, transctl=transctl, zh_variant=zh_variant)
                id_l2[(l1,l2)] = rid
                print(f"[STEP] LV2 {j}/{total2} '{l1} > {l2}' tx_calls={getattr(transctl,'calls_used',None)}")
                if (j % log_every == 0) or (j == total2):
                    log_progress(f"{table} lv2", j, total2, getattr(transctl, "calls_used", None), t1)

            # lv3
            t2 = time.perf_counter()
            for k, ((l1,l2,l3), st) in enumerate(sorted(a3.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])), 1):
                p_id = id_l2.get((l1,l2))
                if not p_id:
                    continue
                code = make_code("addr", 3, l1, l2, l3)
                upsert_region(cur, table,
                              parent_id=p_id, name_ko=l3, level=3, code=code,
                              centroid=st.centroid(), place_count=st.count,
                              translate_mode=translate_mode, transctl=transctl, zh_variant=zh_variant)
                print(f"[STEP] LV3 {k}/{total3} '{l1} > {l2} > {l3}' tx_calls={getattr(transctl,'calls_used',None)}")
                if (k % log_every == 0) or (k == total3):
                    log_progress(f"{table} lv3", k, total3, getattr(transctl, "calls_used", None), t2)

            conn.commit()
            print("[DONE] DB 커밋 완료")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C 감지 → 지금까지 진행분만 커밋 시도")
        try:
            conn.commit()
            print("[DONE] 부분 커밋 완료")
        except Exception:
            conn.rollback()
            print("[ROLLBACK] 커밋 실패 → 롤백")
        finally:
            if translate_mode == "auto":
                save_trans_cache(cache_file)
        sys.exit(1)
    except Exception as e:
        conn.rollback()
        print(f"[ERR] 롤백: {e}", file=sys.stderr)
        raise
    finally:
        conn.close()
        if translate_mode == "auto":
            save_trans_cache(cache_file)

# ---------------- 미리보기(콘솔/JSON) ----------------

def rows_from_stats_lv1(a1, translate_mode: str, transctl: Optional[TransCtl], zh_variant: str):
    rows = []
    for l1, st in sorted(a1.items(), key=lambda x: (-x[1].count, x[0])):
        lat, lng = st.centroid()
        nd = build_region_name_obj(l1, translate_mode, transctl, zh_variant)
        rows.append({
            "name": l1,
            "name_en": nd.get("en",""),
            "name_cn": nd.get("cn",""),
            "name_json": _compact_json(nd),
            "count": st.count, "center_lat": lat, "center_lng": lng
        })
    return rows

def rows_from_stats_lv2(a2, translate_mode: str, transctl: Optional[TransCtl], zh_variant: str):
    rows = []
    for (l1, l2), st in sorted(a2.items(), key=lambda x: (-x[1].count, x[0])):
        lat, lng = st.centroid()
        nd = build_region_name_obj(l2, translate_mode, transctl, zh_variant)
        rows.append({
            "l1": l1,
            "name": l2,
            "name_en": nd.get("en",""),
            "name_cn": nd.get("cn",""),
            "name_json": _compact_json(nd),
            "count": st.count, "center_lat": lat, "center_lng": lng
        })
    return rows

def rows_from_stats_lv3(a3, translate_mode: str, transctl: Optional[TransCtl], zh_variant: str):
    rows = []
    for (l1, l2, l3), st in sorted(a3.items(), key=lambda x: (-x[1].count, x[0])):
        lat, lng = st.centroid()
        nd = build_region_name_obj(l3, translate_mode, transctl, zh_variant)
        rows.append({
            "l1": l1, "l2": l2,
            "name": l3,
            "name_en": nd.get("en",""),
            "name_cn": nd.get("cn",""),
            "name_json": _compact_json(nd),
            "count": st.count, "center_lat": lat, "center_lng": lng
        })
    return rows

def print_table(rows, cols, limit=20, title=""):
    if title:
        print(f"\n[PREVIEW] {title} (top {min(limit, len(rows))}/{len(rows)})")
    if not rows:
        print("  (empty)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c,""))) for r in rows[:limit])) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows[:limit]:
        line = " | ".join(str(r.get(c, "") if r.get(c, "") is not None else "").ljust(widths[c]) for c in cols)
        print(line)

def categories_preview_rows(counter: Counter, translate_mode: str, transctl: Optional[TransCtl], zh_variant: str, limit: int = 30):
    rows = []
    for tok, cnt in counter.most_common(limit):
        nj = build_cat_name_json(tok, translate_mode, transctl, zh_variant)
        rows.append({
            "category": tok,
            "en": nj.get("en",""),
            "cn": nj.get("cn",""),
            "name_json": _compact_json(nj),
            "count": cnt
        })
    return rows

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="addr 3뎁스 → regions 테이블 적재 (Papago 번역/캐시/진행률/사전)")

    # 입력
    ap.add_argument("--files", nargs="+", required=True, help="입력 리스트 JSON 파일/패턴")
    ap.add_argument("--path", default=None, help="아이템 리스트 dot-path (기본 자동 탐색)")

    # 동작 모드
    ap.add_argument("--load-db", default="true", choices=["true","false"],
                    help="DB 업서트 수행 여부 (기본 true). false면 드라이런(미적재)")

    # 번역 옵션
    ap.add_argument("--translate", choices=["auto","romanize","off"], default="romanize")
    ap.add_argument("--translate-timeout", type=float, default=2.0)
    ap.add_argument("--translate-max", type=int, default=0, help="≤0: 무제한")
    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn")
    ap.add_argument("--trace-translate", action="store_true", help="Papago 호출/캐시/스킵 로그")
    ap.add_argument("--papago-honorific", type=str2bool, nargs="?", const=True, default=True,
                    help="Papago 한국어 경어 옵션(true/false)")

    # 진행률/캐시
    ap.add_argument("--log-every", type=int, default=100, help="N건마다 진행률 출력")
    ap.add_argument("--cache-file", default=None, help="번역 결과 캐시 JSON 경로")

    # 미리보기 옵션
    ap.add_argument("--show", default="true", choices=["true","false"],
                    help="집계 결과 콘솔 미리보기 출력 (기본 true)")
    ap.add_argument("--show-levels", default="all", choices=["all","1","2","3"],
                    help="표시할 레벨 선택 (all|1|2|3)")
    ap.add_argument("--show-limit", type=int, default=20,
                    help="각 레벨에서 출력할 최대 행 수 (기본 20)")
    ap.add_argument("--show-categories", default="false", choices=["true","false"],
                    help="카테고리 집계 프리뷰 출력")
    ap.add_argument("--category-limit", type=int, default=30,
                    help="카테고리 프리뷰 행 수")

    ap.add_argument("--export-preview", default=None,
                    help="미리보기 내용을 JSON으로 저장할 경로 (선택)")
    ap.add_argument("--export-categories", default=None,
                    help="카테고리 집계 결과를 JSON으로 저장할 경로 (선택)")

    # DB 설정
    ap.add_argument("--config", default=None, help="Papago 키 및 DB 설정 포함 가능")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--charset", default=None)

    args = ap.parse_args()

    # 파일 expand
    files: List[str] = []
    for pat in args.files:
        files.extend(sorted(glob.glob(pat)))
    files = list(dict.fromkeys(files))
    if not files:
        print("[ERR] 입력 파일이 없습니다.", file=sys.stderr)
        sys.exit(2)

    # 설정 병합 (DB만)
    cfg_json = load_json_config(args.config, args.profile)
    cli_over = {"host": args.host, "port": args.port, "user": args.user,
                "password": args.password, "db": args.db, "charset": args.charset}
    mysql_cfg = build_effective_mysql_config(cfg_json, cli_over)
    dbg = {k: (v if k!="password" else "***") for k,v in mysql_cfg.items()}

    load_db = str2bool(args.load_db)
    print(f"[CFG] MySQL: {dbg}")
    print(f"[TIME] 시작: {now_local_str()}")

    # 번역 컨트롤러 준비
    transctl = None
    if args.translate == "auto":
        tmax = None if args.translate_max is None or args.translate_max <= 0 else int(args.translate_max)
        transctl = TransCtl(
            timeout=args.translate_timeout,
            max_calls=tmax,
            zh_variant=args.zh_variant,
            trace=args.trace_translate,
            config_path=args.config or "config.json",
            papago_honorific=args.papago_honorific,
        )
        max_str = "∞" if tmax is None else str(tmax)
        key_tail = transctl.papago.key_tail() if transctl and transctl.papago else "NONE"
        print(f"[CFG] translate=auto | provider=papago | zh={args.zh_variant} | timeout={args.translate_timeout}s | max={max_str}")
        print(f"[CFG][papago] honorific={args.papago_honorific} | config={args.config or 'config.json'} | key_id={key_tail}")
        load_trans_cache(args.cache_file)
    else:
        print(f"[CFG] translate={args.translate} | provider=none | zh={args.zh_variant}  (Papago 비활성화)")

    # 집계
    a1, a2, a3, cat_counter = gather_stats_addr_and_categories(files, args.path)
    print(f"[STAT] lv1={len(a1)} / lv2={len(a2)} / lv3={len(a3)} (유니크), categories={len(cat_counter)}")

    # ---- 화면 미리보기 & JSON 내보내기 ----
    if str2bool(args.show):
        lv = args.show_levels
        lim = args.show_limit
        if lv in ("all", "1"):
            rows1 = rows_from_stats_lv1(a1, args.translate, transctl, args.zh_variant)
            print_table(
                rows1,
                cols=["name","name_en","name_cn","name_json","count","center_lat","center_lng"],
                limit=lim,
                title="Level 1 (도/광역) — name & name_json"
            )
        if lv in ("all", "2"):
            rows2 = rows_from_stats_lv2(a2, args.translate, transctl, args.zh_variant)
            print_table(
                rows2,
                cols=["l1","name","name_en","name_cn","name_json","count","center_lat","center_lng"],
                limit=lim,
                title="Level 2 (시) — name & name_json"
            )
        if lv in ("all", "3"):
            rows3 = rows_from_stats_lv3(a3, args.translate, transctl, args.zh_variant)
            print_table(
                rows3,
                cols=["l1","l2","name","name_en","name_cn","name_json","count","center_lat","center_lng"],
                limit=lim,
                title="Level 3 (읍/면/동/리) — name & name_json"
            )

        if args.show_categories.lower() == "true":
            cat_rows = categories_preview_rows(cat_counter, args.translate, transctl, args.zh_variant, limit=args.category_limit)
            print_table(
                cat_rows,
                cols=["category","en","cn","name_json","count"],
                limit=args.category_limit,
                title="Categories — glossary/translate"
            )

    if args.export_preview:
        payload = {
            "lv1": rows_from_stats_lv1(a1, args.translate, transctl, args.zh_variant),
            "lv2": rows_from_stats_lv2(a2, args.translate, transctl, args.zh_variant),
            "lv3": rows_from_stats_lv3(a3, args.translate, transctl, args.zh_variant),
            "generated_at": now_local_str(),
        }
        try:
            os.makedirs(os.path.dirname(args.export_preview), exist_ok=True)
        except Exception:
            pass
        with open(args.export_preview, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] preview JSON saved: {args.export_preview}")

    if args.export_categories:
        rows = [
            {
                "category": k,
                "count": v,
                "name_json": build_cat_name_json(k, args.translate, transctl, args.zh_variant)
            }
            for k,v in sorted(cat_counter.items(), key=lambda x:(-x[1], x[0]))
        ]
        try:
            os.makedirs(os.path.dirname(args.export_categories), exist_ok=True)
        except Exception:
            pass
        with open(args.export_categories, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[OK] categories JSON saved: {args.export_categories}")

    # ---- DB 적재 on/off ----
    if not str2bool(args.load_db):
        print("[DRY-RUN] DB 적재를 건너뜁니다 (--load-db=false).")
        if args.translate == "auto":
            save_trans_cache(args.cache_file)
            _print_papago_summary()
        print(f"[TIME] 종료: {now_local_str()}")
        return

    # DB 적재 (translate=auto면 같은 transctl/캐시 재사용)
    t0 = time.perf_counter()
    try:
        load_addr_stats_to_db(
            a1, a2, a3,
            mysql_cfg,
            args.translate,
            transctl,
            args.zh_variant,
            args.log_every,
            args.cache_file
        )
    finally:
        if args.translate == "auto":
            save_trans_cache(args.cache_file)
            _save_papago_errors(".logs/papago_errors.json")
            _print_papago_summary()
        dt = time.perf_counter() - t0
        m = int(dt // 60); s = dt - m*60
        print(f"[TIME] 종료: {now_local_str()}")
        print(f"[TIME] 총 소요: {dt:.3f}s ({m}m {s:.3f}s)")

if __name__ == "__main__":
    main()
