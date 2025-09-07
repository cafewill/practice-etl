#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-지역카테고리.py — DiningCode 리스트 JSON에서
- addr / road_addr 를 3뎁스(l1,l2,l3)로 절단(제주/제주특별자치도 → 제주도 표준화)
- category 를 1뎁스(콤마 분리) 유니크 집계
- 결과 저장 (지번/도로명 분리):
    * data/region/regions-addr-triplets.json
    * data/region/regions-road-addr-triplets.json
    * data/region/regions-addr-unique.json
    * data/region/regions-road-addr-unique.json
    * data/category/categories.json

번역:
- --translate {auto|romanize|off} (기본 romanize)
- Papago 고정(키는 --config config.json; X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY)
- translate-max 기본 무제한(0 → 무제한), 캐시 파일(--cache-file) + 메모리 캐시
- trace 켜면 [CACHE]/[SKIP]/[IDENTITY] 표시 + 400 RAW 미리보기(N2MT05는 원문 반환)

시드(Seed) 병합:
- --region-seed <json> : {"l1","l2","l3","count"} 리스트 파일을 카운터에 병합
- --category-seed <json> : {"category","count"} 리스트 파일을 카운터에 병합
- --seed-top-k N : 시드에서 count 상위 N개만 사용(0이면 제한 없음)
- --seed-min-count M : 시드에서 count ≥ M 인 것만 채택
- 저장되는 unique 결과/리포트/프리뷰에 시드가 반영됨 (per-shop 리스트는 스캔 결과만)

호환:
- --translate-provider 는 더 이상 쓰지 않지만, 전달되면 경고 출력 후 무시
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- 공통 유틸 ----------------

def now_local_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z")

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0"," ").strip())

def normalize_jeju(text: Optional[str]) -> str:
    if not text:
        return ""
    s = _clean_ws(text)
    s = re.sub(r"(?<![가-힣])제주특별자치도(?![가-힣])", "제주도", s)
    s = re.sub(r"(?<![가-힣])제주(?![가-힣])", "제주도", s)
    s = re.sub(r"(제주도)(\s+\1)+", r"\1", s)
    return _clean_ws(s)

def gather_files(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for pat in patterns or []:
        paths.extend(sorted(glob.glob(pat)))
    seen = set(); out=[]
    for p in paths:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON 로드 실패: {path} ({e})", file=sys.stderr)
        return None

# ------------- 아이템 리스트 자동 탐색 --------------

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

def find_items_auto(doc: Any) -> List[Dict[str, Any]]:
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

def standardize_l1(l1: str) -> str:
    t = _clean_ws(l1)
    if not t:
        return ""
    if t in ("제주", "제주특별자치도", "제주도"):
        return "제주도"
    if "제주" in t and t.endswith("도"):
        return "제주도"
    return t

def split_addr_3(addr: str) -> Tuple[str,str,str]:
    s = normalize_jeju(addr)
    if not s: return ("","","")
    toks = s.split()

    l1 = ""
    for t in toks[:3]:
        if t == "제주도" or ("제주" in t and t.endswith("도")):
            l1 = "제주도"; break

    l2 = ""
    for t in toks:
        if RE_L2.search(t):
            l2 = t; break

    l3 = ""
    start = toks.index(l2)+1 if l2 in toks else 1
    for t in toks[start: start+6]:
        if RE_L3.search(t):
            l3 = t; break
    if not l3:
        for t in toks[start: start+6]:
            if RE_L3_ALT.search(t):
                l3 = t; break

    if not l1 and len(toks) >= 1: l1 = toks[0]
    if not l2 and len(toks) >= 2: l2 = toks[1]
    if not l3 and len(toks) >= 3: l3 = toks[2]

    l1 = standardize_l1(l1)
    l2 = _clean_ws(l2)
    l3 = _clean_ws(l3)
    return (l1, l2, l3)

# ---------------- 카테고리 집계 ----------------

def split_categories(cat: Optional[str]) -> List[str]:
    if not cat: return []
    toks = [t.strip() for t in str(cat).split(",")]
    toks = [t for t in toks if t]
    seen=set(); out=[]
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

# ====== 번역/로마자 — Papago 기반 ===================

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
    if _RE_HANGUL.search(s):
        return "ko"
    if _RE_HAN.search(s):
        return "zh"
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    alpha_cnt = sum(1 for ch in s if ch.isalpha())
    if ascii_cnt >= max(1, int(len(s) * 0.9)) and alpha_cnt > 0:
        return "en"
    return "unknown"

def norm_dest_lang(dest: str) -> str:
    d = (dest or "").lower().replace("_", "-")
    if d in ("en", "english"):
        return "en"
    if d in ("cn", "zh", "zh-cn", "zh-hans"):
        return "zh-CN"
    if d in ("tw", "zh-tw", "zh-hant"):
        return "zh-TW"
    return dest

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

# --------- HTTP 세션 ---------

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

# --------- Papago 에러 집계/요약 ---------

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
    if PAPAGO_STATS["ok"] + PAPAGO_STATS["err"] == 0:
        return
    print("\n[TX][papago] 요약")
    print(f"  - 성공 OK: {PAPAGO_STATS['ok']}")
    print(f"  - 실패 ERR: {PAPAGO_STATS['err']}")
    if PAPAGO_STATS["codes"]:
        print("  - HTTP별:", dict(PAPAGO_STATS["codes"].most_common()))
    if PAPAGO_STATS["ecodes"]:
        print("  - 에러코드별:", dict(PAPAGO_STATS["ecodes"].most_common()))
    if PAPAGO_ERRORS:
        print("  - 대표 에러 예시(최신 3건):")
        for ev in PAPAGO_ERRORS[-3:]:
            print(f"    • ts={ev.get('ts')} status={ev.get('status')} "
                  f"ecode={ev.get('errorCode')} msg={ev.get('errorMessage')} "
                  f"len={ev.get('len')} target={ev.get('target')}")

def _save_papago_errors(path: Optional[str]):
    if not path:
        return
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

# --------- Papago 클라이언트 ---------

class PapagoClient:
    PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"

    def __init__(self, config_path: str = "config.json", timeout: float = 8.0, trace: bool = False):
        self.timeout = timeout
        self.trace = trace
        self.session = _requests_session(timeout=timeout)
        self.headers = self._load_headers(config_path)

    @staticmethod
    def _load_headers(config_path: str) -> Dict[str, str]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        client_id = cfg.get("PAPAGO_CLIENT_ID") or cfg.get("X-NCP-APIGW-API-KEY-ID") or ""
        client_secret = cfg.get("PAPAGO_CLIENT_SECRET") or cfg.get("X-NCP-APIGW-API-KEY") or ""
        return {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/json",
        }

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

# --------- 번역 컨트롤 (Papago 고정) ---------

class TransCtl:
    """
    provider:
      - "papago" : Papago 호출(고정)
      - "none"   : 외부 번역 미사용(romanize/off용 내부 경로)
    """
    def __init__(self, use_papago: bool, timeout: float, max_calls: Optional[int],
                 zh_variant: str, trace: bool, config_path: str,
                 papago_honorific: bool = True):
        self.provider = "papago" if use_papago else "none"
        self.timeout = timeout
        self.max_calls = max_calls  # None이면 무제한
        self.calls_used = 0
        self.zh_variant = "zh-TW" if zh_variant == "tw" else "zh-CN"
        self.trace = trace
        self.papago_honorific = papago_honorific
        self.papago = PapagoClient(config_path=config_path, timeout=timeout, trace=trace) if use_papago else None

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

        # 입력언어 = 목적언어 → 스킵
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

        if self.provider == "papago":
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

# --------- 캐시 I/O ---------

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

# ---- 간↔번 변환(카테고리에도 사용) ---------------------------------------------

CN_TO_TW_TABLE = str.maketrans({
    "济": "濟", "岛": "島", "归": "歸", "旧": "舊", "静": "靜",
    "市": "市", "道": "道", "郡": "郡", "邑": "邑", "洞": "洞", "里": "里",
    "面": "麵",
})

def to_tw_if_needed(text_cn: str, zh_variant: str) -> str:
    return text_cn.translate(CN_TO_TW_TABLE) if zh_variant == "tw" else text_cn

# ---- 영어 Title Case 정규화 ----------------------------------------------------

def titlecase_en(s: str) -> str:
    if not s: return s
    parts = re.split(r"([/\-\s])", s)  # 구분자 보존
    def cap(tok: str) -> str:
        if not tok or tok in "/- ":
            return tok
        if tok.isupper() and len(tok) <= 4:
            return tok
        return tok[:1].upper() + tok[1:]
    return "".join(cap(p) for p in parts)

# ---- 제주 지명 전용 중국어 용어집 & 규칙 ---------------------------------------

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

# ---- 카테고리 전용 사전 --------------------------------------------------------

CATEGORY_GLOSSARY_EN: Dict[str, str] = {
    # Cafe & Dessert
    "디저트": "Dessert", "커피": "Coffee", "카페": "Cafe", "브런치": "Brunch",
    "베이커리": "Bakery", "도넛": "Donut", "케이크": "Cake", "소금빵": "Salt Bread",
    "크로플": "Croffle", "와플": "Waffle", "빙수": "Bingsu", "아이스크림": "Ice Cream",
    "젤라또": "Gelato", "북카페": "Book Cafe", "루프탑": "Rooftop Cafe", "와인바": "Wine Bar",
    # Korean & Local
    "국밥": "Gukbap", "해장국": "Haejang-guk", "김밥": "Gimbap", "떡볶이": "Tteokbokki",
    "만두": "Dumplings", "비빔밥": "Bibimbap", "불고기": "Bulgogi", "정육식당": "Butcher's BBQ",
    "돔베고기": "Dombae Pork", "근고기": "Thick-cut Pork",
    # Noodles
    "국수": "Noodles", "칼국수": "Kalguksu", "메밀국수": "Buckwheat Noodles",
    "우동": "Udon", "라멘": "Ramen", "라면": "Ramen (Korean Style)",
    # Japanese
    "스시": "Sushi", "초밥": "Sushi", "오마카세": "Omakase",
    "텐동": "Tendon", "규동": "Gyudon",
    # Western
    "파스타": "Pasta", "피자": "Pizza", "수제버거": "Handmade Burger",
    "햄버거": "Burger", "스테이크": "Steak", "비스트로": "Bistro",
    # Seafood / Jeju
    "물회": "Mulhoe", "해물탕": "Seafood Hot Pot", "해물뚝배기": "Seafood Earthen Pot",
    "전복죽": "Abalone Porridge", "전복돌솥밥": "Abalone Hot-pot Rice", "전복솥밥": "Abalone Pot Rice",
    "전복구이": "Grilled Abalone", "갈치구이": "Grilled Hairtail", "고등어구이": "Grilled Mackerel",
    "옥돔구이": "Grilled Tilefish", "갈치조림": "Braised Hairtail", "갈치국": "Hairtail Soup",
    "딱새우": "Sweet Shrimp", "딱새우회": "Sweet Shrimp Sashimi", "성게비빔밥": "Sea Urchin Bibimbap",
    "성게미역국": "Sea Urchin Seaweed Soup", "보말죽": "Top Shell Porridge", "보말칼국수": "Top Shell Kalguksu",
    # Drinks / Bars
    "맥주": "Beer", "하이볼": "Highball", "포차": "Pocha", "펍": "Pub", "술집": "Bar", "요리주점": "Gastro Pub",
    # Chinese
    "마라탕": "Mala Soup", "짬뽕": "Jjamppong", "짜장면": "Jjajangmyeon", "탕수육": "Sweet & Sour Pork",
    # Chicken / Meat
    "치킨": "Chicken", "삼겹살": "Samgyeopsal", "오겹살": "Ogeopsal",
    "돼지갈비": "Pork Ribs", "양념갈비": "Marinated Ribs", "우대갈비": "Premium Beef Ribs",
    "소갈비살": "Beef Rib Finger",
}
CATEGORY_GLOSSARY_CN: Dict[str, str] = {
    "디저트": "甜点", "커피": "咖啡", "카페": "咖啡店", "브런치": "早午餐",
    "베이커리": "烘焙店", "도넛": "甜甜圈", "케이크": "蛋糕", "소금빵": "盐面包",
    "크로플": "可颂华夫", "와플": "华夫饼", "빙수": "刨冰", "아이스크림": "冰淇淋",
    "젤라또": "意式冰淇淋", "북카페": "书店咖啡", "루프탑": "露台咖啡", "와인바": "葡萄酒吧",
    "국밥": "汤饭", "해장국": "解酒汤", "김밥": "紫菜包饭", "떡볶이": "炒年糕",
    "만두": "饺子", "비빔밥": "拌饭", "불고기": "烤肉（韩式）", "정육식당": "肉铺直烤",
    "돔베고기": "切片猪肉", "근고기": "厚切猪肉",
    "국수": "面", "칼국수": "刀削面", "메밀국수": "荞麦面",
    "우동": "乌冬面", "라멘": "拉面", "라면": "泡面",
    "스시": "寿司", "초밥": "寿司", "오마카세": "主厨精选",
    "텐동": "天丼", "규동": "牛肉盖饭",
    "파스타": "意面", "피자": "披萨", "수제버거": "手工汉堡",
    "햄버거": "汉堡", "스테이크": "牛排", "비스트로": "小酒馆",
    "물회": "凉拌生鱼汤", "해물탕": "海鲜汤", "해물뚝배기": "海鲜砂锅",
    "전복죽": "鲍鱼粥", "전복돌솥밥": "鲍鱼石锅拌饭", "전복솥밥": "鲍鱼砂锅饭",
    "전복구이": "烤鲍鱼", "갈치구이": "烤带鱼", "고등어구이": "烤青花鱼",
    "옥돔구이": "烤条石鲷", "갈치조림": "炖带鱼", "갈치국": "带鱼汤",
    "딱새우": "甜虾", "딱새우회": "甜虾生鱼片", "성게비빔밥": "海胆拌饭",
    "성게미역국": "海胆海带汤", "보말죽": "法螺粥", "보말칼국수": "法螺刀削面",
    "맥주": "啤酒", "하이볼": "高球酒", "포차": "路边摊酒馆", "펍": "啤酒屋", "술집": "小酒吧", "요리주점": "料理酒馆",
    "마라탕": "麻辣烫", "짬뽕": "什锦海鲜面", "짜장면": "炸酱面", "탕수육": "糖醋肉",
    "치킨": "炸鸡", "삼겹살": "五花肉", "오겹살": "五花三层肉",
    "돼지갈비": "猪排骨", "양념갈비": "腌制排骨", "우대갈비": "厚切牛排骨",
    "소갈비살": "牛肋条",
}

def _category_dict_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if not ko: return (None, None)
    en = CATEGORY_GLOSSARY_EN.get(ko)
    cn = CATEGORY_GLOSSARY_CN.get(ko)
    if cn: cn = to_tw_if_needed(cn, zh_variant)
    return (en, cn)

def _category_slashed_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if "/" not in ko:
        return (None, None)
    parts = [p.strip() for p in ko.split("/")]
    en_parts, cn_parts = [], []
    for p in parts:
        en_p, cn_p = _category_dict_lookup(p, zh_variant)
        en_parts.append(en_p or p)
        cn_parts.append(cn_p or p)
    en = " / ".join(titlecase_en(x) for x in en_parts)
    cn = " / ".join(cn_parts)
    return (en, to_tw_if_needed(cn, zh_variant))

# ---------------- 이름 JSON 빌더 ----------------

def build_name_obj_region(ko: str, translate_mode: str,
                          transctl: Optional[TransCtl], zh_variant: str) -> Dict[str,str]:
    ko = (ko or "").strip()
    if not ko:
        return {"ko":"", "en":"", "cn":""}

    # EN
    if translate_mode == "off":
        en = ""
    elif translate_mode == "romanize" or transctl is None or transctl.provider == "none":
        en = romanize_korean(ko) if has_hangul(ko) else ko
    else:
        if has_hangul(ko):
            t_en = transctl.translate(ko, "en")
            en = (t_en or "").strip() or romanize_korean(ko)
        else:
            en = ko
    en = titlecase_en(en)

    # CN
    if translate_mode == "off":
        cn = ""
    else:
        jeju_cn = jeju_ko_to_cn(ko, zh_variant, transctl if translate_mode == "auto" else None)
        if jeju_cn:
            cn = jeju_cn
        else:
            if translate_mode == "romanize" or transctl is None or transctl.provider == "none":
                cn = en if en else (romanize_korean(ko) if has_hangul(ko) else ko)
            else:
                dest = "zh-TW" if zh_variant == "tw" else "zh-CN"
                t_zh = transctl.translate(ko, dest) if has_hangul(ko) else None
                cn = (t_zh or (en if en else romanize_korean(ko))).strip()
                cn = to_tw_if_needed(cn, zh_variant)
    return {"ko": ko, "en": en, "cn": cn}

def build_name_obj_category(ko: str, translate_mode: str,
                            transctl: Optional[TransCtl], zh_variant: str) -> Dict[str,str]:
    ko = (ko or "").strip()
    if not ko:
        return {"ko":"", "en":"", "cn":""}

    en = None
    cn = None

    en, cn = _category_dict_lookup(ko, zh_variant)

    if en is None and cn is None and "/" in ko:
        en, cn = _category_slashed_lookup(ko, zh_variant)

    if en is None or cn is None:
        if translate_mode == "off":
            en = en if en is not None else ""
            cn = cn if cn is not None else ""
        elif translate_mode == "romanize" or transctl is None or transctl.provider == "none":
            en_fb = romanize_korean(ko) if has_hangul(ko) else ko
            cn_fb = en_fb
            en = en if en is not None else en_fb
            cn = cn if cn is not None else cn_fb
        else:
            if en is None:
                t_en = transctl.translate(ko, "en") if has_hangul(ko) else None
                en = (t_en or romanize_korean(ko) if has_hangul(ko) else ko)
            if cn is None:
                dest = "zh-TW" if zh_variant == "tw" else "zh-CN"
                t_zh = transctl.translate(ko, dest) if has_hangul(ko) else None
                cn = (t_zh or (en or romanize_korean(ko))).strip()
                cn = to_tw_if_needed(cn, zh_variant)

    en = titlecase_en(en or "")
    return {"ko": ko, "en": en, "cn": cn or ""}

# ---------------- 시드 병합 로직 ----------------

def _seed_take_top(items: List[Tuple[Any, int]], top_k: int, min_count: int) -> List[Tuple[Any,int]]:
    items = [x for x in items if x[1] >= max(1, int(min_count or 1))]
    items.sort(key=lambda x: (-x[1], str(x[0])))
    if top_k and top_k > 0:
        return items[:top_k]
    return items

def merge_region_seed(addr_triplet_counter: Dict[Tuple[str,str,str], int],
                      seed_path: Optional[str],
                      top_k: int,
                      min_count: int):
    if not seed_path:
        return
    data = read_json(seed_path)
    if not isinstance(data, list):
        print(f"[SEED][REGION] 무시: 형식 오류 {seed_path}")
        return
    rows: List[Tuple[Tuple[str,str,str],int]] = []
    for r in data:
        if not isinstance(r, dict): continue
        l1 = standardize_l1(normalize_jeju(r.get("l1") or ""))
        l2 = _clean_ws(r.get("l2") or "")
        l3 = _clean_ws(r.get("l3") or "")
        try:
            cnt = int(r.get("count") or 1)
        except Exception:
            cnt = 1
        if not (l1 or l2 or l3):
            continue
        rows.append(((l1,l2,l3), cnt))
    picked = _seed_take_top(rows, top_k, min_count)
    for (trip, cnt) in picked:
        addr_triplet_counter[trip] += cnt
    print(f"[SEED][REGION] 병합: src={seed_path} pick={len(picked)} / total={len(rows)} (min={min_count}, top_k={top_k})")

def merge_category_seed(category_counter: Counter,
                        seed_path: Optional[str],
                        top_k: int,
                        min_count: int):
    if not seed_path:
        return
    data = read_json(seed_path)
    if not isinstance(data, list):
        print(f"[SEED][CAT] 무시: 형식 오류 {seed_path}")
        return
    rows: List[Tuple[str,int]] = []
    for r in data:
        if not isinstance(r, dict): continue
        cat = _clean_ws(r.get("category") or r.get("cat") or "")
        if not cat: continue
        try:
            cnt = int(r.get("count") or 1)
        except Exception:
            cnt = 1
        rows.append((cat, cnt))
    picked = _seed_take_top(rows, top_k, min_count)
    for (cat, cnt) in picked:
        category_counter[cat] += cnt
    print(f"[SEED][CAT] 병합: src={seed_path} pick={len(picked)} / total={len(rows)} (min={min_count}, top_k={top_k})")

# ---------------- 집계 처리 (+ 진행 로그) ----------------

def process_files(paths: List[str], dot_path: Optional[str], log_every: int,
                  transctl: Optional[TransCtl]):
    regions_addr_per_shop: List[Dict[str, Any]] = []
    regions_road_per_shop: List[Dict[str, Any]] = []
    addr_triplet_counter = defaultdict(int)
    road_triplet_counter = defaultdict(int)
    category_counter: Counter = Counter()

    total_items = 0
    for p in paths:
        doc = read_json(p)
        if doc is None:
            continue
        items = get_by_path(doc, dot_path) if dot_path else find_items_auto(doc)
        if not (isinstance(items, list) and items and isinstance(items[0], dict)):
            items = find_items_auto(doc)
        total_items += len(items or [])

    done = 0
    t0 = time.perf_counter()
    le = max(1, int(log_every or 100))

    for p in paths:
        doc = read_json(p)
        if doc is None:
            continue
        items = get_by_path(doc, dot_path) if dot_path else find_items_auto(doc)
        if not (isinstance(items, list) and items and isinstance(items[0], dict)):
            items = find_items_auto(doc)

        for it in (items or []):
            if not isinstance(it, dict):
                continue
            rid = (it.get("v_rid") or "").strip()

            # STEP: 현재 진행상황
            done += 1
            print(f"[STEP] {done}/{total_items} v_rid={rid or '-'}")

            addr = normalize_jeju(it.get("addr"))
            l1a,l2a,l3a = split_addr_3(addr) if addr else ("","","")
            regions_addr_per_shop.append({
                "v_rid": rid,
                "addr3": {"l1": l1a, "l2": l2a, "l3": l3a},
            })
            if l1a or l2a or l3a:
                addr_triplet_counter[(l1a,l2a,l3a)] += 1

            raddr = normalize_jeju(it.get("road_addr") or it.get("roadAddress") or it.get("road_addr_full"))
            l1r,l2r,l3r = split_addr_3(raddr) if raddr else ("","","")
            regions_road_per_shop.append({
                "v_rid": rid,
                "road_addr3": {"l1": l1r, "l2": l2r, "l3": l3r},
            })
            if l1r or l2r or l3r:
                road_triplet_counter[(l1r,l2r,l3r)] += 1

            for tok in split_categories(it.get("category")):
                category_counter[tok] += 1

            if (done % le == 0) or (done == total_items):
                elapsed = time.perf_counter() - t0
                pct = (done / total_items * 100.0) if total_items else 100.0
                calls = getattr(transctl, "calls_used", None) if transctl else None
                extra = f", tx_calls={calls}" if calls is not None else ""
                print(f"[PROG] scan {done}/{total_items} ({pct:.1f}%) elapsed={elapsed:.1f}s{extra}")

    return (regions_addr_per_shop, regions_road_per_shop,
            addr_triplet_counter, road_triplet_counter, category_counter)

# ---------------- 프리뷰 출력 ----------------

def _fmt_name_line(tag: str, obj: Dict[str,str], extra: str = "") -> str:
    return f"{tag:8s} | ko: {obj.get('ko','')} | en: {obj.get('en','')} | cn: {obj.get('cn','')}{extra}"

def print_regions_preview(addr_rows: List[Dict[str, Any]],
                          road_rows: List[Dict[str, Any]],
                          limit: int = 10):
    print("\n[PREVIEW] Regions - addr (v_rid | l1/l2/l3)")
    print("v_rid           | addr(l1/l2/l3)")
    print("-"*56)
    for r in addr_rows[:limit]:
        a = r["addr3"]
        addr_s = f"{a['l1']} {a['l2']} {a['l3']}".strip()
        print(f"{(r['v_rid'] or ''):15s} | {addr_s:30s}")
    if len(addr_rows) > limit:
        print(f"... ({len(addr_rows)-limit} more)")

    print("\n[PREVIEW] Regions - road_addr (v_rid | l1/l2/l3)")
    print("v_rid           | road_addr(l1/l2/l3)")
    print("-"*60)
    for r in road_rows[:limit]:
        a = r["road_addr3"]
        addr_s = f"{a['l1']} {a['l2']} {a['l3']}".strip()
        print(f"{(r['v_rid'] or ''):15s} | {addr_s:30s}")
    if len(road_rows) > limit:
        print(f"... ({len(road_rows)-limit} more)")

def print_regions_namejson_preview(addr_triplet_counter: Dict[Tuple[str,str,str], int],
                                   translate_mode: str,
                                   transctl: Optional[TransCtl],
                                   zh_variant: str,
                                   limit_per_level: int = 10):
    l1_cnt = defaultdict(int)
    l2_cnt = defaultdict(int)
    l3_cnt = dict(addr_triplet_counter)

    for (l1,l2,l3), c in addr_triplet_counter.items():
        if l1: l1_cnt[l1] += c
        if l1 and l2: l2_cnt[(l1,l2)] += c

    def top(d, n):
        return list(sorted(d.items(), key=lambda x: (-x[1], x[0])))[:n]

    print("\n[NAMEJSON] Regions L1 (top)")
    for l1, c in top(l1_cnt, limit_per_level):
        nj = build_name_obj_region(l1, translate_mode, transctl, zh_variant)
        print(_fmt_name_line("L1", nj, extra=f"  (count={c})"))

    print("\n[NAMEJSON] Regions L2 (top)")
    for (l1,l2), c in top(l2_cnt, limit_per_level):
        nj1 = build_name_obj_region(l1, translate_mode, transctl, zh_variant)
        nj2 = build_name_obj_region(l2, translate_mode, transctl, zh_variant)
        print(_fmt_name_line("L1", nj1, extra=f"  >"))
        print(_fmt_name_line("L2", nj2, extra=f"  (count={c})"))

    print("\n[NAMEJSON] Regions L3 (top)")
    l3_pairs = [((l1,l2,l3), c) for (l1,l2,l3), c in l3_cnt.items() if l3]
    l3_pairs.sort(key=lambda x: (-x[1], x[0][0], x[0][1], x[0][2]))
    for (l1,l2,l3), c in l3_pairs[:limit_per_level]:
        nj1 = build_name_obj_region(l1, translate_mode, transctl, zh_variant)
        nj2 = build_name_obj_region(l2, translate_mode, transctl, zh_variant)
        nj3 = build_name_obj_region(l3, translate_mode, transctl, zh_variant)
        print(_fmt_name_line("L1", nj1, extra="  >"))
        print(_fmt_name_line("L2", nj2, extra="  >"))
        print(_fmt_name_line("L3", nj3, extra=f"  (count={c})"))
        print("-")

def print_categories_preview(counter: Counter, limit: int = 30):
    print("\n[PREVIEW] Categories (token : count)")
    for tok, cnt in counter.most_common(limit):
        print(f"- {tok} : {cnt}")
    rest = len(counter) - min(limit, len(counter))
    if rest > 0:
        print(f"... (+{rest} more unique)")

def print_categories_namejson_preview(counter: Counter,
                                      translate_mode: str,
                                      transctl: Optional[TransCtl],
                                      zh_variant: str,
                                      limit: int = 20):
    print("\n[NAMEJSON] Categories (top)")
    for tok, cnt in counter.most_common(limit):
        nj = build_name_obj_category(tok, translate_mode, transctl, zh_variant)
        print(_fmt_name_line("CAT", nj, extra=f"  (count={cnt})"))

def print_region_report(addr_triplet_counter: Dict[Tuple[str,str,str], int],
                        limit_l1: int, limit_l2: int, limit_l3: int):
    print("\n[REPORT] Region counts")
    l1_cnt = defaultdict(int)
    l2_cnt = defaultdict(int)
    l3_cnt = defaultdict(int)
    for (l1,l2,l3), c in addr_triplet_counter.items():
        if l1: l1_cnt[l1] += c
        if l1 and l2: l2_cnt[(l1,l2)] += c
        if l1 and l2 and l3: l3_cnt[(l1,l2,l3)] += c

    def top(d, n):
        return list(sorted(d.items(), key=lambda x: (-x[1], x[0])))[:n]

    print("  - L1 top:")
    for l1, c in top(l1_cnt, limit_l1):
        print(f"    * {l1} : {c}")

    print("  - L2 top:")
    for (l1,l2), c in top(l2_cnt, limit_l2):
        print(f"    * {l1} > {l2} : {c}")

    print("  - L3 top:")
    for (l1,l2,l3), c in top(l3_cnt, limit_l3):
        print(f"    * {l1} > {l2} > {l3} : {c}")

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    n = len(obj) if isinstance(obj, (list, dict)) else 1
    print(f"[OK] 저장: {path} (items={n})")

# ---------------- CLI ----------------

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def main():
    start = time.perf_counter()
    print(f"[TIME] 시작: {now_local_str()}")

    ap = argparse.ArgumentParser(
        description="addr/road_addr 3뎁스 분해(제주 표준화) + category 유니크 집계 (basic.json 생성 없음)"
    )
    ap.add_argument("--files", nargs="+", required=True, help="입력 JSON 파일/패턴(공백 구분)")
    ap.add_argument("--path", default=None, help="아이템 리스트 dot-path (예: result_data.poi_section.list)")

    # 화면 출력
    ap.add_argument("--show", default="true", choices=["true","false"], help="콘솔 미리보기 출력 여부 (기본 true)")
    ap.add_argument("--show-name-json", default="true", choices=["true","false"], help="name_json(ko/en/cn) 미리보기 출력")
    ap.add_argument("--show-region-report", default="false", choices=["true","false"], help="L1/L2/L3 상위 리포트 출력")
    ap.add_argument("--name-limit-reg", type=int, default=10, help="지역 L1/L2/L3 프리뷰 개수")
    ap.add_argument("--name-limit-cat", type=int, default=20, help="카테고리 프리뷰 개수")
    ap.add_argument("--report-limit-l1", type=int, default=10)
    ap.add_argument("--report-limit-l2", type=int, default=10)
    ap.add_argument("--report-limit-l3", type=int, default=10)

    # 진행 로그
    ap.add_argument("--log-every", type=int, default=100, help="N건마다 진행 상황 로그 출력 (기본 100)")

    # 저장
    ap.add_argument("--save-region", default="true", choices=["true","false"], help="지역 결과 저장 (기본 true)")
    ap.add_argument("--save-category", default="true", choices=["true","false"], help="카테고리 결과 저장 (기본 true)")
    ap.add_argument("--out-addr-triplets", default="data/region/regions-addr-triplets.json")
    ap.add_argument("--out-road-triplets", default="data/region/regions-road-addr-triplets.json")
    ap.add_argument("--out-addr-unique", default="data/region/regions-addr-unique.json")
    ap.add_argument("--out-road-unique", default="data/region/regions-road-addr-unique.json")
    ap.add_argument("--out-category", default="data/category/categories.json")

    # 번역 옵션 (Papago 고정)
    ap.add_argument("--translate", choices=["auto","romanize","off"], default="romanize")
    ap.add_argument("--translate-timeout", type=float, default=2.0)
    ap.add_argument("--translate-max", type=int, default=0, help="≤0: 무제한 (기본 0)")
    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn")
    ap.add_argument("--trace-translate", action="store_true")
    ap.add_argument("--cache-file", default=None, help="번역 결과 캐시 JSON 경로")

    # Papago 설정/로그
    ap.add_argument("--config", default="config.json", help="Papago 키(JSON) 경로")
    ap.add_argument("--papago-honorific", type=str2bool, nargs="?", const=True, default=True,
                    help="Papago 한국어 경어 옵션(true/false)")
    ap.add_argument("--tx-error-log", default=".logs/papago_errors.json",
                    help="Papago 실패 이벤트 JSON 로그 경로")

    # 시드 옵션
    ap.add_argument("--region-seed", default=None, help="regions-addr-unique.json 형식의 시드 파일")
    ap.add_argument("--category-seed", default=None, help="seed-categories.json 형식의 시드 파일")
    ap.add_argument("--seed-top-k", type=int, default=0, help="시드에서 count 상위 K개만 사용(0: 제한 없음)")
    ap.add_argument("--seed-min-count", type=int, default=1, help="시드에서 count ≥ M 인 항목만 사용")

    # 호환(무시)
    ap.add_argument("--translate-provider", default=None,
                    help="(deprecated) 무시됩니다. Papago 고정입니다.")

    args = ap.parse_args()

    if args.translate_provider:
        print(f"[WARN] --translate-provider 는 더 이상 사용하지 않습니다(Papago 고정). 입력값 '{args.translate_provider}'는 무시됩니다.")

    show = (args.show.lower() == "true")
    show_name = (args.show_name_json.lower() == "true")
    show_report = (args.show_region_report.lower() == "true")
    do_save_region = (args.save_region.lower() == "true")
    do_save_category = (args.save_category.lower() == "true")

    # 번역 컨트롤러 + 캐시
    tmax = None if args.translate_max is None or args.translate_max <= 0 else int(args.translate_max)
    transctl = None
    if args.translate == "auto":
        transctl = TransCtl(
            use_papago=True,                    # 무조건 Papago
            timeout=args.translate_timeout,
            max_calls=tmax,                     # None이면 무제한
            zh_variant=args.zh_variant,
            trace=args.trace_translate,
            config_path=args.config,
            papago_honorific=args.papago_honorific,
        )
    load_trans_cache(args.cache_file)

    # 입력 파일
    files = gather_files(args.files or [])
    if not files:
        print("[ERR] 입력 파일이 없습니다.", file=sys.stderr)
        save_trans_cache(args.cache_file)
        _save_papago_errors(args.tx_error_log)
        _print_papago_summary()
        print(f"[TIME] 종료: {now_local_str()}")
        sys.exit(2)

    # 집계 (STEP/PROG 로그 포함)
    (regions_addr_per_shop, regions_road_per_shop,
     addr_triplet_counter, road_triplet_counter, cat_counter) = process_files(
        files, args.path, args.log_every, transctl
    )

    # ===== 시드 병합 =====
    merge_region_seed(addr_triplet_counter, args.region_seed, args.seed_top_k, args.seed_min_count)
    merge_category_seed(cat_counter, args.category_seed, args.seed_top_k, args.seed_min_count)

    # 화면 출력
    if show:
        print_regions_preview(regions_addr_per_shop, regions_road_per_shop, limit=12)
        print_categories_preview(cat_counter, limit=args.name_limit_cat)
        if show_name:
            print_regions_namejson_preview(addr_triplet_counter,
                                           translate_mode=args.translate,
                                           transctl=transctl,
                                           zh_variant=args.zh_variant,
                                           limit_per_level=args.name_limit_reg)
            print_categories_namejson_preview(cat_counter,
                                              translate_mode=args.translate,
                                              transctl=transctl,
                                              zh_variant=args.zh_variant,
                                              limit=args.name_limit_cat)
        if show_report:
            print_region_report(addr_triplet_counter,
                                limit_l1=args.report_limit_l1,
                                limit_l2=args.report_limit_l2,
                                limit_l3=args.report_limit_l3)

    # 저장
    if do_save_region:
        save_json(regions_addr_per_shop, args.out_addr_triplets)
        save_json(regions_road_per_shop, args.out_road_triplets)
        uniq_addr_list = [
            {"l1":k[0], "l2":k[1], "l3":k[2], "count":cnt}
            for k,cnt in sorted(addr_triplet_counter.items(), key=lambda x:(x[0][0],x[0][1],x[0][2]))
        ]
        uniq_road_list = [
            {"l1":k[0], "l2":k[1], "l3":k[2], "count":cnt}
            for k,cnt in sorted(road_triplet_counter.items(), key=lambda x:(x[0][0],x[0][1],x[0][2]))
        ]
        save_json(uniq_addr_list, args.out_addr_unique)
        save_json(uniq_road_list, args.out_road_unique)

    if do_save_category:
        cat_list = [{"category": k, "count": v} for k,v in sorted(cat_counter.items(), key=lambda x:(-x[1], x[0]))]
        save_json(cat_list, args.out_category)

    # 캐시/로그 마무리
    save_trans_cache(args.cache_file)
    _save_papago_errors(args.tx_error_log)
    _print_papago_summary()

    end = time.perf_counter()
    print(f"[TIME] 종료: {now_local_str()}")
    print(f"[TIME] 총 소요: {end-start:.3f}s")

if __name__ == "__main__":
    main()
