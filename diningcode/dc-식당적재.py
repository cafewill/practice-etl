#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-식당적재.py — Papago 번역 연동 + 운영시간(operating_times) 일괄 적재 + 진행 로그/캐시

주요 변경
- ✅ googletrans/deep_translator 제거 → Papago NMT API 연동
- ✅ config.json: X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY 사용
- ✅ 번역 캐시(메모리+파일) + [CACHE] 로드/저장
- ✅ [TX] 상세 로그 (OK/CACHE/SKIP/IDENTITY/ERR RAW)
- ✅ 400 N2MT05(source=target) → 원문 반환 후 계속
- ✅ restaurants.operating_times 컬럼에 요청 JSON 구조 일괄 저장
- ✅ [STEP]/[PROG] 로 파일·아이템 진행 상태 출력

예시:
python3 dc-식당적재.py \
  --files "20250823-merged-list-whole.json" \
  --config config.json --profile local \
  --limit 0 \
  --owner-id 0 \
  --region-table region \
  --translate auto --translate-timeout 2.0 --translate-max 0 \
  --zh-variant cn \
  --cache-file .cache/restaurant-trans.json \
  --trace-translate
"""

import argparse, glob, json, os, re, sys, time
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import Counter

import pymysql
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===================== 공통 유틸 =====================

RE_WS        = re.compile(r"\s+")
RE_ADDR_KO   = re.compile(r"[가-힣]")
RE_L2        = re.compile(r"(제주시|서귀포시)$")
RE_L3        = re.compile(r".+(읍|면|동)$")
RE_L3_ALT    = re.compile(r".+리$")

def _clean_ws(s: Optional[str]) -> str:
    return RE_WS.sub(" ", (s or "").replace("\xa0", " ").strip())

def eprint(*a, **kw):
    print(*a, file=sys.stderr, **kw)

def norm_key(s: str) -> str:
    return _clean_ws(s).lower()

def now_local_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z")

def parse_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        return float(m.group(0)) if m else None

def str2bool(v) -> bool:
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def mask_tail(s: str, keep: int = 5) -> str:
    if not s: return "NONE"
    tail = s[-keep:] if len(s) >= keep else s
    return f"{'*'*4}{tail}"

# ===================== JEJU 표준화 & 주소 파싱 =====================

def normalize_lv1(l1: str) -> str:
    l1 = _clean_ws(l1)
    if not l1:
        return ""
    if l1 in ("제주", "제주특별자치도") or (l1.endswith("특별자치도") and "제주" in l1):
        return "제주도"
    return l1

def normalize_jeju(text: Optional[str]) -> str:
    if not text:
        return ""
    s = _clean_ws(text)
    s = re.sub(r"제주특별자치도(?=\s|$)", "제주도", s)
    s = re.sub(r"^제주(?=\s)", "제주도", s)
    s = re.sub(r"(?<=\s)제주(?=\s)", "제주도", s)
    s = re.sub(r"(제주도)(\s+\1)+", r"\1", s)
    return _clean_ws(s)

def split_addr_3(addr: str) -> Tuple[str, str, str, str]:
    """
    addr → (L1, L2, L3, rest)  // rest는 상세
    """
    s = normalize_jeju(addr)
    if not s:
        return ("", "", "", "")
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

    l1 = normalize_lv1(l1)
    prefix_cnt = sum(1 for x in (l1,l2,l3) if x)
    rest = " ".join(toks[prefix_cnt:]) if prefix_cnt < len(toks) else ""
    return (_clean_ws(l1), _clean_ws(l2), _clean_ws(l3), _clean_ws(rest))

# ===================== 파일 탐색 =====================

def gather_files(patterns: List[str]) -> List[str]:
    out, seen = [], set()
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if p not in seen:
                out.append(p); seen.add(p)
    return out

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
                if k in ("list","items","pois","restaurants") and _looks_like_item_list(vv):
                    found = vv; return
                dfs(vv)
        elif isinstance(x, list):
            for it in x:
                dfs(it)
    dfs(doc)
    return found

# ===================== DB 연결/설정 =====================

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_mysql_cfg(cfg: Dict[str, Any], profile: str) -> Dict[str, Any]:
    if "profiles" in cfg and profile in cfg["profiles"] and "mysql" in cfg["profiles"][profile]:
        return cfg["profiles"][profile]["mysql"]
    if profile in cfg and "mysql" in cfg[profile]:
        return cfg[profile]["mysql"]
    if "mysql" in cfg:
        return cfg["mysql"]
    raise ValueError("config.json에 mysql 설정이 없습니다.")

def connect_mysql(mysql_cfg: Dict[str, Any]):
    return pymysql.connect(
        host=mysql_cfg.get("host", "localhost"),
        port=int(mysql_cfg.get("port", 3306)),
        user=mysql_cfg.get("user"),
        password=mysql_cfg.get("password"),
        db=mysql_cfg.get("db", "DEMO"),
        charset=mysql_cfg.get("charset", "utf8mb4"),
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )

def table_exists(cur, table: str) -> bool:
    try:
        cur.execute(f"SELECT 1 FROM `{table}` LIMIT 1")
        cur.fetchone()
        return True
    except Exception:
        return False

def resolve_region_table_name(conn, preferred: str) -> str:
    with conn.cursor() as cur:
        if table_exists(cur, preferred): return preferred
        alt = "regions" if preferred == "region" else "region"
        if table_exists(cur, alt):
            eprint(f"[INFO] '{preferred}' 미존재 → '{alt}'로 폴백")
            return alt
        raise RuntimeError(f"region(s) 테이블을 찾을 수 없습니다. '{preferred}', '{alt}' 검사")

# ===================== 로마자/언어감지 =====================

_RE_HANGUL = re.compile(r"[가-힣]")
_RE_HAN = re.compile(r"[\u4E00-\u9FFF]")

_L = ["g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"]
_V = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo","u","wo","we","wi","yu","eu","ui","i"]
_T = ["","k","k","ks","n","nj","nh","t","l","lk","lm","lb","ls","lt","lp","lh","m","p","ps","t","t","ng","t","t","k","t","p","t"]
_roman_cache: Dict[str, str] = {}

def romanize_korean(text: str) -> str:
    key = norm_key(text)
    if key in _roman_cache: return _roman_cache[key]
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

def detect_lang_simple(text: str) -> str:
    s = (text or "").strip()
    if not s: return "unknown"
    if _RE_HANGUL.search(s): return "ko"
    if _RE_HAN.search(s): return "zh"
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    alpha_cnt = sum(1 for ch in s if ch.isalpha())
    if ascii_cnt >= max(1, int(len(s)*0.9)) and alpha_cnt>0: return "en"
    return "unknown"

def norm_dest_lang(dest: str) -> str:
    d = (dest or "").lower().replace("_","-")
    if d in ("en","english"): return "en"
    if d in ("cn","zh","zh-cn","zh-hans"): return "zh-CN"
    if d in ("tw","zh-tw","zh-hant"): return "zh-TW"
    return dest

# ===================== Papago HTTP & 번역 컨트롤러 =====================

def _requests_session(timeout: float = 8.0) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(429,500,502,503,504),
                    allowed_methods=frozenset(["GET","POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.request_timeout = timeout
    return s

PAPAGO_ERRORS: List[Dict[str, Any]] = []
PAPAGO_STATS = {"ok":0, "err":0, "codes":Counter(), "ecodes":Counter()}

def _log_papago_error(event: Dict[str, Any], trace: bool):
    PAPAGO_ERRORS.append(event); PAPAGO_STATS["err"] += 1
    if event.get("status") is not None:
        PAPAGO_STATS["codes"][str(event["status"])] += 1
    if event.get("errorCode"):
        PAPAGO_STATS["ecodes"][str(event["errorCode"])] += 1
    if trace:
        remain = event.get("headers_snippet",{}).get("x-ratelimit-remaining") or event.get("headers_snippet",{}).get("x-ratelimit-remaining-minute")
        print(f"[TX][papago][ERR] status={event.get('status')} ecode={event.get('errorCode')} remain={remain} msg={event.get('errorMessage')}")

def _log_papago_ok():
    PAPAGO_STATS["ok"] += 1

def _print_papago_summary():
    total = PAPAGO_STATS["ok"] + PAPAGO_STATS["err"]
    if total==0: return
    print("\n[TX][papago] 요약")
    print(f"  - 성공 OK: {PAPAGO_STATS['ok']}")
    print(f"  - 실패 ERR: {PAPAGO_STATS['err']}")
    if PAPAGO_STATS["codes"]:  print("  - HTTP별:", dict(PAPAGO_STATS["codes"].most_common()))
    if PAPAGO_STATS["ecodes"]: print("  - 에러코드별:", dict(PAPAGO_STATS["ecodes"].most_common()))

def _save_papago_errors(path: Optional[str]):
    if not path: return
    try: os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception: pass
    try:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(PAPAGO_ERRORS,f,ensure_ascii=False,indent=2)
        print(f"[TX][papago] 에러 로그 저장 → {path} ({len(PAPAGO_ERRORS)}건)")
    except Exception as e:
        print(f"[TX][papago] 에러 로그 저장 실패: {e}", file=sys.stderr)

class PapagoClient:
    PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"
    def __init__(self, config_path: str="config.json", timeout: float=8.0, trace: bool=False):
        self.timeout = timeout; self.trace = trace
        self.session = _requests_session(timeout=timeout)
        self.headers, self._client_id = self._load_headers(config_path)

    @staticmethod
    def _load_headers(config_path: str) -> Tuple[Dict[str,str], str]:
        try:
            with open(config_path,"r",encoding="utf-8") as f:
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
        if not code: return "en"
        c = code.replace("_","-").lower()
        mapping = {"cn":"zh-CN","zh":"zh-CN","zh-cn":"zh-CN","zh-hans":"zh-CN","tw":"zh-TW","zh-tw":"zh-TW","zh-hant":"zh-TW"}
        out = mapping.get(c, code).replace("_","-")
        if out.lower()=="zh-cn": return "zh-CN"
        if out.lower()=="zh-tw": return "zh-TW"
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

    def translate(self, text: str, target: str, source: str="auto", honorific: bool=True) -> str:
        if text is None: return ""
        payload = {"source": source or "auto", "target": self._normalize_lang(target),
                   "text": str(text), "honorific": "true" if honorific else "false"}
        try:
            r = self.session.post(self.PAPAGO_URL, headers=self.headers, json=payload, timeout=self.timeout)
            status = r.status_code
            if status == 200:
                try:
                    data = r.json()
                    out = (data.get("message",{}).get("result",{}).get("translatedText") or "").strip()
                except Exception:
                    out = ""
                if out:
                    _log_papago_ok()
                    if self.trace: print(f"[TX][papago][OK] len={len(text)} → {payload['target']}")
                    return out
                event = {"ts":now_local_str(),"status":status,"errorCode":"EMPTY_RESULT","errorMessage":"translatedText empty",
                         "target":payload["target"],"src":payload["source"],"len":len(text),
                         "headers_snippet":{k.lower():v for k,v in r.headers.items() if k.lower().startswith("x-")}}
                _log_papago_error(event,self.trace)
                return ""

            # 실패 처리 (400 포함)
            err_json, err_text = {}, ""
            try: err_json = r.json()
            except Exception: err_text = r.text

            ecode, emsg = self._parse_error_payload(err_json)
            RAW_MAX = 4096
            raw_body = None
            if status == 400:
                raw_body = (r.text or "")[:RAW_MAX]
                preview = raw_body[:600].replace("\n"," ")
                print(f"[TX][papago][400] target={payload['target']} len={len(text)} RAW({len(raw_body)}B) = {preview}")
                blob = (preview or "").lower()
                if (ecode == "N2MT05") or ("source and target must be different" in blob):
                    if self.trace: print(f"[TX][papago][IDENTITY] source==target; return original (len={len(text)})")
                    _log_papago_ok()
                    return text

            event = {"ts":now_local_str(),"status":status,"errorCode":ecode,
                     "errorMessage": emsg if emsg else (err_text[:400] if err_text else None),
                     "target":payload["target"],"src":payload["source"],"len":len(text),
                     "headers_snippet":{k.lower():v for k,v in r.headers.items() if k.lower().startswith("x-")}}
            if raw_body is not None: event["raw_body"] = raw_body
            _log_papago_error(event,self.trace)
            return ""
        except Exception as e:
            event = {"ts":now_local_str(),"status":None,"errorCode":"EXCEPTION","errorMessage":str(e),
                     "target":payload["target"],"src":source or "auto","len":len(text),"headers_snippet":{}}
            _log_papago_error(event,self.trace)
            return ""

# --- 번역 캐시 (메모리+파일) ---

_trans_cache: Dict[Tuple[str, str], str] = {}  # (dest_norm, norm_text) -> text

def load_trans_cache(path: Optional[str]):
    if not path: return
    try:
        with open(path,"r",encoding="utf-8") as f:
            d = json.load(f)
        cnt=0
        if isinstance(d, dict):
            for dest, m in d.items():
                if not isinstance(m, dict): continue
                dest_norm = norm_dest_lang(dest)
                for norm_text, val in m.items():
                    _trans_cache[(dest_norm, norm_text)] = val; cnt+=1
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass
    with open(path,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(f"[CACHE] 번역 캐시 저장: {len(_trans_cache)}건 -> {path}")

class TransCtl:
    """Papago 고정 컨트롤러. translate_max<=0 → 무제한."""
    def __init__(self, timeout: float, max_calls: Optional[int], zh_variant: str, trace: bool, config_path: str, honorific: bool=True):
        self.timeout = timeout
        self.max_calls = max_calls  # None=무제한
        self.calls_used = 0
        self.zh_variant = "zh-TW" if (zh_variant or "cn")=="tw" else "zh-CN"
        self.trace = trace
        self.honorific = honorific
        self.papago = PapagoClient(config_path=config_path, timeout=timeout, trace=trace)

    def _can_call(self) -> bool:
        return (self.max_calls is None) or (self.calls_used < self.max_calls)

    def translate(self, text: str, dest: str) -> Optional[str]:
        dest_norm = norm_dest_lang(dest)
        key = (dest_norm, norm_key(text))
        # 캐시 hit
        if key in _trans_cache:
            if self.trace: print(f"[TX][papago][OK] len={len(text)} → {dest_norm} [CACHE]")
            return _trans_cache[key]

        # 입력=목적언어 → 원문 유지
        lang = detect_lang_simple(text)
        if (dest_norm=="en" and lang=="en") or (dest_norm in ("zh-CN","zh-TW") and lang=="zh"):
            if self.trace: print(f"[TX][papago][OK] len={len(text)} → {dest_norm} [SKIP]")
            _log_papago_ok()
            _trans_cache[key] = text
            return text

        if not self._can_call():
            if self.trace: print(f"[TX] translate_max reached ({self.max_calls}); keep original.")
            return None

        self.calls_used += 1
        if self.trace: print(f"[TX] papago → {dest_norm}: len={len(text)}")
        out = self.papago.translate(text, target=dest_norm, source="auto", honorific=self.honorific)
        if out:
            _trans_cache[key] = out
        return out

def to_multi(text: str, mode: str, trans: Optional[TransCtl], zh_variant_cli: str) -> Dict[str,str]:
    """단일 텍스트 → {"ko","en","cn"}"""
    src = text or ""
    if mode == "off":
        return {"ko": src, "en": "", "cn": ""}
    if mode == "romanize" or not trans:
        en = romanize_korean(src) if RE_ADDR_KO.search(src) else src
        return {"ko": src, "en": en, "cn": en}
    # auto
    en = trans.translate(src, "en") if RE_ADDR_KO.search(src) else (src if detect_lang_simple(src)=="en" else (trans.translate(src, "en") or romanize_korean(src)))
    zh_dest = "zh-TW" if zh_variant_cli=="tw" else "zh-CN"
    cn = trans.translate(src, zh_dest) if RE_ADDR_KO.search(src) else (src if detect_lang_simple(src)=="zh" else (trans.translate(src, zh_dest) or en))
    return {"ko": src, "en": en or src, "cn": cn or (en or src)}

# ===================== 지역/카테고리 로더 =====================

def detect_name_column(cur, table: str) -> str:
    cur.execute(f"SHOW COLUMNS FROM `{table}`")
    cols = [r["Field"] for r in cur.fetchall()]
    for c in ("name","name_ko","ko_name","label","names","nm","title_ko"):
        if c in cols: return c
    return "name"

def extract_ko_from_value(val: Any) -> str:
    if val is None: return ""
    s = str(val).strip()
    if not s: return ""
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("ko","KO","kr","KR"):
                    if k in obj and isinstance(obj[k], str):
                        return _clean_ws(obj[k])
                for v in obj.values():
                    if isinstance(v, str):
                        return _clean_ws(v)
        except Exception:
            pass
    return _clean_ws(s)

def load_region_map(conn, region_table_preferred: str) -> Dict[Tuple[str, str, str], int]:
    table = resolve_region_table_name(conn, region_table_preferred)
    with conn.cursor() as cur:
        name_col = detect_name_column(cur, table)
        cur.execute(f"SELECT id, parent_id, level, `{name_col}` AS name FROM `{table}`")
        rows = cur.fetchall()
    nodes: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        nodes[r["id"]] = {"parent_id": r["parent_id"], "level": int(r["level"]), "name": extract_ko_from_value(r["name"])}
    triple_to_id: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        if int(r["level"]) != 3: continue
        n3 = nodes[r["id"]]
        p2 = nodes.get(n3["parent_id"]);  n1 = nodes.get(p2["parent_id"]) if p2 else None
        if not (p2 and n1): continue
        l1 = normalize_lv1(n1["name"]); l2 = p2["name"]; l3 = n3["name"]
        triple_to_id[(l1, l2, l3)] = r["id"]
    eprint(f"[INFO] region lv3 loaded(map size): {len(triple_to_id)} rows (table={table})")
    return triple_to_id

def load_categories(conn) -> Dict[str, int]:
    table = "restaurant_categories"
    with conn.cursor() as cur:
        name_col = detect_name_column(cur, table)
        cur.execute(f"SELECT id, `{name_col}` as name FROM `{table}`")
        rows = cur.fetchall()
    m: Dict[str, int] = {}
    for r in rows:
        raw = str(r["name"] or "").strip()
        ko = extract_ko_from_value(raw)
        names = set()
        try:
            obj = json.loads(raw) if raw.startswith("{") else {}
        except Exception:
            obj = {}
        for k in ("ko","KO","en","EN","cn","CN"):
            v = obj.get(k) if isinstance(obj, dict) else None
            if isinstance(v, str) and v.strip(): names.add(_clean_ws(v))
        if ko: names.add(ko)
        if not names and raw: names.add(_clean_ws(raw))
        for name in names:
            m[norm_key(name)] = int(r["id"])
    eprint(f"[INFO] restaurant_categories loaded: {len(m)} tokens")
    return m

def match_categories_for_item(item: Dict[str, Any], cat_map: Dict[str, int]) -> List[int]:
    fields = []
    for k in ["category","categories","category_names","cat","cats","cat_big","cat_mid","cat_small","mainCategory","subCategory"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            fields.extend(re.split(r"[\/\|,;>\]\)\(]+|\s{2,}", v))
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, str): fields.append(it)
    out: List[int] = []; seen: Set[int] = set()
    for token in fields:
        tk = norm_key(token)
        if not tk: continue
        if tk in cat_map:
            cid = cat_map[tk]
            if cid not in seen: out.append(cid); seen.add(cid)
        else:
            if len(tk) >= 2:
                for key, cid in cat_map.items():
                    if key in tk or tk in key:
                        if cid not in seen: out.append(cid); seen.add(cid)
    return out

# ===================== detail/menus 로더 =====================

def _normalize_ml_dict(d: Any) -> Dict[str, str]:
    if isinstance(d, dict):
        ko = d.get("ko") or d.get("KO") or d.get("kr") or d.get("KR") or ""
        en = d.get("en") or d.get("EN") or ""
        cn = d.get("cn") or d.get("CN") or d.get("zh") or ""
        return {"ko": _clean_ws(str(ko)), "en": _clean_ws(str(en)), "cn": _clean_ws(str(cn))}
    elif isinstance(d, str):
        return {"ko": _clean_ws(d), "en": "", "cn": ""}
    return {"ko": "", "en": "", "cn": ""}

def read_detail_json(detail_dir: str, rid: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    path = os.path.join(detail_dir, f"{rid}.json")
    if not os.path.exists(path):
        return ({"ko":"","en":"","cn":""}, {"ko":"","en":"","cn":""})
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return ({"ko":"","en":"","cn":""}, {"ko":"","en":"","cn":""})
    desc = doc.get("description_json") or doc.get("description") or get_by_path(doc,"data.description_json")
    short = doc.get("short_description_json") or doc.get("short_description") or get_by_path(doc,"data.short_description_json")
    return (_normalize_ml_dict(desc), _normalize_ml_dict(short))

# --- 가격 파서 ---

_K_PAT = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[kK]\s*$")
def coerce_price_krw(v) -> Optional[int]:
    if v is None: return None
    if isinstance(v,(int,float)):
        try: n=int(round(float(v))); return n if n>=0 else None
        except Exception: return None
    s=str(v).strip()
    if not s: return None
    m=_K_PAT.match(s)
    if m:
        try: n=float(m.group(1)); n=int(round(n*1000)); return n if n>=0 else None
        except Exception: return None
    s_norm = re.sub(r"[^\d.]", "", s.replace(",",""))
    if not s_norm: return None
    try:
        n = int(round(float(s_norm)))
        return n if n>=0 else None
    except Exception:
        return None

def read_menus(menu_dir: str, rid: str) -> List[Dict[str, Any]]:
    path = os.path.join(menu_dir, f"{rid}.json")
    if not os.path.exists(path): return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return []
    items_src = []
    if isinstance(doc, list):
        items_src = doc
    elif isinstance(doc, dict):
        for k in ("menus","list","items"):
            v = doc.get(k)
            if isinstance(v, list): items_src = v; break
    if not items_src: return []
    out: List[Dict[str, Any]] = []
    for it in items_src:
        if not isinstance(it, dict): continue
        name_json = it.get("name_json") or it.get("name") or it.get("title")
        name_norm = _normalize_ml_dict(name_json)
        price_val = coerce_price_krw(it.get("price_krw") or it.get("price") or it.get("priceKRW"))
        disc_val = coerce_price_krw(it.get("discount_price") or it.get("discountPrice"))
        img = it.get("image_url") or it.get("image") or None
        out.append({
            "name_json": name_norm,
            "price_krw": price_val,
            "discount_price": disc_val,
            "image_url": img.strip() if isinstance(img, str) and img.strip() else None,
        })
    return out

# ===================== 운영시간(operating_times) 생성 =====================

def default_operating_times() -> Dict[str, Any]:
    """요청 스펙대로 mon~sun 동일 스케줄 구성."""
    base = {
        "startTime": "11:00",
        "endTime": "22:00",
        "lastOrderTime": "21:30",
        "isAvailable": True,
        "timeSlotMethod": "total",
        "breakTimes": [{"startTime": "15:00", "endTime": "17:00"}],
        "reservationTimes": {"startTime": "13:00", "endTime": "21:00"},
    }
    return {
        "mon": dict(base),
        "tue": dict(base),
        "wed": dict(base),
        "thu": dict(base),
        "fri": dict(base),
        "sat": dict(base),
        "sun": dict(base),
    }

# ===================== INSERTS =====================

def insert_restaurant(
    cur,
    item: Dict[str, Any],
    owner_id: int,
    region_map: Dict[Tuple[str, str, str], int],
    translate_mode: str,
    transctl: Optional[TransCtl],
    zh_variant: str,
    detail_dir: str,
) -> Optional[int]:
    nm = _clean_ws(item.get("nm") or item.get("name") or "")
    if not nm:
        eprint(f"[WARN] nm 없음 → skip (rid={item.get('v_rid') or item.get('rid')})")
        return None
    ml_name = to_multi(nm, translate_mode, transctl, zh_variant)

    addr = _clean_ws(item.get("addr") or item.get("address") or "")
    l1, l2, l3, rest = split_addr_3(addr)
    region_id = None
    if l1 and l2 and l3:
        region_id = region_map.get((l1, l2, l3))
        if not region_id:
            eprint(f"[WARN] region 매핑 실패: addr='{addr}' → (l1={l1}, l2={l2}, l3={l3})")
    else:
        eprint(f"[WARN] 주소 파싱 실패: '{addr}'")

    # 주소 다국어
    addr_prefix = " ".join([p for p in [l1, l2, l3] if p])
    addr_detail = rest
    ml_addr = to_multi(addr_prefix, translate_mode, transctl, zh_variant) if addr_prefix else {"ko":"","en":"","cn":""}
    ml_addr_detail = to_multi(addr_detail, translate_mode, transctl, zh_variant) if addr_detail else {"ko":"","en":"","cn":""}

    # desc / short_desc
    rid_str = str(item.get("v_rid") or item.get("rid") or "").strip()
    desc_json, short_json = read_detail_json(detail_dir, rid_str) if rid_str else ({"ko":"","en":"","cn":""},{"ko":"","en":"","cn":""})

    phone = _clean_ws(item.get("tel") or item.get("phone") or "")
    lat = parse_float(item.get("lat") or item.get("y"))
    lng = parse_float(item.get("lng") or item.get("x"))
    image = _clean_ws(item.get("image") or item.get("img") or "")

    operating_times = default_operating_times()

    # ⚠️ 스키마 주의: operating_times 컬럼 존재 필요
    cur.execute(
        """
        INSERT INTO restaurants
        (name, description, short_description, phone, address, address_detail,
         latitude, longitude, operating_times,
         waiting_enabled, reservation_enabled, status, main_image_url,
         owner_id, region_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, 1, 1, 'PENDING', %s, %s, %s)
        """,
        (
            json.dumps(ml_name, ensure_ascii=False),
            json.dumps(desc_json, ensure_ascii=False),
            json.dumps(short_json, ensure_ascii=False),
            phone or None,
            json.dumps(ml_addr, ensure_ascii=False),
            json.dumps(ml_addr_detail, ensure_ascii=False),
            lat, lng,
            json.dumps(operating_times, ensure_ascii=False),
            image or None,
            int(owner_id),
            int(region_id or 0),
        ),
    )
    return int(cur.lastrowid)

def insert_category_mappings(cur, restaurant_id: int, category_ids: List[int]) -> int:
    if not category_ids: return 0
    inserted = 0
    for idx, cid in enumerate(category_ids):
        cur.execute(
            "INSERT INTO restaurant_category_mappings (restaurant_id, category_id, is_primary) VALUES (%s,%s,%s)",
            (restaurant_id, cid, 1 if idx==0 else 0),
        )
        inserted += 1
    return inserted

def insert_menus(cur, restaurant_id: int, menus: List[Dict[str, Any]]) -> int:
    inserted = 0
    for m in menus:
        name_json = m.get("name_json") or {"ko":"","en":"","cn":""}
        price = m.get("price_krw");  price = 0 if price is None else int(price)
        discount = m.get("discount_price")
        if discount is None or (isinstance(discount,int) and discount > price):
            discount = price
        image_url = m.get("image_url") or None
        cur.execute(
            """
            INSERT INTO restaurant_menus
              (restaurant_id, name, description, price, discount_price, image_url,
               is_signature, is_popular, is_new, status, display_order)
            VALUES (%s,%s,%s,%s,%s,%s, 0,0,0,'ACTIVE',0)
            """,
            (restaurant_id,
             json.dumps(name_json, ensure_ascii=False),
             json.dumps(name_json, ensure_ascii=False),
             int(price), int(discount), image_url),
        )
        inserted += 1
    return inserted

# ===================== 진행/집계 유틸 =====================

def count_total_items(files: List[str]) -> int:
    total = 0
    for fp in files:
        try:
            with open(fp,"r",encoding="utf-8") as f: doc = json.load(f)
            items = find_items_auto(doc)
            total += len(items)
        except Exception:
            pass
    return total

# ===================== 메인 =====================

def main():
    ap = argparse.ArgumentParser(description="restaurants / restaurant_menus / restaurant_category_mappings 적재기 (Papago 번역 + operating_times)")
    ap.add_argument("--files", nargs="+", required=True, help="입력 리스트 JSON 파일/패턴")
    ap.add_argument("--config", required=True, help="DB & Papago config.json")
    ap.add_argument("--profile", default="local")
    ap.add_argument("--limit", type=int, default=0, help="처리 개수 제한 (<=0 이면 무제한)")
    ap.add_argument("--owner-id", type=int, required=True)
    ap.add_argument("--region-table", default="region", help="region 테이블명 (region | regions)")

    # 경로
    ap.add_argument("--menu-dir", default="data/menu")
    ap.add_argument("--detail-dir", default="data/detail")

    # 번역
    ap.add_argument("--translate", choices=["auto","romanize","off"], default="auto")
    ap.add_argument("--translate-timeout", type=float, default=2.0)
    ap.add_argument("--translate-max", type=int, default=0, help="≤0: 무제한")
    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn")
    ap.add_argument("--papago-honorific", type=str2bool, nargs="?", const=True, default=True,
                    help="Papago 한국어 경어(true/false)")
    ap.add_argument("--trace-translate", action="store_true")
    ap.add_argument("--cache-file", default=".cache/restaurant-trans.json")

    args = ap.parse_args()

    print(f"[TIME] 시작: {now_local_str()}")

    cfg = load_config(args.config)
    mysql_cfg = resolve_mysql_cfg(cfg, args.profile)

    files = gather_files(args.files)
    if not files:
        eprint("[ERR] --files 패턴에 해당하는 파일이 없습니다."); sys.exit(2)

    # 번역 컨트롤러/캐시
    transctl: Optional[TransCtl] = None
    if args.translate == "auto":
        tmax = None if args.translate_max is None or args.translate_max <= 0 else int(args.translate_max)
        transctl = TransCtl(timeout=args.translate_timeout, max_calls=tmax, zh_variant=args.zh_variant,
                            trace=args.trace_translate, config_path=args.config, honorific=args.papago_honorific)
        key_tail = transctl.papago.key_tail()
        max_str = "∞" if tmax is None else str(tmax)
        print(f"[CFG] translate=auto | provider=papago | zh={args.zh_variant} | timeout={args.translate_timeout}s | max={max_str}")
        print(f"[CFG][papago] honorific={args.papago_honorific} | config={args.config} | key_id={key_tail}")
        load_trans_cache(args.cache_file)
    elif args.translate == "romanize":
        print(f"[CFG] translate=romanize | provider=none | zh={args.zh_variant}")
    else:
        print(f"[CFG] translate=off | provider=none | zh={args.zh_variant}")

    # 총 아이템 카운트
    grand_total = count_total_items(files)
    print(f"[STEP] 총 처리 대상: {grand_total}건 (파일 {len(files)}개)")
    limit = int(args.limit or 0)
    unlimited = (limit <= 0)

    conn = connect_mysql(mysql_cfg)
    total_inserted = total_catmap = total_menus = 0
    processed = 0
    t0 = time.perf_counter()

    try:
        with conn.cursor() as cur:
            # 캐시 로딩: 지역/카테고리
            region_map = load_region_map(conn, args.region_table)
            cat_map = load_categories(conn)

            # 파일 반복
            for fidx, fp in enumerate(files, 1):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                except Exception as ex:
                    eprint(f"[WARN] 파일 로드 실패: {fp}: {ex}")
                    continue

                items = find_items_auto(doc)
                eprint(f"[STEP] 파일 {fidx}/{len(files)}: {os.path.basename(fp)} → {len(items)}건")

                for it in items:
                    if (not unlimited) and processed >= limit:
                        break

                    rid = str(it.get("v_rid") or it.get("rid") or "").strip()
                    nm = _clean_ws(it.get("nm") or it.get("name") or "")
                    print(f"[STEP] 아이템 {processed+1}/{grand_total}{'' if unlimited else f' (limit {limit})'} RID={rid or '-'} name='{nm[:40]}' tx_calls={getattr(transctl,'calls_used',None) if transctl else '-'}")

                    rest_id = insert_restaurant(
                        cur, it, args.owner_id, region_map,
                        translate_mode=args.translate, transctl=transctl, zh_variant=args.zh_variant,
                        detail_dir=args.detail_dir
                    )
                    if not rest_id:
                        processed += 1
                        continue

                    # 카테고리 매핑
                    cids = match_categories_for_item(it, cat_map)
                    total_catmap += insert_category_mappings(cur, rest_id, cids)

                    # 메뉴
                    menus = read_menus(args.menu_dir, rid) if rid else []
                    total_menus += insert_menus(cur, rest_id, menus)

                    processed += 1
                    total_inserted += 1

                    if (processed % 10) == 0:
                        conn.commit()
                        save_trans_cache(args.cache_file)
                        elapsed = time.perf_counter() - t0
                        print(f"[PROG] 진행: {processed}/{grand_total} ({processed/max(1,grand_total)*100:.1f}%) "
                              f"ins={total_inserted}, cat_total={total_catmap}, menus_total={total_menus}, "
                              f"elapsed={elapsed:.1f}s, tx_calls={getattr(transctl,'calls_used',None) if transctl else '-'}")

            conn.commit()
            save_trans_cache(args.cache_file)
            print(f"[DONE] restaurants inserted={total_inserted}, category mappings inserted={total_catmap}, menus inserted={total_menus} "
                  f"(limit={'unlimited' if unlimited else limit})")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C → 진행분 커밋 시도")
        try:
            conn.commit(); save_trans_cache(args.cache_file); print("[DONE] 부분 커밋 완료")
        except Exception:
            conn.rollback(); print("[ROLLBACK] 커밋 실패 → 롤백")
        finally:
            _save_papago_errors(".logs/papago_errors.json"); _print_papago_summary()
        sys.exit(1)
    except Exception as e:
        conn.rollback(); eprint(f"[ERR] 롤백: {e}")
        raise
    finally:
        try: conn.close()
        except Exception: pass
        save_trans_cache(args.cache_file)
        _save_papago_errors(".logs/papago_errors.json")
        _print_papago_summary()

    dt = time.perf_counter() - t0
    m = int(dt//60); s = dt - m*60
    print(f"[TIME] 종료: {now_local_str()}")
    print(f"[TIME] 총 소요: {dt:.3f}s ({m}m {s:.3f}s)")

if __name__ == "__main__":
    main()
