#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-카테고리적재.py — 기본 카테고리 트리(JSON)를 restaurant_categories 테이블로 업서트

개선 요약
- ✅ 구글 라이브러리 번역 제거 → Papago NMT API 연동
- ✅ config.json에서 Papago 키 로드 (X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY)
- ✅ 번역 캐시(메모리+파일) 지원, 캐시 HIT 로그 표시
- ✅ 상세 진행 로그: [CFG]/[STEP]/[PROG]/[TX]
- ✅ Papago 400 N2MT05(source=target)인 경우 원문 반환 후 계속
- ✅ 중국어 간/번체 옵션: --zh-variant cn|tw
- ✅ --translate-max ≤0 → 무제한

예시:
python3 dc-카테고리적재.py \
  --basic data/category/basic.json \
  --config config.json \
  --translate auto --translate-timeout 2.0 --translate-max 0 \
  --zh-variant cn \
  --cache-file .cache/category-trans.json \
  --trace-translate \
  --show true --show-limit 20 \
  --export-preview out/cat-preview.json \
  --load-db false
"""

import argparse, json, os, re, sys, time, glob
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict, Counter
from datetime import datetime

import pymysql
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= 공통 유틸 =================

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ").strip())

def now_local_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z")

def _compact_json(obj: Dict[str, Any]) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",",":"))
    except Exception:
        return str(obj)

def str2bool(v) -> bool:
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def mask_tail(s: str, keep: int = 5) -> str:
    if not s: return "NONE"
    tail = s[-keep:] if len(s) >= keep else s
    return f"{'*'*4}{tail}"

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

# ================= 번역/로마자/언어감지 =================

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
    if not s: return "unknown"
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
    parts = re.split(r"([/\-\s])", s)  # 구분자 유지
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

# ================= Papago HTTP =================

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

# ================= 번역 컨트롤러 (Papago 고정) =================

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

# ================= EN TitleCase / CN 간↔번 변환 =================

CN_TO_TW_TABLE = str.maketrans({
    "济":"濟","岛":"島","归":"歸","旧":"舊","静":"靜","馆":"館","条":"條","术":"術",
    "汉":"漢","汤":"湯","面":"麵","酱":"醬","团":"團","凤":"鳳","点":"點","里":"裡",
    "市":"市","道":"道","郡":"郡","邑":"邑","洞":"洞",
})
def to_tw_if_needed(text_cn: str, zh_variant: str) -> str:
    return text_cn.translate(CN_TO_TW_TABLE) if zh_variant == "tw" else text_cn

# ================= 카테고리 용어사전 =================

CATEGORY_GLOSSARY_EN: Dict[str, str] = {
    "디저트":"Dessert","치킨":"Chicken","커피":"Coffee","카페":"Cafe",
    "국밥":"Gukbap","해장국":"Haejang-guk","김밥":"Gimbap","떡볶이":"Tteokbokki",
    "만두":"Dumplings","라멘":"Ramen","라면":"Ramen (Korean Style)","우동":"Udon",
    "국수":"Noodles","칼국수":"Kalguksu","메밀국수":"Buckwheat Noodles",
    "우동/라멘":"Udon/Ramen","초밥":"Sushi","스시":"Sushi","오마카세":"Omakase",
    "텐동":"Tendon","규동":"Gyudon","파스타":"Pasta","피자":"Pizza",
    "수제버거":"Handmade Burger","햄버거":"Burger","스테이크":"Steak",
    "비빔밥":"Bibimbap","불고기":"Bulgogi","삼겹살":"Samgyeopsal","오겹살":"Ogeopsal",
    "돼지갈비":"Pork Ribs","근고기":"Thick-cut Pork","양념갈비":"Marinated Ribs",
    "우대갈비":"Premium Beef Ribs","소갈비살":"Beef Rib Finger","정육식당":"Butcher's BBQ",
    "돔베고기":"Dombae Pork","물회":"Mulhoe","해물탕":"Seafood Hot Pot",
    "해물뚝배기":"Seafood Earthen Pot","전복죽":"Abalone Porridge","전복돌솥밥":"Abalone Hot-pot Rice",
    "전복솥밥":"Abalone Pot Rice","전복구이":"Grilled Abalone",
    "갈치구이":"Grilled Hairtail","고등어구이":"Grilled Mackerel","옥돔구이":"Grilled Tilefish",
    "갈치조림":"Braised Hairtail","갈치국":"Hairtail Soup","딱새우":"Sweet Shrimp",
    "딱새우회":"Sweet Shrimp Sashimi","성게비빔밥":"Sea Urchin Bibimbap","성게미역국":"Sea Urchin Seaweed Soup",
    "보말죽":"Top Shell Porridge","보말칼국수":"Top Shell Kalguksu","마라탕":"Mala Soup",
    "짬뽕":"Jjamppong","짜장면":"Jjajangmyeon","탕수육":"Sweet & Sour Pork",
    "브런치":"Brunch","베이커리":"Bakery","빙수":"Bingsu","아이스크림":"Ice Cream",
    "젤라또":"Gelato","도넛":"Donut","케이크":"Cake","소금빵":"Salt Bread",
    "크로플":"Croffle","북카페":"Book Cafe","루프탑":"Rooftop Cafe","와인바":"Wine Bar",
    "맥주":"Beer","하이볼":"Highball","포차":"Pocha","펍":"Pub","술집":"Bar","요리주점":"Gastro Pub",
}
CATEGORY_GLOSSARY_CN: Dict[str, str] = {
    "디저트":"甜点","치킨":"炸鸡","커피":"咖啡","카페":"咖啡店","국밥":"汤饭","해장국":"解酒汤",
    "김밥":"紫菜包饭","떡볶이":"炒年糕","만두":"饺子","라멘":"拉面","라면":"泡面","우동":"乌冬面",
    "국수":"面","칼국수":"刀削面","메밀국수":"荞麦面","초밥":"寿司","스시":"寿司","오마카세":"主厨精选",
    "텐동":"天丼","규동":"牛肉盖饭","파스타":"意面","피자":"披萨","수제버거":"手工汉堡","햄버거":"汉堡",
    "스테이크":"牛排","비빔밥":"拌饭","불고기":"烤肉（韩式）","삼겹살":"五花肉","오겹살":"五花三层肉",
    "돼지갈비":"猪排骨","근고기":"厚切猪肉","양념갈비":"腌制排骨","우대갈비":"厚切牛排骨","소갈비살":"牛肋条",
    "정육식당":"肉铺直烤","돔베고기":"切片猪肉","물회":"凉拌生鱼汤","해물탕":"海鲜汤","해물뚝배기":"海鲜砂锅",
    "전복죽":"鲍鱼粥","전복돌솥밥":"鲍鱼石锅拌饭","전복솥밥":"鲍鱼砂锅饭","전복구이":"烤鲍鱼",
    "갈치구이":"烤带鱼","고등어구이":"烤青花鱼","옥돔구이":"烤条石鲷","갈치조림":"炖带鱼","갈치국":"带鱼汤",
    "딱새우":"甜虾","딱새우회":"甜虾生鱼片","성게비빔밥":"海胆拌饭","성게미역국":"海胆海带汤",
    "보말죽":"法螺粥","보말칼국수":"法螺刀削面","마라탕":"麻辣烫","짬뽕":"什锦海鲜面","짜장면":"炸酱面",
    "탕수육":"糖醋肉","브런치":"早午餐","베이커리":"烘焙店","빙수":"刨冰","아이스크림":"冰淇淋",
    "젤라또":"意式冰淇淋","도넛":"甜甜圈","케이크":"蛋糕","소금빵":"盐面包","크로플":"可颂华夫",
    "북카페":"书店咖啡","루프탑":"露台咖啡","와인바":"葡萄酒吧","맥주":"啤酒","하이볼":"高球酒",
    "포차":"路边摊酒馆","펍":"啤酒屋","술집":"小酒吧","요리주점":"料理酒馆",
}

def cat_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if not ko: return (None, None)
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

# ================= '메타토큰' 처리 =================
# 예) "국수 | 혼밥 , 해장" → main="국수", meta=["혼밥","해장"] (번역은 main만)

META_SEP_RE = re.compile(r"\s*\|\s*")
META_SPLIT_COMMA_RE = re.compile(r"\s*,\s*")

def split_main_and_meta(ko: str) -> Tuple[str, List[str]]:
    if not ko: return ("", [])
    parts = META_SEP_RE.split(ko, maxsplit=1)
    if len(parts) == 1:
        return (ko.strip(), [])
    main = parts[0].strip()
    tail = parts[1].strip()
    metas = [t for t in META_SPLIT_COMMA_RE.split(tail) if t]
    return (main, metas)

# ================= 이름 JSON 빌더 =================

def build_cat_name_json(ko_in: str,
                        en_in: str,
                        cn_in: str,
                        translate_mode: str,
                        transctl: Optional[TransCtl],
                        zh_variant: str,
                        lang_case: str) -> Dict[str, str]:
    """
    - ko는 원문 유지(메타 포함)하되, 번역 타겟은 main 토큰만 사용
    - en/cn 비거나 한글 포함 시 보완 (사전 → 슬래시 → 번역/로마자)
    - EN Title Case, CN 간/번 변환
    """
    ko_raw = (ko_in or "").strip()
    en_raw = (en_in or "").strip()
    cn_raw = (cn_in or "").strip()

    main_ko, _meta = split_main_and_meta(ko_raw)

    def need_fill(v: str) -> bool:
        return not v or has_hangul(v)

    # 1) glossary 우선
    en_g, cn_g = cat_lookup(main_ko, zh_variant)
    if en_g is None and cn_g is None and "/" in main_ko:
        en_g, cn_g = cat_lookup_slashed(main_ko, zh_variant)

    en = en_raw
    cn = cn_raw

    # EN
    if need_fill(en):
        if en_g:
            en = en_g
        else:
            if translate_mode == "off":
                en = ""
            elif translate_mode == "romanize" or not transctl:
                en = romanize_korean(main_ko) if has_hangul(main_ko) else main_ko
            else:
                t = transctl.translate(main_ko, "en") if has_hangul(main_ko) else None
                en = (t or (romanize_korean(main_ko) if has_hangul(main_ko) else main_ko)).strip()
    en = titlecase_en(en)

    # CN
    if need_fill(cn):
        if cn_g:
            cn = cn_g
        else:
            if translate_mode == "off":
                cn = ""
            elif translate_mode == "romanize" or not transctl:
                cn = en if en else (romanize_korean(main_ko) if has_hangul(main_ko) else main_ko)
            else:
                dest = "zh-TW" if zh_variant == "tw" else "zh-CN"
                t = transctl.translate(main_ko, dest) if has_hangul(main_ko) else None
                cn = (t or (en if en else romanize_korean(main_ko))).strip()
        cn = to_tw_if_needed(cn, zh_variant)

    # 키 케이스
    if lang_case == "upper":
        return {"KR": ko_raw, "EN": en, "CN": cn}
    return {"ko": ko_raw, "en": en, "cn": cn}

def parse_name_dict(name_any: Any) -> Dict[str, str]:
    if isinstance(name_any, str):
        return {"ko": name_any, "en": "", "cn": ""}
    if isinstance(name_any, dict):
        return {
            "ko": name_any.get("ko") or name_any.get("KR") or "",
            "en": name_any.get("en") or name_any.get("EN") or "",
            "cn": name_any.get("cn") or name_any.get("CN") or "",
        }
    return {"ko":"", "en":"", "cn":""}

# ================= 입력 JSON 파서 =================

def _normalize_node(n: Any) -> Dict[str, Any]:
    """
    허용 형태:
      - {"name": "커피"}
      - {"name": {"ko":"커피","en":"","cn":""}, "children":[...] }
    반환 공통키:
      {"name":{"ko","en","cn"}, "icon_url","display_order","is_active","is_default","description","children":[...]}
    """
    if isinstance(n, str):
        n = {"name": n}
    name_obj = parse_name_dict(n.get("name"))
    return {
        "name": name_obj,
        "icon_url": n.get("icon_url"),
        "display_order": n.get("display_order"),
        "is_active": n.get("is_active", 1),
        "is_default": n.get("is_default", 0),
        "description": n.get("description"),
        "children": n.get("children") or [],
    }

def load_basic_tree(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    roots = doc.get("categories") if isinstance(doc, dict) else doc
    if not isinstance(roots, list):
        raise ValueError("basic.json 루트는 배열 또는 {categories: [...]} 이어야 합니다.")
    flat: List[Dict[str, Any]] = []
    q = deque([("", _normalize_node(n)) for n in roots])
    while q:
        parent_path, cur = q.popleft()
        nm = cur["name"]["ko"] or cur["name"]["en"] or cur["name"]["cn"]
        path = nm if not parent_path else f"{parent_path} > {nm}"
        flat.append({"parent_path": parent_path, "path": path, **cur})
        for c in cur["children"]:
            q.append((path, _normalize_node(c)))
    return flat

# ================= DB 업서트 =================

def ensure_category(cur,
                    parent_id: Optional[int],
                    name_json: str,
                    name_key_for_match: str,
                    name_ko: str,
                    icon_url: Optional[str],
                    display_order: Optional[int],
                    is_active: int,
                    is_default: int,
                    description: Optional[str]) -> int:
    if parent_id is None:
        parent_clause = "parent_id IS NULL"
        params = []
    else:
        parent_clause = "parent_id = %s"
        params = [parent_id]

    cur.execute(
        f"""SELECT id FROM restaurant_categories
            WHERE {parent_clause}
              AND JSON_UNQUOTE(JSON_EXTRACT(name, '$.{name_key_for_match}')) = %s
            LIMIT 1""",
        params + [name_ko]
    )
    row = cur.fetchone()
    if row:
        rid = row["id"]
        cur.execute(
            """UPDATE restaurant_categories
               SET name=%s, icon_url=%s,
                   display_order=COALESCE(%s, display_order),
                   is_active=%s, is_default=%s,
                   description=%s,
                   updated_at=CURRENT_TIMESTAMP
               WHERE id=%s""",
            (name_json, icon_url, display_order, is_active, is_default, description, rid)
        )
        return rid
    else:
        cur.execute(
            """INSERT INTO restaurant_categories
               (parent_id, name, icon_url, display_order, is_active, is_default, description)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (parent_id, name_json, icon_url, display_order or 0, is_active, is_default, description)
        )
        return cur.lastrowid

# ================= 미리보기 출력 =================

def print_table(rows: List[Dict[str, Any]], cols: List[str], limit: int = 20, title: str = ""):
    if title:
        print(f"\n[PREVIEW] {title} (top {min(limit, len(rows))}/{len(rows)})")
    if not rows:
        print("  (empty)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c,""))) for r in rows[:limit])) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-"*len(header))
    for r in rows[:limit]:
        line = " | ".join(str(r.get(c,"") if r.get(c,"") is not None else "").ljust(widths[c]) for c in cols)
        print(line)

# ================= 메인 =================

def main():
    ap = argparse.ArgumentParser(description="기본 카테고리 JSON → restaurant_categories 업서트 (Papago 번역/캐시/진행로그)")

    # 입력/동작
    ap.add_argument("--basic", required=True, help="기본 카테고리 JSON 경로 (예: data/category/basic.json)")
    ap.add_argument("--load-db", default="true", choices=["true","false"], help="DB 업서트 수행 여부 (기본 true). false면 드라이런")
    ap.add_argument("--lang-case", choices=["lower","upper"], default="lower", help="name 키 대소문자 선택(lower=ko/en/cn, upper=KR/EN/CN)")

    # 번역
    ap.add_argument("--translate", choices=["auto","romanize","off"], default="romanize")
    ap.add_argument("--translate-timeout", type=float, default=2.0)
    ap.add_argument("--translate-max", type=int, default=0, help="≤0: 무제한 (기본 0)")
    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn")
    ap.add_argument("--trace-translate", action="store_true")
    ap.add_argument("--papago-honorific", type=str2bool, nargs="?", const=True, default=True,
                    help="Papago 한국어 경어 옵션(true/false)")

    # 진행률/캐시
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--cache-file", default=None, help="번역 결과 캐시 JSON 경로")

    # 미리보기
    ap.add_argument("--show", default="true", choices=["true","false"], help="화면 미리보기 출력 (기본 true)")
    ap.add_argument("--show-limit", type=int, default=20, help="미리보기 최대 행 수 (기본 20)")
    ap.add_argument("--export-preview", default=None, help="미리보기 JSON 저장 경로")

    # DB 설정
    ap.add_argument("--config", default="config.json", help="Papago 키 및 DB 설정 포함 가능")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--charset", default=None)

    args = ap.parse_args()
    load_db = str2bool(args.load_db)

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
        print(f"[CFG][papago] honorific={args.papago_honorific} | config={args.config} | key_id={key_tail}")

        # 캐시 로드
        if args.cache_file:
            try:
                with open(args.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cnt = 0
                if isinstance(data, dict):
                    for dest, m in data.items():
                        if not isinstance(m, dict): continue
                        dest_norm = norm_dest_lang(dest)
                        for norm_text, val in m.items():
                            _trans_cache[(dest_norm, norm_text)] = val
                            cnt += 1
                print(f"[CACHE] 번역 캐시 로드: {cnt}건 from {args.cache_file}")
            except FileNotFoundError:
                print(f"[CACHE] 파일 없음(새로 생성 예정): {args.cache_file}")
            except Exception as e:
                print(f"[CACHE] 로드 실패({e}) — 무시하고 진행")
    else:
        print(f"[CFG] translate={args.translate} | provider=none | zh={args.zh_variant}  (Papago 비활성화)")

    # DB 설정 병합
    cfg_json = load_json_config(args.config, args.profile)
    cli_over = {"host": args.host, "port": args.port, "user": args.user,
                "password": args.password, "db": args.db, "charset": args.charset}
    mysql_cfg = build_effective_mysql_config(cfg_json, cli_over)
    dbg = {k: (v if k!="password" else "***") for k,v in mysql_cfg.items()}
    print(f"[CFG] MySQL: {dbg}")

    # 입력 로드 & 플랫 변환
    flat = load_basic_tree(args.basic)
    print(f"[INFO] 입력 카테고리 노드 수: {len(flat)}")

    # ---- 미리보기용 행 구성 ----
    next_order_by_parent: Dict[str, int] = defaultdict(int)
    preview_rows: List[Dict[str, Any]] = []
    total = len(flat)
    t0 = time.perf_counter()
    for i, node in enumerate(flat, 1):
        parent_path = node["parent_path"]
        nm = node["name"]
        ko = nm.get("ko","")
        en_in = nm.get("en","")
        cn_in = nm.get("cn","")

        name_obj = build_cat_name_json(
            ko, en_in, cn_in,
            translate_mode=args.translate,
            transctl=transctl,
            zh_variant=args.zh_variant,
            lang_case=args.lang_case
        )
        if node.get("display_order") is not None:
            disp = int(node["display_order"])
        else:
            next_order_by_parent[parent_path] += 10
            disp = next_order_by_parent[parent_path]

        preview_rows.append({
            "path": node["path"],
            "depth": 1 + (parent_path.count(">") + (1 if parent_path else 0)),
            "name": ko,
            "name_en": name_obj.get("en") or name_obj.get("EN",""),
            "name_cn": name_obj.get("cn") or name_obj.get("CN",""),
            "name_json": _compact_json(name_obj),
            "icon_url": node.get("icon_url") or "",
            "display_order": disp,
            "active": int(node.get("is_active", 1) or 0),
            "default": int(node.get("is_default", 0) or 0),
            "desc": (node.get("description") or "")[:80]
        })

        if (i % max(1, args.log_every) == 0) or (i == total):
            elapsed = time.perf_counter() - t0
            calls = getattr(transctl, "calls_used", None) if transctl else None
            extra = f", tx_calls={calls}" if calls is not None else ""
            print(f"[PROG] build preview {i}/{total} ({i/total*100:.1f}%) elapsed={elapsed:.1f}s{extra}")

    # 보기 좋게 정렬
    preview_rows.sort(key=lambda r: (r["path"].count(">"), r["path"]))

    # ---- 화면 미리보기 ----
    if str2bool(args.show):
        print_table(
            preview_rows,
            cols=["path","depth","name","name_en","name_cn","name_json","display_order","active","default"],
            limit=args.show_limit,
            title="카테고리 업서트 미리보기 (glossary/translate 적용)"
        )

    # ---- JSON 내보내기 ----
    if args.export_preview:
        try:
            os.makedirs(os.path.dirname(args.export_preview), exist_ok=True)
        except Exception:
            pass
        payload = {"generated_at": now_local_str(), "total": len(preview_rows), "rows": preview_rows}
        with open(args.export_preview, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] preview JSON saved: {args.export_preview}")

    # ---- DB 적재 on/off ----
    if not load_db:
        print("[DRY-RUN] DB 적재를 건너뜁니다 (--load-db=false).")
        # 캐시 저장
        if args.cache_file:
            out: Dict[str, Dict[str, str]] = {}
            for (dest_norm, norm_text), val in _trans_cache.items():
                out.setdefault(dest_norm, {})[norm_text] = val
            try:
                os.makedirs(os.path.dirname(args.cache_file), exist_ok=True)
            except Exception:
                pass
            with open(args.cache_file, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"[CACHE] 번역 캐시 저장: {len(_trans_cache)}건 -> {args.cache_file}")
        _save_papago_errors(".logs/papago_errors.json")
        _print_papago_summary()
        print(f"[TIME] 종료: {now_local_str()}")
        return

    # ---- 실제 DB 업서트 ----
    t1 = time.perf_counter()
    conn = connect_mysql(mysql_cfg)
    try:
        with conn.cursor() as cur:
            path2id: Dict[str, Optional[int]] = {"": None}
            next_disp_db: Dict[Optional[int], int] = defaultdict(int)
            name_key_for_match = "ko" if args.lang_case == "lower" else "KR"

            for i, node in enumerate(flat, 1):
                parent_path = node["parent_path"]
                parent_id = path2id.get(parent_path)

                nm = node["name"]
                ko = nm.get("ko","")
                en_in = nm.get("en","")
                cn_in = nm.get("cn","")

                name_obj = build_cat_name_json(
                    ko, en_in, cn_in,
                    translate_mode=args.translate,
                    transctl=transctl,
                    zh_variant=args.zh_variant,
                    lang_case=args.lang_case
                )
                name_json = json.dumps(name_obj, ensure_ascii=False)

                # display_order: 파일값 우선, 없으면 부모별 +10
                if node.get("display_order") is not None:
                    disp = int(node["display_order"])
                else:
                    next_disp_db[parent_id] += 10
                    disp = next_disp_db[parent_id]

                rid = ensure_category(
                    cur=cur,
                    parent_id=parent_id,
                    name_json=name_json,
                    name_key_for_match=name_key_for_match,
                    name_ko=ko,
                    icon_url=node.get("icon_url"),
                    display_order=disp,
                    is_active=int(node.get("is_active", 1) or 0),
                    is_default=int(node.get("is_default", 0) or 0),
                    description=node.get("description")
                )
                path2id[node["path"]] = rid

                if (i % max(1, args.log_every) == 0) or (i == len(flat)):
                    elapsed = time.perf_counter() - t1
                    calls = getattr(transctl, "calls_used", None) if transctl else None
                    extra = f", tx_calls={calls}" if calls is not None else ""
                    print(f"[PROG] upsert {i}/{len(flat)} ({i/len(flat)*100:.1f}%) elapsed={elapsed:.1f}s{extra}")

        conn.commit()
        print("[DONE] 커밋 완료")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C 감지 → 부분 커밋 시도")
        try:
            conn.commit(); print("[DONE] 부분 커밋 완료")
        except Exception:
            conn.rollback(); print("[ROLLBACK] 커밋 실패 → 롤백")
        finally:
            pass
        sys.exit(1)
    except Exception as e:
        conn.rollback()
        print(f"[ERR] 롤백: {e}", file=sys.stderr)
        raise
    finally:
        conn.close()
        # 캐시 저장
        if args.cache_file:
            out: Dict[str, Dict[str, str]] = {}
            for (dest_norm, norm_text), val in _trans_cache.items():
                out.setdefault(dest_norm, {})[norm_text] = val
            try:
                os.makedirs(os.path.dirname(args.cache_file), exist_ok=True)
            except Exception:
                pass
            with open(args.cache_file, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"[CACHE] 번역 캐시 저장: {len(_trans_cache)}건 -> {args.cache_file}")
        _save_papago_errors(".logs/papago_errors.json")
        _print_papago_summary()

    dt = time.perf_counter() - t1
    m = int(dt // 60); s = dt - m*60
    print(f"[TIME] 종료: {now_local_str()}")
    print(f"[TIME] 총 소요: {dt:.3f}s ({m}m {s:.3f}s)")

if __name__ == "__main__":
    main()
