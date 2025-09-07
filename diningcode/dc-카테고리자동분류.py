#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-카테고리자동분류.py — flat 카테고리 목록(categories.json) → basic 트리(JSON) 자동 생성
- Papago 번역 API 연동 (구글 라이브러리 제거)
- 캐시 파일 + 메모리 캐시 사용
- STEP/PROG 진행 로그 + Papago 연동 상태 [CFG] 로그 항상 출력
- Papago 400(N2MT05)시 원문 유지, 이후 처리 계속

예)
python3 dc-카테고리자동분류.py \
  --in-category data/category/categories.json \
  --out data/category/basic.json \
  --translate auto --translate-timeout 2.0 --translate-max 0 \
  --zh-variant cn \
  --cache-file .cache/cat-trans.json \
  --config config.json \
  --trace-translate --show true --show-limit 8 --log-every 100
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========================= 공통 유틸 =========================

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ").strip())

def now_local_str() -> str:
    import datetime
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z")

def mask_tail(s: str, keep: int = 5) -> str:
    if not s:
        return "NONE"
    tail = s[-keep:] if len(s) >= keep else s
    return f"{'*'*4}{tail}"

# ========================= 번역/로마자 =========================

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
        if not tok or tok in "/- ":
            return tok
        if tok.isupper() and len(tok) <= 4:
            return tok
        return tok[:1].upper() + tok[1:]
    return "".join(cap(p) for p in parts)

def norm_dest_lang(dest: str) -> str:
    d = (dest or "").lower().replace("_", "-")
    if d in ("en", "english"):
        return "en"
    if d in ("cn", "zh", "zh-cn", "zh-hans"):
        return "zh-CN"
    if d in ("tw", "zh-tw", "zh-hant"):
        return "zh-TW"
    return dest

# ========================= Papago HTTP =========================

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
    if total == 0:
        return
    print("\n[TX][papago] 요약")
    print(f"  - 성공 OK: {PAPAGO_STATS['ok']}")
    print(f"  - 실패 ERR: {PAPAGO_STATS['err']}")
    if PAPAGO_STATS["codes"]:
        print("  - HTTP별:", dict(PAPAGO_STATS["codes"].most_common()))
    if PAPAGO_STATS["ecodes"]:
        print("  - 에러코드별:", dict(PAPAGO_STATS["ecodes"].most_common()))

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

# ========================= 번역 컨트롤러 (Papago 고정) =========================

class TransCtl:
    """
    provider:
      - "papago" : Papago 호출(고정)
      - "none"   : 외부 번역 미사용(romanize/off 경로)
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

# ========================= 캐시 I/O =========================

def load_trans_cache(path: Optional[str]):
    if not path: return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cnt = 0
        if isinstance(data, dict):
            for dest, m in data.items():
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
    "济": "濟", "岛": "島", "归": "歸", "旧": "舊", "静": "靜",
    "馆": "館", "条": "條", "术": "術", "汉": "漢", "汤": "湯", "面": "麵",
    "酱": "醬", "团": "團", "凤": "鳳", "点": "點", "里": "裡",
})

def to_tw_if_needed(text_cn: str, zh_variant: str) -> str:
    return text_cn.translate(CN_TO_TW_TABLE) if zh_variant == "tw" else text_cn

# ========================= 사전/룰 =========================

TOP_BUCKETS = [
    ("카페·디저트", "Cafe & Dessert", "咖啡与甜品", 10),
    ("해산물",       "Seafood",         "海鲜",      20),
    ("고기/바비큐",  "Meat / BBQ",      "烧烤",      30),
    ("한식",         "Korean",          "韩国料理",  40),
    ("면/분식",      "Noodles & Snacks","面食/小吃", 50),
    ("일식",         "Japanese",        "日本料理",  60),
    ("중식",         "Chinese",         "中国料理",  70),
    ("양식",         "Western",         "西餐",      80),
    ("주점·펍",     "Pubs & Bars",     "酒馆/酒吧", 90),
]

CANON_TRANSLATIONS: Dict[str, Tuple[str, str]] = {
    # (중략) — 동일: 표준 번역 테이블 (이전 버전과 동일)
    "카페": ("Cafe", "咖啡店"), "커피": ("Coffee", "咖啡"), "아메리카노": ("Americano", "美式咖啡"),
    "라떼": ("Latte", "拿铁"), "에스프레소": ("Espresso", "浓缩咖啡"), "콜드브루": ("Cold Brew", "冷萃咖啡"),
    "드립커피": ("Pour-over", "手冲咖啡"), "디저트": ("Dessert", "甜点"), "브런치": ("Brunch", "早午餐"),
    "베이커리": ("Bakery", "烘焙店"), "도넛": ("Donut", "甜甜圈"), "케이크": ("Cake", "蛋糕"),
    "베이글": ("Bagel", "百吉饼"), "소금빵": ("Salt Bread", "盐面包"), "크로플": ("Croffle", "可颂华夫"),
    "와플": ("Waffle", "华夫饼"), "빙수": ("Bingsu", "刨冰"), "아이스크림": ("Ice Cream", "冰淇淋"),
    "젤라또": ("Gelato", "意式冰淇淋"), "땅콩아이스크림": ("Peanut Ice Cream", "花生冰淇淋"),
    "북카페": ("Book Cafe", "书店咖啡"), "루프탑": ("Rooftop Cafe", "露台咖啡"),
    # Seafood ...
    "회/활어회": ("Sashimi / Raw Fish", "生鱼片"),
    "활어회": ("Live Sashimi", "活鱼生鱼片"),
    "숙성회": ("Aged Sashimi", "熟成生鱼片"),
    "모듬회": ("Assorted Sashimi", "综合生鱼片"),
    "고등어회": ("Mackerel Sashimi", "青花鱼生鱼片"),
    "방어회": ("Amberjack Sashimi", "鰤鱼生鱼片"),
    "생선구이": ("Grilled Fish", "烤鱼"),
    "갈치구이": ("Grilled Hairtail", "烤带鱼"),
    "고등어구이": ("Grilled Mackerel", "烤青花鱼"),
    "옥돔구이": ("Grilled Tilefish", "烤条石鲷"),
    "갈치요리": ("Hairtail Dishes", "带鱼料理"),
    "갈치조림": ("Braised Hairtail", "炖带鱼"),
    "갈치국": ("Hairtail Soup", "带鱼汤"),
    "물회": ("Mulhoe (Cold Raw Fish Soup)", "凉拌生鱼汤"),
    "전복요리": ("Abalone Dishes", "鲍鱼料理"),
    "전복죽": ("Abalone Porridge", "鲍鱼粥"),
    "전복돌솥밥": ("Abalone Hot-pot Rice", "鲍鱼石锅拌饭"),
    "전복솥밥": ("Abalone Pot Rice", "鲍鱼砂锅饭"),
    "전복구이": ("Grilled Abalone", "烤鲍鱼"),
    "해물탕/뚝배기": ("Seafood Stew / Earthen Pot", "海鲜汤/砂锅"),
    "해물탕": ("Seafood Hot Pot", "海鲜汤"),
    "해물뚝배기": ("Seafood Earthen Pot", "海鲜砂锅"),
    "딱새우": ("Sweet Shrimp", "甜虾"),
    "딱새우회": ("Sweet Shrimp Sashimi", "甜虾生鱼片"),
    "성게/보말": ("Sea Urchin / Top Shell", "海胆/法螺"),
    "성게비빔밥": ("Sea Urchin Bibimbap", "海胆拌饭"),
    "성게미역국": ("Sea Urchin Seaweed Soup", "海胆海带汤"),
    "보말죽": ("Top Shell Porridge", "法螺粥"),
    "보말칼국수": ("Top Shell Kalguksu", "法螺刀削面"),
    # Meat/BBQ ...
    "흑돼지": ("Jeju Black Pork", "济州黑猪"),
    "흑돼지오겹": ("Black Pork Belly", "黑猪五花"),
    "흑돼지근고기": ("Thick-cut Black Pork", "厚切黑猪肉"),
    "흑돼지구이": ("Grilled Black Pork", "黑猪烤肉"),
    "흑돼지돈까스": ("Black Pork Cutlet", "黑猪猪排"),
    "흑돼지불고기": ("Black Pork Bulgogi", "黑猪烤肉（韩式）"),
    "돼지고기구이": ("Pork BBQ", "猪肉烤肉"),
    "삼겹살": ("Samgyeopsal", "五花肉"),
    "오겹살": ("Ogeopsal", "五花三层肉"),
    "돼지갈비": ("Pork Ribs", "猪排骨"),
    "근고기": ("Thick-cut Pork", "厚切猪肉"),
    "소고기구이": ("Beef BBQ", "牛肉烤肉"),
    "양념갈비": ("Marinated Ribs", "腌制排骨"),
    "우대갈비": ("Premium Beef Ribs", "厚切牛排骨"),
    "소갈비살": ("Beef Rib Finger", "牛肋条"),
    "정육식당": ("Butcher's BBQ", "肉铺直烤"),
    "돔베고기": ("Dombae Pork (Sliced)", "切片猪肉"),
    # Korean ...
    "백반/한정식": ("Set Meal / Hansik Course", "韩式套餐/韩定食"),
    "백반": ("Set Meal (Baekban)", "家庭套餐"),
    "한정식": ("Korean Course Meal", "韩式定食"),
    "국/탕/찌개": ("Soups & Stews", "汤/锅/炖"),
    "국밥": ("Gukbap", "汤饭"),
    "해장국": ("Haejang-guk", "解酒汤"),
    "김치찌개": ("Kimchi Stew", "泡菜锅"),
    "순두부": ("Soft Tofu Stew", "嫩豆腐锅"),
    "돼지국밥": ("Pork Gukbap", "猪肉汤饭"),
    "소고기해장국": ("Beef Hangover Soup", "牛肉解酒汤"),
    "비빔밥/볶음": ("Bibimbap / Stir-fry", "拌饭/炒菜"),
    "비빔밥": ("Bibimbap", "拌饭"),
    "제육볶음": ("Spicy Pork Stir-fry", "辣炒猪肉"),
    "제주향토": ("Jeju Local Dishes", "济州乡土料理"),
    "고기국수": ("Pork Noodle Soup", "猪肉面"),
    "몸국": ("Momguk (Seaweed Pork Soup)", "海藻猪肉汤"),
    # Noodles & Snacks ...
    "면/분식": ("Noodles & Snacks", "面食/小吃"),
    "국수/칼국수": ("Noodles / Kalguksu", "面/刀削面"),
    "국수": ("Noodles", "面"),
    "칼국수": ("Kalguksu", "刀削面"),
    "메밀국수": ("Buckwheat Noodles", "荞麦面"),
    "우동": ("Udon", "乌冬面"),
    "라면": ("Ramen (Korean Style)", "泡面"),
    "분식": ("Korean Snack Foods", "韩式小吃"),
    "떡볶이": ("Tteokbokki", "炒年糕"),
    "김밥": ("Gimbap", "紫菜包饭"),
    "만두": ("Dumplings", "饺子"),
    "중화면": ("Chinese-style Noodles", "中式面"),
    "짬뽕": ("Jjamppong", "什锦海鲜面"),
    "짜장면": ("Jjajangmyeon", "炸酱面"),
    "탕수육": ("Sweet & Sour Pork", "糖醋肉"),
    # Japanese ...
    "일식": ("Japanese", "日本料理"),
    "스시/오마카세": ("Sushi / Omakase", "寿司/主厨套餐"),
    "스시": ("Sushi", "寿司"),
    "초밥": ("Sushi (Nigiri)", "寿司（握寿司）"),
    "오마카세": ("Omakase", "主厨精选"),
    "라멘/우동/덮밥": ("Ramen / Udon / Donburi", "拉面/乌冬/盖饭"),
    "라멘": ("Ramen", "拉面"),
    "덮밥": ("Donburi", "盖饭"),
    "규동": ("Beef Bowl (Gyudon)", "牛肉盖饭"),
    "텐동": ("Tendon (Tempura Rice Bowl)", "天丼"),
    "이자카야": ("Izakaya", "居酒屋"),
    "일본가정식": ("Japanese Home-style", "日本家常菜"),
    # Chinese ...
    "중식": ("Chinese", "中国料理"),
    "중식당": ("Chinese Restaurant", "中餐馆"),
    "마라탕": ("Mala Soup", "麻辣烫"),
    # Western ...
    "양식": ("Western", "西餐"),
    "파스타": ("Pasta", "意面"),
    "피자": ("Pizza", "披萨"),
    "수제버거": ("Handmade Burger", "手工汉堡"),
    "햄버거": ("Burger", "汉堡"),
    "스테이크": ("Steak", "牛排"),
    "비스트로": ("Bistro", "小酒馆"),
    # Pubs & Bars ...
    "주점·펍": ("Pubs & Bars", "酒馆/酒吧"),
    "술집": ("Bar", "小酒吧"),
    "펍": ("Pub", "啤酒屋"),
    "요리주점": ("Gastro Pub", "料理酒馆"),
    "포차": ("Pocha (K-Food Bar)", "路边摊酒馆"),
    "와인바": ("Wine Bar", "葡萄酒吧"),
    "맥주": ("Beer", "啤酒"),
    "하이볼": ("Highball", "高球酒"),
}

RULES: List[Tuple[str, Optional[str], str]] = [
    ("카페·디저트", "커피", r"(카페|커피|에스프레소|라떼|아메리카노|콜드브루|드립)"),
    ("카페·디저트", "베이커리", r"(베이커리|빵|도넛|케이크|크로플|와플|베이글|소금빵)"),
    ("카페·디저트", "빙수/아이스크림", r"(빙수|아이스크림|젤라또|젤라토|땅콩아이스크림)"),
    ("카페·디저트", None, r"(브런치|디저트|북카페|루프탑)"),
    ("해산물", "회/활어회", r"(회|활어회|숙성회|모듬회|고등어회|방어회)"),
    ("해산물", "생선구이", r"(생선구이|갈치구이|고등어구이|옥돔구이)"),
    ("해산물", "갈치요리", r"(갈치조림|갈치국)"),
    ("해산물", None, r"(물회|전복|해물탕|뚝배기|딱새우|성게|보말|해물|조개|문어|낙지|오징어)"),
    ("고기/바비큐", "흑돼지", r"(흑돼지)"),
    ("고기/바비큐", "돼지고기구이", r"(삼겹살|오겹살|돼지갈비|근고기)"),
    ("고기/바비큐", "소고기구이", r"(소고기구이|양념갈비|우대갈비|소갈비살)"),
    ("고기/바비큐", None, r"(정육식당|바비큐|BBQ|숯불|돔베고기)"),
    ("한식", "백반/한정식", r"(백반|한정식)"),
    ("한식", "국/탕/찌개", r"(국밥|해장국|김치찌개|순두부|돼지국밥|소고기해장국|전골|탕|찌개)"),
    ("한식", "비빔밥/볶음", r"(비빔밥|제육볶음)"),
    ("한식", "제주향토", r"(고기국수|몸국)"),
    ("면/분식", "국수/칼국수", r"(국수|칼국수|메밀국수|우동|라면(?!\s*/\s*우동))"),
    ("면/분식", "분식", r"(분식|떡볶이|김밥|만두)"),
    ("면/분식", "중화면", r"(짬뽕|짜장면|탕수육)"),
    ("일식", "스시/오마카세", r"(스시|초밥|오마카세)"),
    ("일식", "라멘/우동/덮밥", r"(라멘|우동(?!.*국수)|덮밥|규동|텐동)"),
    ("일식", None, r"(이자카야|일본가정식)"),
    ("중식", None, r"(중식|중식당|짬뽕|짜장면|탕수육|마라탕)"),
    ("양식", None, r"(양식|파스타|피자|수제버거|스테이크|비스트로)"),
    ("주점·펍", None, r"(술집|펍|요리주점|포차|와인바|맥주|하이볼)"),
]

# ========================= 입력/출력 =========================

def load_categories(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        out = []
        for it in doc:
            if isinstance(it, dict) and "category" in it:
                out.append({"category": _clean_ws(it["category"]), "count": int(it.get("count", 0) or 0)})
            elif isinstance(it, str):
                out.append({"category": _clean_ws(it), "count": 0})
        return out
    raise ValueError("in-category JSON은 배열이어야 하며 각 원소는 {category, count} 또는 문자열이어야 합니다.")

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] 저장: {path}")

# ========================= EN/CN 채우기 =========================

def to_tw_if_needed(text_cn: str, zh_variant: str) -> str:
    return text_cn.translate(CN_TO_TW_TABLE) if zh_variant == "tw" else text_cn

def _dict_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if not ko: return (None, None)
    pair = CANON_TRANSLATIONS.get(ko)
    if not pair: return (None, None)
    en, cn = pair
    en = titlecase_en(en)
    cn = to_tw_if_needed(cn, zh_variant)
    return (en, cn)

def _slash_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if "/" not in ko: return (None, None)
    parts = [p.strip() for p in ko.split("/")]
    en_parts, cn_parts = [], []
    for p in parts:
        en_p, cn_p = _dict_lookup(p, zh_variant)
        en_parts.append(en_p or p)
        cn_parts.append(cn_p or p)
    en = " / ".join(titlecase_en(x) for x in en_parts)
    cn = " / ".join(cn_parts)
    return (en, to_tw_if_needed(cn, zh_variant))

def fill_langs(ko: str, en_in: Optional[str], cn_in: Optional[str],
               fill_policy: str, translate_mode: str,
               transctl: Optional[TransCtl], zh_variant: str) -> Dict[str, str]:
    ko = (ko or "").strip()
    en = (en_in or "").strip()
    cn = (cn_in or "").strip()

    def _rr_or_copy(s: str) -> str:
        return romanize_korean(s) if has_hangul(s) else (s or ko)

    def _tx_en(s: str) -> str:
        if translate_mode == "off":
            return ""
        if translate_mode == "romanize" or not transctl:
            return _rr_or_copy(s)
        t = transctl.translate(s, "en")
        return (t or _rr_or_copy(s)).strip()

    def _tx_cn(s: str) -> str:
        if translate_mode == "off":
            return ""
        if translate_mode == "romanize" or not transctl:
            return _rr_or_copy(s)
        dest = "zh-TW" if zh_variant == "tw" else "zh-CN"
        t = transctl.translate(s, dest)
        return (t or _rr_or_copy(s)).strip()

    # 1) 사전 우선
    if not en or not cn:
        en_d, cn_d = _dict_lookup(ko, zh_variant)
        if en_d and not en: en = en_d
        if cn_d and not cn: cn = cn_d

    # 2) 슬래시 복합
    if (not en or not cn) and "/" in ko:
        en_s, cn_s = _slash_lookup(ko, zh_variant)
        if en_s and not en: en = en_s
        if cn_s and not cn: cn = cn_s

    # 3) 폴백(translate/romanize/off)
    if not en:
        if fill_policy == "translate": en = _tx_en(ko)
        elif fill_policy == "romanize": en = _rr_or_copy(ko)
        else: en = ""
    if not cn:
        if fill_policy == "translate": cn = _tx_cn(ko)
        elif fill_policy == "romanize": cn = _rr_or_copy(ko)
        else: cn = ""

    en = titlecase_en(en)
    cn = to_tw_if_needed(cn, zh_variant)
    return {"ko": ko, "en": en, "cn": cn}

# ========================= 트리 빌더 =========================

def make_node(ko: str, fill_policy: str, translate_mode: str,
              transctl: Optional[TransCtl], zh_variant: str,
              display_order: Optional[int] = None) -> Dict[str, Any]:
    name_json = fill_langs(ko, None, None, fill_policy, translate_mode, transctl, zh_variant)
    node = {
        "name": name_json,
        "icon_url": None,
        "is_active": 1,
        "is_default": 0,
    }
    if display_order is not None:
        node["display_order"] = display_order
    return node

def classify_category(token: str) -> Tuple[str, Optional[str]]:
    s = _clean_ws(token)
    for bucket, sub, patt in RULES:
        if re.search(patt, s):
            return (bucket, sub)
    return ("기타", None)

def build_tree(flat: List[Dict[str, Any]],
               fill_policy: str, translate_mode: str,
               transctl: Optional[TransCtl], zh_variant: str,
               log_every: int) -> List[Dict[str, Any]]:
    # 루트 버킷(표시순서 고정)
    roots: List[Dict[str, Any]] = []
    bucket_map: Dict[str, Dict[str, Any]] = {}
    for ko, en, cn, disp in TOP_BUCKETS:
        node = {
            "name": {"ko": ko, "en": titlecase_en(en), "cn": to_tw_if_needed(cn, zh_variant)},
            "icon_url": None,
            "is_active": 1,
            "is_default": 0,
            "display_order": disp,
            "children": [],
        }
        roots.append(node)
        bucket_map[ko] = node
    misc_bucket: Optional[Dict[str, Any]] = None

    next_order: Dict[Tuple[str, Optional[str]], int] = defaultdict(lambda: 10)
    def _next(dkey: Tuple[str, Optional[str]]) -> int:
        v = next_order[dkey]
        next_order[dkey] = v + 10
        return v

    total = len(flat)
    t0 = time.perf_counter()
    print(f"[INFO] 분류 대상: {total}개 (count 내림차순)")

    for i, rec in enumerate(sorted(flat, key=lambda x: -int(x.get("count", 0) or 0)), 1):
        tok = rec["category"]
        if not tok:
            continue

        # STEP: 진행상태
        calls = getattr(transctl, "calls_used", None) if transctl else None
        extra = f" tx_calls={calls}" if calls is not None else ""
        print(f"[STEP] {i}/{total} token='{tok}'{extra}")

        bucket_ko, sub_ko = classify_category(tok)

        if bucket_ko == "기타":
            if misc_bucket is None:
                misc_bucket = make_node("기타", fill_policy, translate_mode, transctl, zh_variant, display_order=999)
                misc_bucket["children"] = []
                roots.append(misc_bucket)
            bucket_node = misc_bucket
        else:
            bucket_node = bucket_map[bucket_ko]

        if sub_ko:
            # 서브 노드
            sub_node = None
            for c in bucket_node["children"]:
                if (c.get("name", {}) or {}).get("ko") == sub_ko:
                    sub_node = c; break
            if sub_node is None:
                sub_node = make_node(sub_ko, fill_policy, translate_mode, transctl, zh_variant, display_order=_next((bucket_ko, None)))
                sub_node["children"] = []
                bucket_node["children"].append(sub_node)

            # 리프
            exists = any((ch.get("name", {}) or {}).get("ko") == tok for ch in sub_node["children"])
            if not exists and tok != sub_ko:
                leaf = make_node(tok, fill_policy, translate_mode, transctl, zh_variant, display_order=_next((bucket_ko, sub_ko)))
                sub_node["children"].append(leaf)
        else:
            # 버킷 직속 리프
            exists = any((ch.get("name", {}) or {}).get("ko") == tok for ch in bucket_node["children"])
            if not exists and tok != bucket_ko:
                leaf = make_node(tok, fill_policy, translate_mode, transctl, zh_variant, display_order=_next((bucket_ko, None)))
                bucket_node["children"].append(leaf)

        # PROG 로그
        if (i % max(1, log_every) == 0) or (i == total):
            elapsed = time.perf_counter() - t0
            pct = (i / total * 100.0) if total else 100.0
            calls = getattr(transctl, "calls_used", None) if transctl else None
            extra = f", tx_calls={calls}" if calls is not None else ""
            print(f"[PROG] classify {i}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s{extra} last='{tok}'")

    # 정렬(표시순서)
    for b in roots:
        if "children" in b and isinstance(b["children"], list):
            b["children"].sort(key=lambda x: int(x.get("display_order", 0)))
            for c in b["children"]:
                if "children" in c and isinstance(c["children"], list):
                    c["children"].sort(key=lambda x: int(x.get("display_order", 0)))
    roots.sort(key=lambda x: int(x.get("display_order", 0)))
    return roots

# ========================= 미리보기 =========================

def print_preview(tree: List[Dict[str, Any]], limit: int):
    print("\n[PREVIEW] basic 트리 (name | name_json)")
    for root in tree:
        rname = root["name"]
        print(f"- {rname['ko']}  |  {rname}")
        children = root.get("children") or []
        shown = 0
        for c in children:
            cname = c["name"]
            print(f"  • {cname['ko']}  |  {cname}")
            shown += 1
            gkids = c.get("children") or []
            for g in gkids[:max(3, limit - 2)]:
                gname = g["name"]
                print(f"     - {gname['ko']}  |  {gname}")
            if shown >= limit:
                break

# ========================= CLI =========================

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def main():
    ap = argparse.ArgumentParser(description="flat 카테고리 → basic 트리 자동 분류 생성기 (Papago 번역/캐시/진행로그)")
    ap.add_argument("--in-category", required=True, help="입력 flat 카테고리 JSON (예: data/category/categories.json)")
    ap.add_argument("--out", required=True, help="출력 basic 트리 JSON (예: data/category/basic.json)")

    # 번역/채우기 옵션
    ap.add_argument("--fill-unknown-lang", choices=["translate","romanize","off"], default="translate",
                    help="미번역 항목(en/cn) 채우기 정책 (기본 translate)")
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

    # 호환(무시)
    ap.add_argument("--translate-provider", default=None,
                    help="(deprecated) 무시됩니다. Papago 고정입니다.")

    # 미리보기/진행 로그
    ap.add_argument("--show", choices=["true","false"], default="true")
    ap.add_argument("--show-limit", type=int, default=8, help="버킷당 미리보기 자식 표시 제한")
    ap.add_argument("--log-every", type=int, default=100, help="N건마다 진행 상황 로그 출력 (기본 100)")

    args = ap.parse_args()
    show = (args.show.lower() == "true")

    print(f"[TIME] 시작: {now_local_str()}")

    if args.translate_provider:
        print(f"[WARN] --translate-provider 는 더 이상 사용하지 않습니다(Papago 고정). 입력값 '{args.translate_provider}'는 무시됩니다.")

    # 번역 컨트롤러
    tmax = None if args.translate_max is None or args.translate_max <= 0 else int(args.translate_max)
    transctl = None
    if args.translate == "auto":
        transctl = TransCtl(
            use_papago=True,
            timeout=args.translate_timeout,
            max_calls=tmax,                         # None이면 무제한
            zh_variant=args.zh_variant,
            trace=args.trace_translate,
            config_path=args.config,
            papago_honorific=args.papago_honorific,
        )
        max_str = "∞" if tmax is None else str(tmax)
        print(f"[CFG] translate=auto | provider=papago | zh={args.zh_variant} | timeout={args.translate_timeout}s | max={max_str}")
        key_tail = transctl.papago.key_tail() if transctl and transctl.papago else "NONE"
        print(f"[CFG][papago] honorific={args.papago_honorific} | config={args.config} | key_id={key_tail}")
    else:
        print(f"[CFG] translate={args.translate} | provider=none | zh={args.zh_variant}  (Papago 비활성화)")

    # 캐시 로드
    load_trans_cache(args.cache_file)

    # 입력 로드
    flat = load_categories(args.in_category)
    print(f"[INFO] 입력 카테고리 수: {len(flat)} (예: {flat[:3]})")

    # 트리 빌드 (STEP/PROG 포함)
    tree = build_tree(
        flat=flat,
        fill_policy=args.fill_unknown_lang,
        translate_mode=args.translate,
        transctl=transctl,
        zh_variant=args.zh_variant,
        log_every=args.log_every,
    )

    # 저장
    save_json(tree, args.out)

    # 미리보기
    if show:
        print_preview(tree, limit=max(3, args.show_limit))

    # 캐시/에러로그/요약 저장
    save_trans_cache(args.cache_file)
    _save_papago_errors(args.tx_error_log)
    _print_papago_summary()

    print(f"[TIME] 종료: {now_local_str()}")

if __name__ == "__main__":
    main()
