#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-상세수집.py — DiningCode 상세/메뉴 수집기
- Papago 번역 + 캐시(HIT 시 API 생략, 로그 끝에 [CACHE] 표기)
- 입력이 이미 목표언어면 Papago 호출 스킵(로그 끝에 [SKIP] 표기)
- 400(N2MT05: source=target) → 원문 그대로 사용(IDENTITY) + 캐시 저장 + 진행 지속
- 400일 때 RAW 응답 미리보기(콘솔) + JSON 로그(raw_body 최대 4KB)
- translate_max 기본 무제한(≤0 이면 무제한)
- 입력 파일별/전체 RID 수 표시 + 각 RID 처리 순번 로그([STEP] i/total)

예시)
python3 dc-상세수집.py \
  --files "20250830-merged-list.json" \
  --translate auto \
  --translate-timeout 8.0 \
  --translate-max 0 \
  --zh-variant cn \
  --config config.json \
  --cache-file .cache/restaurant-trans.json \
  --tx-error-log .logs/papago_errors.json \
  --trace-translate true \
  --show true
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import Counter
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------- 정규식/유틸 --------------------
PRICE_PAT = re.compile(r"([0-9][0-9,\.]*)\s*원")
CLEAN_WS = re.compile(r"\s+")
RE_HANGUL = re.compile(r"[가-힣]")
RE_HAN = re.compile(r"[\u4E00-\u9FFF]")  # CJK Unified Ideographs

def now_local_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z")

def _clean(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = CLEAN_WS.sub(" ", s).strip()
    return s

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def _to_price_int(price_text: Optional[str]) -> Optional[int]:
    if not price_text:
        return None
    m = PRICE_PAT.search(price_text)
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace(".", "")
    try:
        return int(raw)
    except Exception:
        return None

def _mk_url(rid: str) -> str:
    return f"https://www.diningcode.com/profile.php?rid={rid}"

# === 라이트 언어 감지 및 대상 언어 정규화 ===
def detect_lang_simple(text: str) -> str:
    """
    매우 가벼운 감지: 'ko'/'zh'/'en'/'unknown'
    - 한글 있으면 'ko'
    - 한자 있으면 'zh'
    - 90% 이상 ASCII + 알파벳 존재하면 'en'
    """
    s = (text or "").strip()
    if not s:
        return "unknown"
    if RE_HANGUL.search(s):
        return "ko"
    if RE_HAN.search(s):
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

# -------------------- 파일에서 v_rid 수집 --------------------
def _iter_dicts(x: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _iter_dicts(v)
    elif isinstance(x, list):
        for it in x:
            yield from _iter_dicts(it)

def collect_vrids_from_file(path: str) -> List[str]:
    """입력 JSON 내 모든 위치에서 v_rid를 모아 반환."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"[WARN] JSON 로드 실패: {path} ({e})", file=sys.stderr)
        return []
    vrids: List[str] = []
    seen: Set[str] = set()
    for d in _iter_dicts(doc):
        rid = d.get("v_rid")
        if isinstance(rid, str):
            rid = rid.strip()
        if isinstance(rid, str) and rid and rid not in seen:
            seen.add(rid); vrids.append(rid)
    return vrids

def gather_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    uniq, seen = [], set()
    for p in files:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

# -------------------- HTTP --------------------
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

def fetch_html(url: str, timeout: float = 8.0) -> str:
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "ko,ko-KR;q=0.9,en;q=0.8",
        "Referer": "https://www.diningcode.com/",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

# -------------------- 파서 --------------------
def parse_store_name(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    el = soup.select_one("div.tit-point h1.tit") or soup.select_one("h1.tit")
    if el:
        name = _clean(el.get_text(" ", strip=True))
        if name:
            return name
    meta = soup.select_one('meta[property="og:title"]')
    if meta:
        t = _clean(meta.get("content") or "")
        if t:
            t = re.split(r"\s*[|\-–]\s*", t)[0]
            return t
    if soup.title and soup.title.string:
        t = _clean(soup.title.string)
        if t:
            t = re.split(r"\s*[|\-–]\s*", t)[0]
            return t
    return ""

def parse_menuinfo_items(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    containers = soup.select("div.menu-info")
    if not containers:
        return []
    items: List[Dict[str, Any]] = []

    def push_from_text(raw: str):
        txt = _clean(raw)
        if not txt:
            return
        recommended = "추천" in txt
        m = PRICE_PAT.search(txt)
        price_text = m.group(0) if m else ""
        price_krw = _to_price_int(price_text)

        name = txt
        if price_text:
            name = name.replace(price_text, " ")
        name = name.replace("추천", " ")
        name = name.replace("접기", " ").replace("펴기", " ")
        name = re.sub(r"[·•⋯·\.]{2,}", " ", name)
        name = _clean(name)
        if not name:
            return

        items.append({
            "name": name,
            "price_text": price_text,
            "price_krw": price_krw,
            "recommended": bool(recommended),
        })

    for box in containers:
        for ul in box.find_all(["ul", "ol"]):
            for li in ul.find_all("li"):
                push_from_text(li.get_text(" ", strip=True))
        for dd in box.find_all("dd"):
            push_from_text(dd.get_text(" ", strip=True))
        for tr in box.find_all("tr"):
            cells = tr.find_all(["td","th"])
            if not cells:
                continue
            row_txt = " ".join(c.get_text(" ", strip=True) for c in cells)
            if PRICE_PAT.search(row_txt):
                push_from_text(row_txt)
            else:
                push_from_text(cells[0].get_text(" ", strip=True))

    # dedupe
    seen = set(); deduped=[]
    for it in items:
        key = (it["name"], it["price_text"] or it["price_krw"])
        if key in seen: continue
        seen.add(key); deduped.append(it)
    return deduped

def parse_textinfo(html: str) -> Dict[str, List[str]]:
    soup = BeautifulSoup(html, "lxml")
    btxt, tags, chars = [], [], []
    for div in soup.select("div.btxt"):
        t = _clean(div.get_text(" ", strip=True))
        if t: btxt.append(t)
    for li in soup.select("li.tag"):
        t = _clean(li.get_text(" ", strip=True))
        if t: tags.append(t)
    for li in soup.select("li.char"):
        t = _clean(li.get_text(" ", strip=True))
        if t: chars.append(t)

    def dedup(seq: List[str]) -> List[str]:
        seen = set(); out=[]
        for s in seq:
            if s not in seen:
                seen.add(s); out.append(s)
        return out

    return {"btxt": dedup(btxt), "tags": dedup(tags), "chars": dedup(chars)}

def compose_description(btxt: List[str], tags: List[str], chars: List[str]) -> str:
    parts: List[str] = []
    if btxt: parts.append(" ".join(btxt))
    if tags: parts.append(", ".join(tags))
    if chars: parts.append(", ".join(chars))
    return _clean(" | ".join([p for p in parts if p.strip()]))

def compose_short_description(btxt: List[str], tags: List[str]) -> str:
    parts: List[str] = []
    if btxt: parts.append(" ".join(btxt))
    if tags: parts.append(", ".join(tags))
    return _clean(" | ".join([p for p in parts if p.strip()]))

# -------------------- 번역/로마자/캐시 --------------------
def _norm_key(s: str) -> str:
    return CLEAN_WS.sub(" ", (s or "").strip()).lower()

_L = ["g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"]
_V = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo","u","wo","we","wi","yu","eu","ui","i"]
_T = ["","k","k","ks","n","nj","nh","t","l","lk","lm","lb","ls","lt","lp","lh","m","p","ps","t","t","ng","t","t","k","t","p","t"]

_roman_cache: Dict[str, str] = {}
# 캐시 키는 (norm_dest_lang(dest), _norm_key(text))
_trans_cache: Dict[Tuple[str, str], str] = {}

def has_hangul(s: str) -> bool:
    return bool(RE_HANGUL.search(s or ""))

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

# -------------------- Papago 에러 로깅/통계 --------------------
PAPAGO_ERRORS: List[Dict[str, Any]] = []
PAPAGO_STATS = {"ok": 0, "err": 0, "codes": Counter(), "ecodes": Counter()}

def _log_papago_error(event: Dict[str, Any], trace: bool):
    """event: {ts,status,errorCode,errorMessage,target,src,len,headers_snippet,raw_body?}"""
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
        print("  - HTTP 상태코드별:", dict(PAPAGO_STATS["codes"].most_common()))
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

# -------------------- Papago 클라이언트 --------------------
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
        """
        Papago 오류 응답은 케이스별로 형태가 다름:
          - {"error":{"errorCode":"N2MT05","message":"..."}}  ← 샘플
          - {"errorCode":"...", "errorMessage":"..."}
        안전하게 errorCode, errorMessage를 뽑아줌.
        """
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

            # ── 성공 ──────────────────────────────────────────────────────────
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
                # 200인데 결과 없음
                event = {
                    "ts": now_local_str(), "status": status, "errorCode": "EMPTY_RESULT",
                    "errorMessage": "translatedText empty", "target": payload["target"],
                    "src": payload["source"], "len": len(text),
                    "headers_snippet": {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-")}
                }
                _log_papago_error(event, self.trace)
                return ""

            # ── 실패(400/401/403/429/5xx 등) ──────────────────────────────────
            err_json, err_text = {}, ""
            try:
                err_json = r.json()
            except Exception:
                err_text = r.text

            # 안전 파싱
            ecode, emsg = self._parse_error_payload(err_json)

            # 400 전용: RAW 본문 미리보기 + JSON 로그 raw_body 저장
            RAW_MAX = 4096
            raw_body = None
            if status == 400:
                raw_body = (r.text or "")[:RAW_MAX]
                preview = raw_body[:600].replace("\n", " ")
                print(f"[TX][papago][400] target={payload['target']} len={len(text)} RAW({len(raw_body)}B) = {preview}")

                # N2MT05: source=target → 원문 그대로 반환, 에러로 집계하지 않음(진행 지속)
                blob = (preview or "").lower()
                if (ecode == "N2MT05") or ("source and target must be different" in blob):
                    if self.trace:
                        print(f"[TX][papago][IDENTITY] source==target; return original (len={len(text)})")
                    _log_papago_ok()  # OK 집계로 처리
                    return text

            # 그 외 에러: 집계/저장
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
                "errorMessage": str(e), "target": payload["target"], "src": payload["source"], "len": len(text),
                "headers_snippet": {}
            }
            _log_papago_error(event, self.trace)
            return ""

# -------------------- 번역 컨트롤 --------------------
class TransCtl:
    """
    provider:
      - "papago"     : Papago 호출
      - "googletrans": 레거시(옵션)
      - "deep"       : 레거시(옵션)
      - "auto_chain" : googletrans→deep 폴백(레거시)
      - "none"       : 외부 번역 미사용
    """
    def __init__(self,
                 provider: str,
                 timeout: float,
                 max_calls: Optional[int],
                 zh_variant: str,
                 trace: bool = False,
                 config_path: str = "config.json",
                 papago_honorific: bool = True):
        self.provider = provider
        self.timeout = timeout
        self.max_calls = max_calls  # None이면 무제한
        self.calls_used = 0
        self.zh_variant = "zh-TW" if zh_variant == "tw" else "zh-CN"
        self.trace = trace

        self.papago_honorific = papago_honorific
        self.papago = PapagoClient(config_path=config_path, timeout=timeout, trace=trace) if provider == "papago" else None

    def _can_call(self) -> bool:
        return (self.max_calls is None) or (self.calls_used < self.max_calls)

    def translate(self, text: str, dest: str) -> Optional[str]:
        # --- 캐시 조회(정규화된 대상언어 + 정규화 텍스트) ---
        dest_norm = norm_dest_lang(dest)
        key = (dest_norm, _norm_key(text))
        if key in _trans_cache:
            if self.trace:
                # 캐시 사용을 OK 라인 포맷으로, 우측에 [CACHE] 태그
                print(f"[TX][papago][OK] len={len(text)} → {dest_norm} [CACHE]")
            return _trans_cache[key]

        # --- 사전 스킵(입력언어 = 타깃언어) ---
        lang = detect_lang_simple(text)
        if (dest_norm == "en" and lang == "en") or \
           (dest_norm in ("zh-CN", "zh-TW") and lang == "zh"):
            if self.trace:
                # 스킵도 OK 포맷으로 표기하여 흐름 일관성 유지, 우측 [SKIP]
                print(f"[TX][papago][OK] len={len(text)} → {dest_norm} [SKIP]")
            _log_papago_ok()  # 성공으로 간주
            _trans_cache[key] = text  # 캐시에 저장하여 이후 호출 생략
            return text

        # 호출 한도 체크(기본 무제한)
        if not self._can_call():
            if self.trace:
                print(f"[TX] translate_max reached ({self.max_calls}); further texts will be kept as original.")
            return None

        out: Optional[str] = None

        # 1) Papago
        if self.provider == "papago":
            self.calls_used += 1
            if self.trace: print(f"[TX] papago → {dest_norm}: len={len(text)}")
            try:
                out = self.papago.translate(text, target=dest_norm, source="auto", honorific=self.papago_honorific)
            except Exception as e:
                _log_papago_error({
                    "ts": now_local_str(), "status": None, "errorCode": "EXCEPTION",
                    "errorMessage": f"TransCtl: {e}", "target": dest_norm, "src": "auto", "len": len(text),
                    "headers_snippet": {}
                }, trace=self.trace)
                out = None

        # 2) 레거시 제공자 (옵션 유지)
        if not out and self.provider in ("googletrans", "auto_chain"):
            self.calls_used += 1
            if self.trace: print(f"[TX] googletrans → {dest_norm}: len={len(text)}")
            try:
                from googletrans import Translator  # type: ignore
                tr = Translator()
                res = tr.translate(text, src="ko", dest=dest_norm)
                out = res.text if getattr(res, "text", None) else None
            except Exception:
                out = None

        if not out and self.provider in ("deep", "auto_chain"):
            if not self._can_call():
                return None
            self.calls_used += 1
            if self.trace: print(f"[TX] deep_translator → {dest_norm}: len={len(text)}")
            try:
                from deep_translator import GoogleTranslator  # type: ignore
                tr = GoogleTranslator(source="ko", target=dest_norm)
                out = tr.translate(text)
            except Exception:
                out = None

        if out:
            _trans_cache[key] = out  # 성공/IDENTITY 결과 모두 캐시에 저장됨
        return out

# -------------------- EN TitleCase / CN 번체 처리 --------------------
CN_TO_TW_TABLE = str.maketrans({
    "济":"濟","岛":"島","归":"歸","旧":"舊","静":"靜","馆":"館","条":"條","术":"術",
    "汉":"漢","汤":"湯","面":"麵","酱":"醬","团":"團","凤":"鳳","点":"點","里":"裡",
})
def to_tw_if_needed(text_cn: str, zh_variant: str) -> str:
    return text_cn.translate(CN_TO_TW_TABLE) if zh_variant == "tw" else text_cn

def titlecase_en(s: str) -> str:
    if not s: return s
    parts = re.split(r"([/\-\s])", s)
    def cap(tok: str) -> str:
        if not tok or tok in "/- ":
            return tok
        if tok.isupper() and len(tok) <= 4:
            return tok
        return tok[:1].upper() + tok[1:]
    return "".join(cap(p) for p in parts)

# -------------------- 용어 사전 --------------------
TERM_DICT: Dict[str, Tuple[str, str]] = {
    "카페": ("Cafe", "咖啡店"),
    "아이스크림": ("Ice Cream", "冰淇淋"),
    "브런치": ("Brunch", "早午餐"),
    "흑돼지": ("Jeju Black Pork", "济州黑猪"),
    "물회": ("Mulhoe (Cold Raw Fish Soup)", "凉拌生鱼汤"),
    "고기국수": ("Pork Noodle Soup", "猪肉面"),
    "회": ("Sashimi", "生鱼片"),
    "갈치구이": ("Grilled Hairtail", "烤带鱼"),
    "갈치조림": ("Braised Hairtail", "炖带鱼"),
    "한정식": ("Korean Course Meal", "韩式定食"),
    "비빔밥": ("Bibimbap", "拌饭"),
    "파스타": ("Pasta", "意面"),
    "피자": ("Pizza", "披萨"),
    "버거": ("Burger", "汉堡"),
    "스시": ("Sushi", "寿司"),
    "오마카세": ("Omakase", "主厨精选"),
    "데이트": ("Date Spot", "约会圣地"),
    "가성비": ("Value for Money", "性价比高"),
    "주차": ("Parking", "停车"),
}

def dict_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    pair = TERM_DICT.get(ko)
    if not pair: return (None, None)
    en, cn = pair
    return (titlecase_en(en), to_tw_if_needed(cn, zh_variant))

def _split_tokens_for_dict(s: str) -> List[str]:
    parts = re.split(r"[,\|/·•⋯\s]+", s or "")
    return [p for p in (t.strip() for t in parts) if p]

def ml_with_dict(ko_text: str,
                 translate_mode: str,
                 transctl: Optional[TransCtl],
                 zh_variant_cli: str,
                 dict_mode: str = "token") -> Dict[str, str]:
    ko = ko_text or ""
    zh_var = "tw" if (transctl and transctl.zh_variant == "zh-TW") else "cn"

    # 외부 번역 미사용 모드
    if translate_mode in ("off", "romanize") or not transctl or transctl.provider == "none":
        en = romanize_korean(ko) if translate_mode == "romanize" else ko
        cn = ko
        return {"ko": ko, "en": en, "cn": cn}

    if dict_mode == "exact":
        en_d, cn_d = dict_lookup(ko, zh_var)
        if en_d or cn_d:
            en = en_d or (transctl.translate(ko, "en") or ko)
            cn_raw = cn_d or (transctl.translate(ko, transctl.zh_variant) or ko)
            cn = to_tw_if_needed(cn_raw, zh_var)
            return {"ko": ko, "en": en or ko, "cn": cn or ko}

    en_tok: List[str] = []
    cn_tok: List[str] = []
    used = False
    if dict_mode == "token":
        for tk in _split_tokens_for_dict(ko):
            en_d, cn_d = dict_lookup(tk, zh_var)
            if en_d or cn_d:
                used = True
            en_tok.append(en_d or tk)
            cn_tok.append(cn_d or tk)

    if dict_mode == "token" and used:
        en = " ".join(en_tok)
        cn_raw = " ".join(cn_tok)
        cn = to_tw_if_needed(cn_raw, zh_var)
        return {"ko": ko, "en": en or ko, "cn": cn or ko}

    # 사전 미적용 → 기계번역 (캐시 활용은 TransCtl에서 자동 처리)
    en = transctl.translate(ko, "en") or ko
    cn_raw = transctl.translate(ko, transctl.zh_variant) or ko
    cn = to_tw_if_needed(cn_raw, zh_var)
    return {"ko": ko, "en": en or ko, "cn": cn or ko}

def build_ml_json(ko_text: str, translate_mode: str, transctl: Optional[TransCtl]) -> Dict[str, str]:
    if translate_mode in ("off", "romanize") or not transctl or transctl.provider == "none":
        en = romanize_korean(ko_text) if translate_mode == "romanize" else ko_text
        cn = ko_text
        return {"ko": ko_text, "en": en, "cn": cn}
    en = transctl.translate(ko_text, "en") or ko_text
    cn = transctl.translate(ko_text, transctl.zh_variant) or ko_text
    return {"ko": ko_text, "en": en, "cn": cn}

def build_name_ml_json(name: str, translate_mode: str, transctl: Optional[TransCtl], zh_variant_cli: str) -> Dict[str, str]:
    return ml_with_dict(name, translate_mode, transctl, zh_variant_cli, dict_mode="token")

# -------------------- 파일 저장 --------------------
def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)

def save_json(path: str, data: Dict[str, Any]):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------------------- 업그레이드(기존 메뉴 파일에 name_json 채움) --------------------
def upgrade_existing_menu_file(menu_path: str,
                               translate_mode: str,
                               transctl: Optional[TransCtl],
                               zh_variant: str,
                               dict_mode: str = "token") -> bool:
    try:
        with open(menu_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return False
    changed = False
    for r in rows if isinstance(rows, list) else []:
        if "name" in r and not r.get("name_json"):
            r["name_json"] = ml_with_dict(r.get("name",""), translate_mode, transctl, zh_variant, dict_mode=dict_mode)
            changed = True
    if changed:
        save_json(menu_path, rows)
    return changed

# -------------------- CLI/Args --------------------
@dataclass
class Args:
    files: List[str]
    timeout: float
    sleep: float
    force: bool
    upgrade: bool
    show: bool
    translate: str
    translate_timeout: float
    translate_max: Optional[int]  # None이면 무제한
    translate_provider: str
    zh_variant: str
    trace_translate: bool
    config: str
    cache_file: Optional[str]
    log_every: int
    papago_honorific: bool
    tx_error_log: Optional[str]
    dict_mode: str

def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="DiningCode 메뉴/상세 수집기 (Papago 번역/캐시/에러가시화)")
    ap.add_argument("--files", nargs="+", required=True, help="JSON 파일/패턴 (공백 구분, 글롭 지원)")
    ap.add_argument("--timeout", type=float, default=8.0, help="HTTP 타임아웃(초)")
    ap.add_argument("--sleep", type=float, default=0.3, help="요청 간 대기(초)")
    ap.add_argument("--force", action="store_true", help="이미 존재해도 재수집/덮어쓰기 (메뉴/텍스트 모두)")
    ap.add_argument("--upgrade", type=str2bool, default=False, nargs="?", const=True,
                    help="(메뉴 파일만) name_json 없으면 채워 저장")
    ap.add_argument("--show", type=str2bool, default=False, nargs="?", const=True,
                    help="수집/기존 '메뉴' 표 + '텍스트 요약'을 화면에 출력")

    # 번역 옵션
    ap.add_argument("--translate", choices=["auto","romanize","off"], default="romanize",
                    help="다국어 생성 모드 (auto=Papago, romanize=외부호출 없음)")
    ap.add_argument("--translate-timeout", type=float, default=8.0, help="번역 호출 타임아웃(초)")
    ap.add_argument("--translate-max", type=int, default=0,  # ≤0 → 무제한
                    help="외부 번역 총 호출 상한(언어 합산). 0 이하면 무제한.")
    ap.add_argument("--translate-provider",
                    choices=["papago","auto_chain","googletrans","deep"],
                    default="papago", help="번역 제공자 (기본 papago)")

    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn", help="중국어 변형: cn=간체, tw=번체")
    ap.add_argument("--trace-translate", type=str2bool, default=False, nargs="?", const=True,
                    help="번역 호출/캐시 상세 로그 출력")

    # 추가 옵션
    ap.add_argument("--cache-file", default=".cache/restaurant-trans.json", help="번역 캐시 JSON 경로")
    ap.add_argument("--dict-mode", choices=["off","exact","token"], default="token", help="사전 적용 범위")
    ap.add_argument("--log-every", type=int, default=100, help="N개 처리마다 진행 로그(보조 로그)")

    # Papago 설정
    ap.add_argument("--config", default="config.json", help="Papago 키를 포함한 설정 파일 경로")
    ap.add_argument("--papago-honorific", type=str2bool, default=True, nargs="?",
                    const=True, help="Papago 한국어 경어 옵션(true/false)")

    # 에러 로그 파일
    ap.add_argument("--tx-error-log", default=".logs/papago_errors.json",
                    help="Papago 실패 이벤트 JSON 로그 경로")

    ns = ap.parse_args()
    # translate_max: ≤0 → None(무제한)
    tmax: Optional[int] = None if ns.translate_max is None or ns.translate_max <= 0 else int(ns.translate_max)
    return Args(
        files=ns.files,
        timeout=ns.timeout,
        sleep=ns.sleep,
        force=ns.force,
        upgrade=ns.upgrade,
        show=ns.show,
        translate=ns.translate,
        translate_timeout=ns.translate_timeout,
        translate_max=tmax,
        translate_provider=ns.translate_provider,
        zh_variant=ns.zh_variant,
        trace_translate=ns.trace_translate,
        config=ns.config,
        cache_file=ns.cache_file,
        log_every=ns.log_every,
        papago_honorific=ns.papago_honorific,
        tx_error_log=ns.tx_error_log,
        dict_mode=ns.dict_mode,
    )

# -------------------- 캐시 I/O --------------------
def load_trans_cache(path: Optional[str]):
    if not path: return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cnt = 0
        if isinstance(data, dict):
            for dest, m in data.items():
                if not isinstance(m, dict): continue
                # 과거 파일에 'cn'/'zh' 등 비표준 키가 들어있어도 읽을 때 정규화
                dest_norm = norm_dest_lang(dest)
                for norm_text, val in m.items():
                    _trans_cache[(dest_norm, norm_text)] = val
                    cnt += 1
        print(f"[CACHE] 로드: {cnt}건 from {path}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[CACHE] 로드 실패({e}) — 무시")

def save_trans_cache(path: Optional[str]):
    if not path: return
    out: Dict[str, Dict[str, str]] = {}
    for (dest_norm, norm_text), val in _trans_cache.items():
        out.setdefault(dest_norm, {})[norm_text] = val
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[CACHE] 저장: {len(_trans_cache)}건 -> {path}")
    except Exception as e:
        print(f"[CACHE] 저장 실패({e}) — 무시")

# -------------------- 처리 --------------------
def process_record(session: requests.Session, trans: Optional[TransCtl], rid: str, args: Args) -> Dict[str, Any]:
    url = _mk_url(rid)
    html = fetch_html(url, timeout=args.timeout)

    # 이름
    name = parse_store_name(html)
    name_json = build_name_ml_json(name, args.translate, trans, args.zh_variant)

    # 메뉴
    rows_menu = parse_menuinfo_items(html)
    for r in rows_menu:
        r["name_json"] = ml_with_dict(
            r.get("name",""),
            args.translate,
            trans,
            args.zh_variant,
            dict_mode=args.dict_mode
        )

    # 텍스트
    textinfo = parse_textinfo(html)
    btxt = textinfo.get("btxt", [])
    tags = textinfo.get("tags", [])
    chars = textinfo.get("chars", [])

    description = compose_description(btxt, tags, chars)
    description_json = build_ml_json(description, args.translate, trans)

    short_description = compose_short_description(btxt, tags)
    short_description_json = build_ml_json(short_description, args.translate, trans)

    return {
        "rid": rid,
        "url": url,
        "name": name,
        "name_json": name_json,  # {"ko","en","cn"}
        "description": description,
        "description_json": description_json,  # {"ko","en","cn"}
        "short_description": short_description,
        "short_description_json": short_description_json,  # {"ko","en","cn"}
        "menu_rows": rows_menu,
        "_meta": {
            "collected_at": now_local_str(),
            "translator": (trans.provider if trans else "none"),
            "zh_variant": args.zh_variant,
        }
    }

# -------------------- 메인 --------------------
def main():
    start_perf = time.perf_counter()
    print(f"[TIME] 시작: {now_local_str()}")

    args = parse_args()

    # 번역 컨트롤러 구성
    transctl: Optional[TransCtl] = None
    if args.translate == "auto":
        provider = args.translate_provider or "papago"
        transctl = TransCtl(
            provider=provider,
            timeout=args.translate_timeout,
            max_calls=args.translate_max,  # None이면 무제한
            zh_variant=args.zh_variant,
            trace=args.trace_translate,
            config_path=args.config,
            papago_honorific=args.papago_honorific,
        )
    else:
        transctl = TransCtl(
            provider="none",
            timeout=args.translate_timeout,
            max_calls=None,
            zh_variant=args.zh_variant,
            trace=False,
        )

    # 캐시 로드
    load_trans_cache(args.cache_file)

    files = gather_files(args.files)
    if not files:
        print("[ERR] --files에 해당하는 입력이 없습니다.", file=sys.stderr)
        _save_papago_errors(args.tx_error_log)
        _print_papago_summary()
        print(f"[TIME] 종료: {now_local_str()}")
        sys.exit(2)

    # 파일별 RID 수 & 전체 수집
    vrids: List[str] = []
    seen: Set[str] = set()
    cumulative = 0
    print(f"[INFO] 입력 파일 {len(files)}개")
    for path in files:
        ids = collect_vrids_from_file(path)
        unique_ids = []
        for rid in ids:
            if rid not in seen:
                seen.add(rid)
                unique_ids.append(rid)
        cumulative += len(unique_ids)
        vrids.extend(unique_ids)
        print(f"[FILE] {path} → rid:{len(unique_ids)}건 (누적:{cumulative})")
    print(f"[INFO] 총 대상 RID {len(vrids)}건")

    session = _requests_session(timeout=args.timeout)

    # 루프
    ok = skip = err = 0
    t0 = time.perf_counter()
    total = len(vrids)
    for i, rid in enumerate(vrids, 1):
        # 매 항목 진행상황 출력
        print(f"[STEP] {i}/{total} rid={rid}")

        # 보조 진행 로그(주기적)
        if args.log_every and (i == 1 or i % args.log_every == 0):
            calls = getattr(transctl, "calls_used", None) if transctl else None
            elapsed = time.perf_counter() - t0
            print(f"[PROG] {i}/{total} elapsed={elapsed:.1f}s tx_calls={calls}")

        menu_out = os.path.join("data", "menu", f"{rid}.json")
        text_out = os.path.join("data", "detail", f"{rid}.json")

        need_menu = args.force or (not os.path.exists(menu_out))
        need_text = args.force or (not os.path.exists(text_out))
        if not need_menu and not need_text and not args.upgrade:
            skip += 1
            continue

        try:
            data = process_record(session, transctl, rid, args)
            if need_menu and data.get("menu_rows"):
                save_json(menu_out, data["menu_rows"])
            elif args.upgrade and os.path.exists(menu_out):
                if upgrade_existing_menu_file(menu_out, args.translate, transctl, args.zh_variant, dict_mode=args.dict_mode):
                    print(f"[UPG] menu name_json 채움: {menu_out}")

            if need_text:
                save_json(text_out, {
                    "rid": rid,
                    "url": data["url"],
                    "name": data["name"],
                    "name_json": data["name_json"],
                    "description": data["description"],
                    "description_json": data["description_json"],
                    "short_description": data["short_description"],
                    "short_description_json": data["short_description_json"],
                    "_meta": data["_meta"],
                })
            ok += 1

            if args.show:
                print(f"[OK] RID={rid} | name={data['name_json']}")
                if data.get("menu_rows"):
                    print(f"  MENU x{len(data['menu_rows'])}: " + ", ".join(r['name'] for r in data["menu_rows"][:6]))

        except Exception as e:
            err += 1
            print(f"[ERR] RID={rid} 처리 실패: {e}", file=sys.stderr)

        if args.sleep and i < total:
            time.sleep(args.sleep)

    # 캐시 저장
    save_trans_cache(args.cache_file)

    # Papago 요약/로그 저장
    _save_papago_errors(args.tx_error_log)
    _print_papago_summary()

    print(f"[DONE] ok:{ok} skip:{skip} err:{err}")
    elapsed = time.perf_counter() - start_perf
    print(f"[TIME] 종료: {now_local_str()} (elapsed={elapsed:.1f}s)")

if __name__ == "__main__":
    main()
