#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-상세수집.py — DiningCode 상세 수집기 (cache/dict/progress 옵션 포함)

기능
1) 메뉴 수집 → data/menu/[rid].json
   - 항목: name | price_text | price_krw | recommended | name_json({"ko","en","cn"})
   - name_json은 사전/토큰 매핑 → 자동번역(googletrans→deep_translator) → 로마자 폴백
2) 상세 텍스트 수집 → data/detail/[rid].json
   - name / name_json({"ko","en","cn"})              ✅
   - description / description_json({"ko","en","cn"})
   - short_description / short_description_json({"ko","en","cn"})

CLI 옵션
  --cache-file PATH        : 번역 캐시 파일 경로 (없어도 자동 생성)
  --dict-mode off|exact|token  : 사전 적용 범위(기본 token)
  --log-every N            : 진행 로그 주기
  --translate auto|romanize|off, --zh-variant cn|tw, --trace-translate
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import requests
from bs4 import BeautifulSoup

# -------------------- 정규식/유틸 --------------------
PRICE_PAT = re.compile(r"([0-9][0-9,\.]*)\s*원")
CLEAN_WS = re.compile(r"\s+")
RE_HANGUL = re.compile(r"[가-힣]")

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
            seen.add(rid)
            vrids.append(rid)
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

# -------------------- 웹 요청 / 파서 --------------------
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

# -------------------- 번역/캐시/로마자 --------------------
def _norm_key(s: str) -> str:
    return CLEAN_WS.sub(" ", (s or "").strip()).lower()

_L = ["g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"]
_V = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo","u","wo","we","wi","yu","eu","ui","i"]
_T = ["","k","k","ks","n","nj","nh","t","l","lk","lm","lb","ls","lt","lp","lh","m","p","ps","t","t","ng","t","t","k","t","p","t"]

_roman_cache: Dict[str, str] = {}
_trans_cache: Dict[Tuple[str, str], str] = {}
_mljson_cache: Dict[Tuple[str, str, str], Dict[str, str]] = {}

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

def _try_googletrans(text: str, dest: str, src: str = "ko") -> Optional[str]:
    try:
        from googletrans import Translator  # type: ignore
        tr = Translator()
        res = tr.translate(text, src=src, dest=dest)
        return res.text if getattr(res, "text", None) else None
    except Exception:
        return None

def _try_deeptranslator(text: str, dest: str, src: str = "ko") -> Optional[str]:
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        tr = GoogleTranslator(source=src, target=dest)
        return tr.translate(text)
    except Exception:
        return None

def _run_with_timeout(fn, timeout_sec: float, *args, **kwargs) -> Optional[str]:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeout:
            return None
        except Exception:
            return None

class TransCtl:
    def __init__(self, provider: str, timeout: float, max_calls: int, zh_variant: str, trace: bool = False):
        self.provider = provider
        self.timeout = timeout
        self.max_calls = max_calls
        self.calls_used = 0
        self.zh_variant = "zh-TW" if zh_variant == "tw" else "zh-CN"
        self.trace = trace

    def _can_call(self) -> bool:
        return self.max_calls is None or self.calls_used < self.max_calls

    def translate(self, text: str, dest: str) -> Optional[str]:
        key = (dest, _norm_key(text))
        if key in _trans_cache:
            return _trans_cache[key]
        if not self._can_call():
            return None

        out: Optional[str] = None
        if self.provider in ("googletrans", "auto_chain"):
            self.calls_used += 1
            if self.trace: print(f"[TX] googletrans → {dest}: {text!r}")
            out = _run_with_timeout(_try_googletrans, self.timeout, text, dest, "ko")
        if not out and self.provider in ("deep", "auto_chain"):
            if not self._can_call():
                return None
            self.calls_used += 1
            if self.trace: print(f"[TX] deep_translator → {dest}: {text!r}")
            out = _run_with_timeout(_try_deeptranslator, self.timeout, text, dest, "ko")

        if out:
            _trans_cache[key] = out
        return out

# 캐시 파일 입출력
def load_trans_cache(path: Optional[str]):
    if not path: return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cnt = 0
        if isinstance(data, dict):
            for dest, m in data.items():
                if not isinstance(m, dict): continue
                for norm_text, val in m.items():
                    _trans_cache[(dest, norm_text)] = val
                    cnt += 1
        print(f"[CACHE] 로드: {cnt}건 from {path}")
    except FileNotFoundError:
        print(f"[CACHE] 파일 없음(신규 생성 예정): {path}")
    except Exception as e:
        print(f"[CACHE] 로드 실패({e}) — 무시")

def save_trans_cache(path: Optional[str]):
    if not path: return
    out: Dict[str, Dict[str, str]] = {}
    for (dest, norm_text), val in _trans_cache.items():
        out.setdefault(dest, {})[norm_text] = val
    try:
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[CACHE] 저장: {len(_trans_cache)}건 -> {path}")
    except Exception as e:
        print(f"[CACHE] 저장 실패({e}) — 무시")

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

# -------------------- 용어 사전 (요약) --------------------
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

def slash_lookup(ko: str, zh_variant: str) -> Tuple[Optional[str], Optional[str]]:
    if "/" not in ko: return (None, None)
    parts = [p.strip() for p in ko.split("/")]
    en_parts, cn_parts = [], []
    for p in parts:
        en_p, cn_p = dict_lookup(p, zh_variant)
        en_parts.append(en_p or p)
        cn_parts.append(cn_p or p)
    en = " / ".join(titlecase_en(x) for x in en_parts)
    cn = " / ".join(cn_parts)
    return (en, to_tw_if_needed(cn, zh_variant))

def token_map(ko: str, zh_variant: str) -> Tuple[str, str, bool]:
    if not ko: return ("", "", False)
    toks = re.split(r"([ +\-])", ko)
    en_out, cn_out, replaced = [], [], False
    for t in toks:
        if t in (" ", "+", "-"):
            en_out.append(t); cn_out.append(t); continue
        en_t, cn_t = dict_lookup(t, zh_variant)
        if en_t or cn_t: replaced = True
        en_out.append(en_t or t)
        cn_out.append(cn_t or t)
    en_res = titlecase_en("".join(en_out))
    cn_res = to_tw_if_needed("".join(cn_out), zh_variant)
    return (en_res, cn_res, replaced)

def ml_with_dict(ko: str, translate_mode: str, transctl: Optional[TransCtl],
                 zh_variant: str, dict_mode: str = "token") -> Dict[str, str]:
    ko = (ko or "").strip()
    if not ko:
        return {"ko":"", "en":"", "cn":""}

    # exact
    en_d, cn_d = dict_lookup(ko, zh_variant)
    if en_d or cn_d:
        return {"ko": ko, "en": en_d or "", "cn": cn_d or ""}

    # slash
    en_s, cn_s = slash_lookup(ko, zh_variant)
    if en_s or cn_s:
        return {"ko": ko, "en": en_s or "", "cn": cn_s or ""}

    # token
    if dict_mode == "token":
        en_t, cn_t, replaced = token_map(ko, zh_variant)
        if replaced:
            return {"ko": ko, "en": en_t, "cn": cn_t}

    # fallback — 일반 번역
    base = build_ml_json(ko, translate_mode, transctl)
    return {"ko": ko, "en": base.get("en",""), "cn": base.get("cn","")}

# 일반 텍스트 → (ko,en,cn)
def build_ml_json(ko: str, translate_mode: str, transctl: Optional[TransCtl]) -> Dict[str, str]:
    ko = (ko or "").strip()
    if not ko:
        return {"ko": "", "en": "", "cn": ""}

    key = (_norm_key(ko), translate_mode, getattr(transctl, "zh_variant", "zh-CN"))
    if key in _mljson_cache:
        c = _mljson_cache[key]
        return {"ko": ko, "en": c.get("en",""), "cn": c.get("cn","")}

    if translate_mode == "off":
        res = {"ko": ko, "en": "", "cn": ""}
        _mljson_cache[key] = res
        return res

    if translate_mode == "romanize" or transctl is None:
        en = romanize_korean(ko) if has_hangul(ko) else ko
        cn = en
        res = {"ko": ko, "en": en, "cn": cn}
        _mljson_cache[key] = res
        return res

    en = ""
    cn = ""
    if has_hangul(ko):
        t_en = transctl.translate(ko, "en")
        en = (t_en or "").strip()
    else:
        en = ko
    if not en or has_hangul(en):
        en = romanize_korean(ko)
    t_cn = transctl.translate(ko, transctl.zh_variant) if has_hangul(ko) else None
    cn = (t_cn or "").strip() if t_cn else ""
    if not cn or has_hangul(cn):
        cn = en
    res = {"ko": ko, "en": titlecase_en(en), "cn": to_tw_if_needed(cn, "tw" if getattr(transctl, "zh_variant", "zh-CN")=="zh-TW" else "cn")}
    _mljson_cache[key] = res
    return res

# 상호명 전용
def build_name_ml_json(ko_name: str, translate_mode: str, transctl: Optional[TransCtl], zh_variant: str) -> Dict[str, str]:
    ko_name = (ko_name or "").strip()
    if not ko_name:
        return {"ko":"", "en":"", "cn":""}

    if translate_mode == "off":
        return {"ko": ko_name, "en": "", "cn": ""}

    if translate_mode == "romanize" or transctl is None:
        en = titlecase_en(romanize_korean(ko_name) if has_hangul(ko_name) else ko_name)
        return {"ko": ko_name, "en": en, "cn": en}

    t_en = transctl.translate(ko_name, "en") or ""
    t_cn = transctl.translate(ko_name, transctl.zh_variant) or ""
    if not t_en or has_hangul(t_en):
        t_en = romanize_korean(ko_name)
    if not t_cn or has_hangul(t_cn):
        t_cn = t_en
    t_en = titlecase_en(t_en.strip())
    t_cn = to_tw_if_needed(t_cn.strip(), zh_variant)
    return {"ko": ko_name, "en": t_en, "cn": t_cn}

# -------------------- 표시/저장 --------------------
def _namejson_str(v: Any) -> str:
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, separators=(",",":"))
    if isinstance(v, str):
        return v
    return ""

def print_menu_table(rows: List[Dict[str, Any]]):
    if not rows:
        print("(수집된 메뉴 없음)")
        return
    cols = ["name", "price_text", "price_krw", "recommended", "name_json"]
    def w(col, cap):
        vals = []
        for r in rows:
            x = r.get(col, "")
            if col == "name_json": x = _namejson_str(x)
            vals.append(len(str(x)))
        mx = max([len(col)] + vals)
        return min(max(6, mx), cap)
    widths = {
        "name": w("name", 40),
        "price_text": w("price_text", 10),
        "price_krw": w("price_krw", 10),
        "recommended": w("recommended", 11),
        "name_json": w("name_json", 70),
    }
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c == "name_json": v = _namejson_str(v)
            cells.append(str(v).ljust(widths[c]))
        print(" | ".join(cells))

def save_json(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[OK] 저장: {path} ({len(rows)}건)")

def save_text_json(
    rid: str,
    url: str,
    name: str,
    name_json: Dict[str, str],
    textinfo: Dict[str, List[str]],
    description: str,
    description_json: Dict[str, str],
    short_description: str,
    short_description_json: Dict[str, str],
    path: str
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "v_rid": rid,
        "url": url,
        "name": name,
        "name_json": name_json,  # {"ko","en","cn"}
        "btxt": textinfo.get("btxt", []),
        "tags": textinfo.get("tags", []),
        "chars": textinfo.get("chars", []),
        "description": description,
        "description_json": description_json,
        "short_description": short_description,
        "short_description_json": short_description_json,
        "fetched_at": now_local_str(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] 텍스트 저장: {path} (name:'{name}', btxt:{len(payload['btxt'])}, tags:{len(payload['tags'])}, chars:{len(payload['chars'])})")

def print_textinfo(payload: Dict[str, Any]):
    print("[TEXT]")
    print(f"  url        : {payload.get('url','')}")
    print(f"  name       : {payload.get('name','')}")
    try:
        nj = json.dumps(payload.get('name_json', {}), ensure_ascii=False, separators=(",",":"))
    except Exception:
        nj = str(payload.get('name_json', {}))
    print(f"  name_json  : {nj}")
    print(f"  btxt({len(payload.get('btxt',[]))}): { ' | '.join(payload.get('btxt',[])) }")
    print(f"  tags({len(payload.get('tags',[]))}): { ', '.join(payload.get('tags',[])) }")
    print(f"  chars({len(payload.get('chars',[]))}): { ', '.join(payload.get('chars',[])) }")
    print(f"  description: {payload.get('description','')}")
    try:
        dj = json.dumps(payload.get('description_json', {}), ensure_ascii=False, separators=(",",":"))
    except Exception:
        dj = str(payload.get('description_json', {}))
    print(f"  description_json     : {dj}")
    print(f"  short_description    : {payload.get('short_description','')}")
    try:
        sdj = json.dumps(payload.get('short_description_json', {}), ensure_ascii=False, separators=(",",":"))
    except Exception:
        sdj = str(payload.get('short_description_json', {}))
    print(f"  short_description_json: {sdj}")

# -------------------- 업그레이드(메뉴 파일 보강) --------------------
def upgrade_existing_menu_file(path: str, translate_mode: str, transctl: Optional[TransCtl],
                               zh_variant: str, dict_mode: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as e:
        print(f"[WARN] 업그레이드용 기존 메뉴 파일 읽기 실패: {e}", file=sys.stderr)
        return False

    changed = False
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict): continue
            nj = r.get("name_json")
            if not (isinstance(nj, dict) and nj.get("ko") and "cn" in nj):
                r["name_json"] = ml_with_dict(r.get("name",""), translate_mode, transctl, zh_variant, dict_mode=dict_mode)
                changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[OK] 업그레이드 저장(메뉴 보강): {path} ({len(rows)}건)")
    else:
        print(f"[OK] 업그레이드 필요 없음(메뉴): {path}")
    return changed

# -------------------- 메인 --------------------
def main():
    start_perf = time.perf_counter()
    print(f"[TIME] 시작: {now_local_str()}")

    ap = argparse.ArgumentParser(description="DiningCode 메뉴/상세 수집기 (이름 번역/캐시/사전 지원)")
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
                    help="다국어 생성 모드 (기본 romanize: 외부 호출 없음)")
    ap.add_argument("--translate-timeout", type=float, default=2.0, help="번역 호출 타임아웃(초)")
    ap.add_argument("--translate-max", type=int, default=30, help="외부 번역 총 호출 상한(언어 합산)")
    ap.add_argument("--translate-provider", choices=["auto_chain","googletrans","deep"], default="auto_chain",
                    help="auto_chain: googletrans→deep_translator 폴백")
    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn", help="중국어 변형: cn=간체, tw=번체")
    ap.add_argument("--trace-translate", action="store_true", help="번역 호출 로그 출력")

    # ✅ 추가 옵션
    ap.add_argument("--cache-file", default=None, help="번역 캐시 JSON 경로")
    ap.add_argument("--dict-mode", choices=["off","exact","token"], default="token", help="사전 적용 범위")
    ap.add_argument("--log-every", type=int, default=100, help="N개 처리마다 진행 로그")

    args = ap.parse_args()

    transctl: Optional[TransCtl] = None
    if args.translate == "auto":
        transctl = TransCtl(
            provider=args.translate_provider,
            timeout=args.translate_timeout,
            max_calls=args.translate_max,
            zh_variant=args.zh_variant,
            trace=args.trace_translate,
        )

    # 캐시 로드
    load_trans_cache(args.cache_file)

    file_list = gather_files(args.files)
    if not file_list:
        print("[ERR] --files 패턴에 해당하는 파일이 없습니다.", file=sys.stderr)
        _end(start_perf); save_trans_cache(args.cache_file); sys.exit(2)

    vrids: List[str] = []
    seen: Set[str] = set()
    for path in file_list:
        ids = collect_vrids_from_file(path)
        for rid in ids:
            if rid not in seen:
                seen.add(rid); vrids.append(rid)

    if not vrids:
        print("[ERR] 입력 파일들에서 v_rid를 찾지 못했습니다.", file=sys.stderr)
        _end(start_perf); save_trans_cache(args.cache_file); sys.exit(2)

    print(f"[INFO] 대상 RID {len(vrids)}건")
    m_ok = m_skip = m_upg = 0
    t_ok = t_skip = 0
    total_err = 0
    t0 = time.perf_counter()

    for idx, rid in enumerate(vrids, 1):
        url = _mk_url(rid)
        menu_out = os.path.join("data", "menu", f"{rid}.json")
        text_out = os.path.join("data", "detail", f"{rid}.json")

        if (idx == 1) or (idx % max(1, args.log_every) == 0):
            elapsed = time.perf_counter() - t0
            calls = getattr(transctl, "calls_used", None) if transctl else None
            extra = f", tx_calls={calls}" if calls is not None else ""
            print(f"[PROG] {idx}/{len(vrids)} elapsed={elapsed:.1f}s rid={rid}{extra}")

        print(f"\n[{idx}/{len(vrids)}] RID={rid}")

        # 업그레이드(메뉴 파일만)
        if os.path.exists(menu_out) and args.upgrade and not args.force:
            if upgrade_existing_menu_file(menu_out, args.translate, transctl, args.zh_variant, args.dict_mode):
                m_upg += 1
            else:
                m_skip += 1

        need_menu = args.force or (not os.path.exists(menu_out))
        need_text = args.force or (not os.path.exists(text_out))
        need_fetch = need_menu or need_text

        rows_menu: List[Dict[str, Any]] = []
        textinfo: Dict[str, List[str]] = {}
        name = ""
        name_json = {"ko":"", "en":"", "cn":""}
        description = ""
        description_json: Dict[str, str] = {"ko":"", "en":"", "cn":""}
        short_description = ""
        short_description_json: Dict[str, str] = {"ko":"", "en":"", "cn":""}

        if need_fetch:
            print(f"[INFO] 요청: {url}")
            try:
                html = fetch_html(url, timeout=args.timeout)
            except Exception as e:
                print(f"[ERR] 페이지 요청 실패(RID={rid}): {e}", file=sys.stderr)
                total_err += 1
                if args.sleep and idx < len(vrids): time.sleep(args.sleep)
                continue

            # 이름 & 번역
            name = parse_store_name(html)
            name_json = build_name_ml_json(name, args.translate, transctl, args.zh_variant)

            # 메뉴
            if need_menu:
                rows_menu = parse_menuinfo_items(html)
                for r in rows_menu:
                    r["name_json"] = ml_with_dict(r.get("name",""), args.translate, transctl, args.zh_variant, dict_mode=args.dict_mode)

            # 상세 텍스트
            if need_text:
                textinfo = parse_textinfo(html)
                btxt = textinfo.get("btxt", [])
                tags = textinfo.get("tags", [])
                chars = textinfo.get("chars", [])

                description = compose_description(btxt, tags, chars)
                description_json = build_ml_json(description, args.translate, transctl)

                short_description = compose_short_description(btxt, tags)
                short_description_json = build_ml_json(short_description, args.translate, transctl)

        # 저장/스킵 — 메뉴
        try:
            if need_menu:
                save_json(rows_menu, menu_out)
                m_ok += 1
            else:
                print(f"[SKIP] 메뉴 파일 존재: {menu_out} (force 미사용)")
                m_skip += 1
        except Exception as e:
            print(f"[ERR] 메뉴 저장 실패(RID={rid}): {e}", file=sys.stderr)
            total_err += 1

        # 저장/스킵 — 상세
        try:
            if need_text:
                save_text_json(
                    rid, url,
                    name, name_json,
                    textinfo,
                    description, description_json,
                    short_description, short_description_json,
                    text_out
                )
                t_ok += 1
            else:
                print(f"[SKIP] 텍스트 파일 존재: {text_out} (force 미사용)")
                t_skip += 1
        except Exception as e:
            print(f"[ERR] 텍스트 저장 실패(RID={rid}): {e}", file=sys.stderr)
            total_err += 1

        # 화면 표시
        if args.show:
            try:
                rows_for_show = rows_menu
                if not rows_for_show and os.path.exists(menu_out):
                    with open(menu_out, "r", encoding="utf-8") as f:
                        rows_for_show = json.load(f)
                print(f"[SHOW] 메뉴 {len(rows_for_show)}건")
                print_menu_table(rows_for_show)
            except Exception as e:
                print(f"[WARN] 메뉴 표시 실패: {e}", file=sys.stderr)

            try:
                text_payload = {
                    "url": url,
                    "name": name,
                    "name_json": name_json,
                    "btxt": textinfo.get("btxt", []),
                    "tags": textinfo.get("tags", []),
                    "chars": textinfo.get("chars", []),
                    "description": description,
                    "description_json": description_json,
                    "short_description": short_description,
                    "short_description_json": short_description_json,
                }
                print_textinfo(text_payload)
            except Exception as e:
                print(f"[WARN] 텍스트 표시 실패: {e}", file=sys.stderr)

        if args.sleep and idx < len(vrids):
            time.sleep(args.sleep)

    print("\n[SUMMARY]")
    print(f"  메뉴 새로 저장: {m_ok}건")
    print(f"  메뉴 업그레이드: {m_upg}건")
    print(f"  메뉴 스킵: {m_skip}건")
    print(f"  텍스트 새로 저장: {t_ok}건")
    print(f"  텍스트 스킵: {t_skip}건")
    print(f"  실패: {total_err}건")
    _end(start_perf)
    save_trans_cache(args.cache_file)

def _end(start_perf: float):
    print(f"[TIME] 종료: {now_local_str()}")
    dt = time.perf_counter() - start_perf
    m = int(dt // 60); s = dt - m*60
    print(f"[TIME] 총 소요: {dt:.3f}s ({m}m {s:.3f}s)")

if __name__ == "__main__":
    main()
