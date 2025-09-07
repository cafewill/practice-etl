#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc-보여주기.py — DiningCode JSON 요약·필터·정렬·CSV 저장 + 진행 로그
(+주소 일관화, 다국어 JSON 필드: nm/addr/road_addr/category/keyword → *_json(ko/en/zh),
 번역 호출 상한/타임아웃, 메모리 캐시 3종, 실행 시간 출력)

캐시 종류(필드 공용)
- _trans_cache[(dest, text_norm)] : 외부 번역 결과 캐시
- _roman_cache[text_norm]         : 로마자 표기 캐시
- _mljson_cache[(text_norm, translate_mode, zh_variant)] : en/zh 세트 캐시

실행 시간 출력
- 시작/종료 시각: 로컬 타임존(now().astimezone()) 기준, 예: 2025-08-23 10:11:12 KST+0900
- 총 소요시간: 초 단위(소수점 3자리), m:s 표기도 함께 출력

진행 로그
- --log-every N : 아이템 N건마다 [PROG] i/total elapsed rate 출력 (예: 10 또는 100)
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# ---------- 유틸 ----------

def ellipsis(s: Optional[str], maxlen: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"

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
    for path in ("result_data.poi_section.list",):
        v = get_by_path(doc, path)
        if _looks_like_item_list(v):
            return v  # type: ignore
    for path in ("list", "result.list", "data.list"):
        v = get_by_path(doc, path)
        if _looks_like_item_list(v):
            return v  # type: ignore
    found: List[Dict[str, Any]] = []
    def dfs(x: Any):
        nonlocal found
        if found:
            return
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

def _join_terms(kw: Any) -> str:
    terms: List[str] = []
    if isinstance(kw, list):
        for k in kw:
            if isinstance(k, dict) and "term" in k:
                terms.append(str(k["term"]))
            elif isinstance(k, str):
                terms.append(k)
    elif isinstance(kw, dict) and "term" in kw:
        terms.append(str(kw["term"]))
    elif isinstance(kw, str):
        terms.append(kw)
    return ", ".join(terms)

def _join_area(area: Any) -> str:
    if isinstance(area, list):
        return ", ".join(str(x) for x in area)
    return "" if area is None else str(area)

def _safe_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# 주소 일관화: "제주특별자치도" → "제주도", 중복/공백 정리
def normalize_jeju(text: Optional[str]) -> str:
    if not text:
        return ""
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"제주특별자치도\s*", "제주도 ", s)
    s = re.sub(r"(제주도)(\s+\1)+", r"\1", s)
    s = re.sub(r"\s+([,)\]])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- 정규화/캐시 키 ----------

def _norm_key(s: str) -> str:
    """캐시 키용: 앞뒤 공백 제거, 다중 공백 1개로, 소문자."""
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

# ---------- 번역 도우미/캐시 ----------

_RE_HANGUL = re.compile(r"[가-힣]")

def has_hangul(s: str) -> bool:
    return bool(_RE_HANGUL.search(s))

# Revised Romanization (간단 버전)
_L = ["g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"]
_V = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo","u","wo","we","wi","yu","eu","ui","i"]
_T = ["","k","k","ks","n","nj","nh","t","l","lk","lm","lb","ls","lt","lp","lh","m","p","ps","t","t","ng","t","t","k","t","p","t"]

_roman_cache: Dict[str, str] = {}

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

def _try_external_translate_googletrans(text: str, dest: str, src: str = "ko") -> Optional[str]:
    try:
        from googletrans import Translator  # type: ignore
        tr = Translator()
        res = tr.translate(text, src=src, dest=dest)
        return res.text if getattr(res, "text", None) else None
    except Exception:
        return None

def _try_external_translate_deep(text: str, dest: str, src: str = "ko") -> Optional[str]:
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

# 번역 캐시: (dest, text_norm) → translated
_trans_cache: Dict[Tuple[str, str], str] = {}

class TransCtl:
    """번역 호출 상한/타임아웃/공급자 제어 + 캐시 사용"""
    def __init__(self, provider: str, timeout: float, max_calls: int, zh_variant: str):
        self.provider = provider  # auto_chain | googletrans | deep
        self.timeout = timeout
        self.max_calls = max_calls
        self.calls_used = 0
        self.zh_variant = "zh-TW" if zh_variant == "tw" else "zh-CN"

    def _can_call(self) -> bool:
        return self.max_calls is None or self.calls_used < self.max_calls

    def translate(self, text: str, dest: str) -> Optional[str]:
        key = (dest, _norm_key(text))
        if key in _trans_cache:
            return _trans_cache[key]
        if not self._can_call():
            return None

        t: Optional[str] = None
        if self.provider in ("googletrans", "auto_chain"):
            self.calls_used += 1
            t = _run_with_timeout(_try_external_translate_googletrans, self.timeout, text, dest, "ko")

        if not t and self.provider in ("deep", "auto_chain"):
            if not self._can_call():
                return None
            self.calls_used += 1
            t = _run_with_timeout(_try_external_translate_deep, self.timeout, text, dest, "ko")

        if t:
            _trans_cache[key] = t
        return t

# 공용 다국어 JSON 캐시: (text_norm, translate_mode, zh_variant) → {"en":..., "zh":...}
_mljson_cache: Dict[Tuple[str, str, str], Dict[str, str]] = {}

def build_multilang_json(text_ko: Optional[str],
                         translate_mode: str,
                         transctl: Optional[TransCtl]) -> Dict[str, str]:
    """
    입력: 한국어 문자열(정규화/가공된 최종 ko 텍스트)
    출력: {"ko": ko, "en": en, "zh": zh}
    정책:
      - translate_mode=='auto': 외부 번역(상한/타임아웃/캐시) 시도, 실패 시 폴백(en=로마자, zh=en/로마자)
      - translate_mode=='romanize': en=로마자, zh=en/로마자
      - translate_mode=='off': en="", zh=""
    """
    ko = (text_ko or "").strip()
    if ko == "":
        return {"ko": "", "en": "", "zh": ""}

    key = (_norm_key(ko), translate_mode, getattr(transctl, "zh_variant", "zh-CN"))
    if key in _mljson_cache:
        c = _mljson_cache[key]
        return {"ko": ko, "en": c.get("en",""), "zh": c.get("zh","")}

    en = ""
    zh = ""

    if translate_mode == "auto" and transctl is not None:
        if has_hangul(ko):
            t_en = transctl.translate(ko, "en")
            en = (t_en or "").strip()
        else:
            en = ko
        if not en:
            en = romanize_korean(ko)

        t_zh = None
        if has_hangul(ko):
            t_zh = transctl.translate(ko, transctl.zh_variant)
        if t_zh:
            zh = t_zh.strip()
        else:
            zh = en if en else romanize_korean(ko)

    elif translate_mode == "romanize":
        en = romanize_korean(ko)
        zh = en

    elif translate_mode == "off":
        en = ""
        zh = ""

    _mljson_cache[key] = {"en": en, "zh": zh}
    return {"ko": ko, "en": en, "zh": zh}

# ---------- 추출/로딩/출력 ----------

DEFAULT_COLS = [
    "nm","nm_json",
    "addr","addr_json",
    "road_addr","road_addr_json",
    "phone",
    "category","category_json",
    "keyword","keyword_json",
    "area","open_status","score","user_score","review_cnt"
]

SUPPORTED_COLS = set(DEFAULT_COLS) | {
    "branch","v_rid","lat","lng","image","image_list",
    "review_user","review_text","review_dt","dist_km"
}

def extract_row(item: Dict[str, Any],
                want_cols: List[str],
                near: Optional[Tuple[float,float]],
                translate_mode: str,
                transctl: Optional[TransCtl]) -> Dict[str, Any]:
    # 원문 추출
    nm = item.get("nm") or item.get("name") or item.get("res_name") or item.get("title")
    addr_raw = item.get("addr") or item.get("address")
    road_addr_raw = item.get("road_addr") or item.get("roadAddress") or item.get("road_addr_full")
    phone = item.get("phone") or item.get("tel") or item.get("telephone")
    category_raw = item.get("category") or item.get("cate") or item.get("type")
    branch = item.get("branch")
    v_rid = item.get("v_rid")

    # 정규화
    addr = normalize_jeju(addr_raw)
    road_addr = normalize_jeju(road_addr_raw)

    keyword_raw = _join_terms(item.get("keyword"))
    area = _join_area(item.get("area"))
    open_status = item.get("open_status") or item.get("status")
    score = item.get("score")
    user_score = item.get("user_score")
    review_cnt = item.get("review_cnt")

    lat = _safe_float(item.get("lat"))
    lng = _safe_float(item.get("lng"))
    image = item.get("image")
    image_list = item.get("image_list")

    rv = item.get("display_review") or {}
    if isinstance(rv, dict):
        review_user = rv.get("user_nm")
        review_text = rv.get("review_cont")
        review_dt = rv.get("review_reg_dt")
    else:
        review_user = review_text = review_dt = None

    dist_km_val = None
    if near and lat is not None and lng is not None:
        dist_km_val = _haversine_km(near[0], near[1], lat, lng)

    # nm_json (아이템 내 en/zh 우선 고려)
    nm_en0 = item.get("nm_en") or item.get("name_en")
    nm_zh0 = item.get("nm_zh") or item.get("name_zh")
    if isinstance(nm_en0, str) and nm_en0.strip() and isinstance(nm_zh0, str) and nm_zh0.strip():
        nm_json_obj = {"ko": (nm or "").strip(), "en": nm_en0.strip(), "zh": nm_zh0.strip()}
    else:
        nm_json_obj = build_multilang_json(nm, translate_mode, transctl)
    nm_json_str = json.dumps(nm_json_obj, ensure_ascii=False, separators=(",", ":"))

    # addr_json / road_addr_json / category_json / keyword_json
    addr_json_str = json.dumps(build_multilang_json(addr, translate_mode, transctl), ensure_ascii=False, separators=(",", ":"))
    road_addr_json_str = json.dumps(build_multilang_json(road_addr, translate_mode, transctl), ensure_ascii=False, separators=(",", ":"))
    category_json_str = json.dumps(build_multilang_json(category_raw or "", translate_mode, transctl), ensure_ascii=False, separators=(",", ":"))
    keyword_json_str  = json.dumps(build_multilang_json(keyword_raw or "", translate_mode, transctl), ensure_ascii=False, separators=(",", ":"))

    base_map = {
        "nm": nm or "",
        "nm_json": nm_json_str,
        "addr": addr,
        "addr_json": addr_json_str,
        "road_addr": road_addr,
        "road_addr_json": road_addr_json_str,
        "phone": phone or "",
        "category": category_raw or "",
        "category_json": category_json_str,
        "keyword": keyword_raw,
        "keyword_json": keyword_json_str,
        "area": area,
        "open_status": open_status or "",
        "score": score if score is not None else "",
        "user_score": user_score if user_score is not None else "",
        "review_cnt": review_cnt if review_cnt is not None else "",
        "branch": branch or "",
        "v_rid": v_rid or "",
        "lat": lat if lat is not None else "",
        "lng": lng if lng is not None else "",
        "image": image or "",
        "image_list": ", ".join(image_list) if isinstance(image_list, list) else (image_list or ""),
        "review_user": review_user or "",
        "review_text": review_text or "",
        "review_dt": review_dt or "",
        "dist_km": round(dist_km_val, 3) if dist_km_val is not None else "",
    }

    return {c: base_map.get(c, "") for c in want_cols}

def gather_files(dir_path: Optional[str], file_patterns: List[str]) -> List[str]:
    files: List[str] = []
    if dir_path:
        files.extend(sorted(glob.glob(os.path.join(dir_path, "api_page*.json"))))
    for pat in file_patterns:
        files.extend(sorted(glob.glob(pat)))
    return sorted(list(dict.fromkeys(files)))

def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON 로드 실패: {path} ({e})", file=sys.stderr)
        return None

def parse_cols(s: Optional[str]) -> List[str]:
    if not s:
        return DEFAULT_COLS[:]
    cols = [c.strip() for c in s.split(",") if c.strip()]
    unknown = [c for c in cols if c not in SUPPORTED_COLS]
    if unknown:
        print(f"[WARN] 지원하지 않는 컬럼: {', '.join(unknown)} (무시됨).", file=sys.stderr)
        cols = [c for c in cols if c in SUPPORTED_COLS]
    return cols or DEFAULT_COLS[:]

def parse_sort(s: Optional[str]) -> List[Tuple[str, bool]]:
    if not s:
        return []
    pairs: List[Tuple[str,bool]] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            field, order = token.split(":", 1)
            asc = order.strip().lower() != "desc"
        else:
            field, asc = token, True
        pairs.append((field.strip(), asc))
    return pairs

def _row_key_for_sort(row: Dict[str, Any], field: str):
    v = row.get(field, "")
    try:
        return float("-inf") if v == "" else float(v)
    except Exception:
        return str(v)

def sort_rows(rows: List[Dict[str, Any]], orders: List[Tuple[str,bool]]) -> List[Dict[str, Any]]:
    if not orders:
        return rows
    for field, asc in reversed(orders):
        rows.sort(key=lambda r: _row_key_for_sort(r, field), reverse=not asc)
    return rows

def dedup_rows(rows: List[Dict[str, Any]], keypref: str = "v_rid") -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = r.get(keypref) or (r.get("nm","") + "|" + r.get("addr",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def _pass_filters(r: Dict[str, Any], f: Dict[str, Any]) -> bool:
    if f.get("area"):
        if not any(a.strip() in r.get("area", "") for a in f["area"]):
            return False
    if f.get("open"):
        if r.get("open_status", "") not in f["open"]:
            return False
    if f.get("category"):
        cat = str(r.get("category", "")).lower()
        if not any(c.lower() in cat for c in f["category"]):
            return False

    def _num(v):
        try:
            return float(v)
        except Exception:
            return None

    ms = f.get("min_score")
    if ms is not None:
        v = _num(r.get("score"))
        if v is None or v < ms:
            return False

    mus = f.get("min_user_score")
    if mus is not None:
        v = _num(r.get("user_score"))
        if v is None or v < mus:
            return False

    mr = f.get("min_reviews")
    if mr is not None:
        v = _num(r.get("review_cnt"))
        if v is None or v < mr:
            return False

    if f.get("max_km") is not None:
        try:
            dist = float(r.get("dist_km"))
        except Exception:
            return False
        if dist > f["max_km"]:
            return False

    return True

def load_rows(paths: List[str],
              limit: Optional[int],
              dot_path: Optional[str],
              want_cols: List[str],
              near: Optional[Tuple[float,float]],
              filters: Dict[str, Any],
              do_dedup: bool,
              sort_orders: List[Tuple[str,bool]],
              translate_mode: str,
              transctl: Optional[TransCtl],
              progress_cb: Optional[Callable[[], None]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        doc = read_json(p)
        if doc is None:
            continue

        items = get_by_path(doc, dot_path) if dot_path else find_items_auto(doc)
        if not (isinstance(items, list) and items and isinstance(items[0], dict)):
            items = find_items_auto(doc)

        # 파일 단위 정보 출력(선택)
        print(f"[FILE] {p} items={len(items or [])}")

        for it in (items or []):
            # 비정상 아이템도 '처리'로 간주해 진행 카운트 갱신
            if not isinstance(it, dict):
                if progress_cb: progress_cb()
                continue

            r = extract_row(it, want_cols, near, translate_mode, transctl)
            if _pass_filters(r, filters):
                rows.append(r)

            if progress_cb: progress_cb()

            if limit is not None and len(rows) >= limit:
                # limit 도달 시 즉시 반환 (진행 콜백은 이미 호출됨)
                if do_dedup:
                    rows = dedup_rows(rows, "v_rid" if "v_rid" in want_cols else "nm")
                if sort_orders:
                    rows = sort_rows(rows, sort_orders)
                return rows

    if do_dedup:
        rows = dedup_rows(rows, "v_rid" if "v_rid" in want_cols else "nm")
    if sort_orders:
        rows = sort_rows(rows, sort_orders)
    if limit is not None:
        rows = rows[:limit]
    return rows

def _compute_widths(rows: List[Dict[str, Any]], headers: List[str]) -> Dict[str, int]:
    base_min = 6
    base_max = {
        "nm": 28, "nm_json": 46,
        "addr": 40, "addr_json": 46,
        "road_addr": 40, "road_addr_json": 46,
        "phone": 16,
        "category": 20, "category_json": 40,
        "keyword": 32, "keyword_json": 46,
        "area": 14, "open_status": 10, "score": 6, "user_score": 9,
        "review_cnt": 10, "branch": 10, "v_rid": 12, "lat": 9, "lng": 10,
        "image": 24, "image_list": 24, "review_user": 10, "review_text": 40,
        "review_dt": 10, "dist_km": 7
    }
    w: Dict[str,int] = {}
    for h in headers:
        maxlen = max([len(str(h))] + [len(str(r.get(h,""))) for r in rows]) if rows else len(h)
        cap = base_max.get(h, 18)
        w[h] = max(base_min, min(maxlen, cap))
    return w

def print_table(rows: List[Dict[str, Any]], headers: List[str]):
    if not rows:
        print("(표시할 항목이 없습니다)")
        return
    widths = _compute_widths(rows, headers)
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = []
        for h in headers:
            s = ellipsis(str(r.get(h, "")), widths[h])
            cells.append(s.ljust(widths[h]))
        print(" | ".join(cells))

def save_csv(rows: List[Dict[str, Any]], out_path: str, headers: List[str]):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})
    print(f"[OK] CSV 저장: {out_path} ({len(rows)}건)")

# ---------- 진행/카운트 유틸 ----------

def count_total_items(paths: List[str], dot_path: Optional[str]) -> int:
    total = 0
    for p in paths:
        doc = read_json(p)
        if doc is None:
            continue
        items = get_by_path(doc, dot_path) if dot_path else find_items_auto(doc)
        if not (isinstance(items, list) and items and isinstance(items[0], dict)):
            items = find_items_auto(doc)
        total += len(items or [])
    return total

# ---------- CLI ----------

def parse_near(s: Optional[str]) -> Optional[Tuple[float,float]]:
    if not s:
        return None
    try:
        lat_s, lng_s = s.split(",", 1)
        return (float(lat_s.strip()), float(lng_s.strip()))
    except Exception:
        raise SystemExit('--near 는 "lat,lng" 형식이어야 합니다. 예: --near "33.514,126.959"')

def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")

def _fmt_elapsed(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{seconds:.3f}s ({m}m {s:.3f}s)"

def main():
    # 타이머 시작
    start_wall = datetime.now().astimezone()
    start_perf = time.perf_counter()
    print(f"[TIME] 시작: {_fmt_dt(start_wall)}")

    ap = argparse.ArgumentParser(description="DiningCode JSON → 요약표 (선택 컬럼/필터/정렬/CSV)")
    ap.add_argument("--dir", default=None, help="폴더 경로 (api_page*.json 자동 스캔)")
    ap.add_argument("--files", nargs="*", default=[], help="파일/패턴 (공백 구분)")
    ap.add_argument("--limit", type=int, default=None, help="최대 출력 행 수")
    ap.add_argument("--csv", default=None, help="CSV 저장 경로")
    ap.add_argument("--path", default=None, help="아이템 리스트 dot-path (예: result_data.poi_section.list)")

    ap.add_argument("--cols", default=",".join(DEFAULT_COLS),
                    help=f"표시할 컬럼들 (지원: {', '.join(sorted(SUPPORTED_COLS))})")
    ap.add_argument("--sort", default=None, help='정렬 지정. 예: "score:desc,review_cnt:desc,user_score:desc"')
    ap.add_argument("--dedup", action="store_true", help="중복 제거 (v_rid→없으면 nm+addr)")

    ap.add_argument("--filter-area", default=None, help='지역 포함 필터(쉼표 구분). 예: "우도,애월"')
    ap.add_argument("--filter-open", default=None, help='영업상태 정확일치(쉼표 구분). 예: "영업 중,준비 중"')
    ap.add_argument("--filter-category", default=None, help='카테고리 부분일치(쉼표 구분). 예: "버거,짬뽕"')
    ap.add_argument("--min-score", type=float, default=None, help="최소 score")
    ap.add_argument("--min-user-score", type=float, default=None, help="최소 user_score")
    ap.add_argument("--min-reviews", type=int, default=None, help="최소 review_cnt")

    ap.add_argument("--near", default=None, help='거리 계산 기준점 "lat,lng" (예: "33.514,126.959")')
    ap.add_argument("--km", type=float, default=None, help="--near 와 함께 사용. dist_km <= km 만 표시")

    ap.add_argument("--translate", choices=["auto","romanize","off"], default="auto",
                    help="다국어 JSON 생성 모드: auto(외부 번역) / romanize(영문만 로마자) / off(번역 안 함)")
    ap.add_argument("--translate-timeout", type=float, default=2.5,
                    help="외부 번역 호출 타임아웃(초). 기본 2.5")
    ap.add_argument("--translate-max", type=int, default=40,
                    help="외부 번역 총 호출 상한(언어 합산). 기본 40")
    ap.add_argument("--translate-provider", choices=["auto_chain","googletrans","deep"], default="auto_chain",
                    help="번역 공급자: auto_chain(googletrans → deep) / googletrans / deep")
    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn",
                    help="중국어 변형: cn=간체(zh-CN), tw=번체(zh-TW)")

    # ★ 진행 로그: N개마다 출력 (10 또는 100 추천)
    ap.add_argument("--log-every", type=int, default=100,
                    help="아이템 N건마다 진행 로그 출력 (0이면 비활성). 예: --log-every 10")

    args = ap.parse_args()

    paths = gather_files(args.dir, args.files)
    if not paths:
        print("[ERR] 읽을 JSON 파일이 없습니다. --dir 또는 --files 지정", file=sys.stderr)
        # 종료 시간/소요 시간 출력 후 종료
        end_wall = datetime.now().astimezone()
        elapsed = time.perf_counter() - start_perf
        print(f"[TIME] 종료: {_fmt_dt(end_wall)}")
        print(f"[TIME] 총 소요: {_fmt_elapsed(elapsed)}")
        sys.exit(2)

    want_cols = parse_cols(args.cols)
    near = parse_near(args.near)

    # 필터 구성
    filters: Dict[str, Any] = {}
    if args.filter_area:
        filters["area"] = [s.strip() for s in args.filter_area.split(",") if s.strip()]
    if args.filter_open:
        filters["open"] = [s.strip() for s in args.filter_open.split(",") if s.strip()]
    if args.filter_category:
        filters["category"] = [s.strip() for s in args.filter_category.split(",") if s.strip()]
    if args.min_score is not None:
        filters["min_score"] = args.min_score
    if args.min_user_score is not None:
        filters["min_user_score"] = args.min_user_score
    if args.min_reviews is not None:
        filters["min_reviews"] = args.min_reviews
    if near and args.km is not None:
        if "dist_km" not in want_cols:
            want_cols = want_cols + ["dist_km"]
        filters["max_km"] = args.km

    sort_orders = parse_sort(args.sort)

    # 번역 컨트롤러 준비 (auto에서만 외부 호출; romanize/off는 외부 호출 없음)
    transctl: Optional[TransCtl] = None
    if args.translate == "auto":
        transctl = TransCtl(
            provider=args.translate_provider,
            timeout=args.translate_timeout,
            max_calls=args.translate_max,
            zh_variant=args.zh_variant
        )

    # -------- 진행 로그 준비 --------
    total_items: Optional[int] = None
    if args.log_every and args.log_every > 0:
        total_items = count_total_items(paths, args.path)
        print(f"[INFO] 대상 파일: {len(paths)}개, 추출 대상 아이템: {total_items}건")

    t0 = time.perf_counter()
    processed = 0

    def progress_cb():
        nonlocal processed
        processed += 1
        if not args.log_every or args.log_every <= 0:
            return
        if processed == 1 or (processed % args.log_every == 0) or (total_items and processed == total_items):
            elapsed = time.perf_counter() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            total_str = str(total_items) if total_items is not None else "?"
            tx = getattr(transctl, "calls_used", None) if transctl else None
            extra = f", tx_calls={tx}" if tx is not None else ""
            print(f"[PROG] items {processed}/{total_str} elapsed={elapsed:.1f}s rate={rate:.1f}/s{extra}")

    rows = load_rows(paths, args.limit, args.path, want_cols, near, filters,
                     args.dedup, sort_orders, args.translate, transctl,
                     progress_cb=progress_cb)

    if args.limit is not None and len(rows) >= args.limit:
        print(f"[NOTE] --limit={args.limit} 도달로 조기 종료됨.")

    print_table(rows, want_cols)
    if args.csv:
        save_csv(rows, args.csv, want_cols)

    # 타이머 종료
    end_wall = datetime.now().astimezone()
    elapsed = time.perf_counter() - start_perf
    print(f"[TIME] 종료: {_fmt_dt(end_wall)}")
    print(f"[TIME] 총 소요: {_fmt_elapsed(elapsed)}")

if __name__ == "__main__":
    main()
