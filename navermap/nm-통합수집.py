# -*- coding: utf-8 -*-
"""
Naver Map crawler (Playwright, Chrome)

Step #1  네이버맵 접속
Step #2  검색어로 목록 수집 (이름/카테고리/평점/리뷰/상태/태그라인/광고여부/프로모/리스트썸네일들)
Step #3  좌측 목록의 a.place_bluelink 클릭 → 우측 entryIframe 상세에서
        상단(홈) 카드 소개문(description) 및 카테고리(category),
        주소(도로명/지번), 전화번호,
        정보 탭(place_section_content 하위 ul>li 및 li div)의 편의시설(노이즈 제거·중복 제거),
        메뉴 탭 클릭 후 메뉴명/가격(0원/무의미명 제외),
        사진 탭 클릭 후 상세 사진 목록 수집(특정 도메인 제외; 사진 탭의 place_section_content
        내부 a.place_thumb > img 우선 추출, 기본 5장)

출력:
- 목록: --out / --out-list   (예: data/list/jeju/*.json → *는 <검색어_타임스탬프>)
- 상세: --detail-out / --out-detail (예: data/detail/jeju/*.json → *는 <상호명-슬러그>)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError,
)

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Place:
    rank: int
    name: str
    category: str
    rating: str
    reviews: str
    status: str
    tagline: str
    ad: bool
    promo: str
    image: str = ""
    images: List[str] = field(default_factory=list)

@dataclass
class PlaceDetail:
    name: str
    description: str = ""    # 상단(홈) 카드의 소개 문구
    category: str = ""       # 상단(홈) 영역의 카테고리 (예: 카페,디저트)
    address: str = ""        # 최종 선택된 주소(도로명 우선)
    road_address: str = ""   # 도로명
    jibun_address: str = ""  # 지번
    tel: str = ""
    amenities: List[str] = field(default_factory=list)        # 정보 탭 기반
    menu: List[Dict[str, Any]] = field(default_factory=list)  # {name, price, raw}
    photos: List[str] = field(default_factory=list)           # 상세 사진 URL들

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(t: Optional[str]) -> str:
    if not t:
        return ""
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", str(t))
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def strip_after_copy_word(s: str) -> str:
    s = re.sub(r"\s*복사\s*$", "", s)
    s = re.sub(r"\s*\(복사\)\s*$", "", s)
    return s.strip()

def only_digits(s: Optional[str]) -> str:
    if not s:
        return ""
    m = re.search(r"(\d[\d,]*)", s.replace("+", ""))
    return m.group(1) if m else ""

def debug_print(enabled: bool, *args):
    if enabled:
        print("[nm]", *args)

def safe_text(locator, timeout_ms: int) -> str:
    try:
        return clean_text(locator.first.inner_text(timeout=timeout_ms))
    except Exception:
        try:
            t = locator.first.text_content(timeout=timeout_ms)
            return clean_text(t or "")
        except Exception:
            return ""

def slugify(text: str) -> str:
    s = re.sub(r"[^\w\-가-힣]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "item"

def resolve_out_path(spec: str, key: str, default_ext: str = ".json") -> str:
    """--out 규칙 처리 (* → key_타임스탬프)"""
    spec = os.path.expanduser(os.path.expandvars(spec or ""))
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{slugify(key)}_{ts}"

    if "*" in spec:
        path = spec.replace("*", name, 1)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        root, ext = os.path.splitext(path)
        if not ext:
            path = path + default_ext
        return path

    if spec.endswith(os.sep) or (os.path.exists(spec) and os.path.isdir(spec)):
        os.makedirs(spec, exist_ok=True)
        return os.path.join(spec, name + default_ext)

    d = os.path.dirname(spec)
    if d:
        os.makedirs(d, exist_ok=True)
    root, ext = os.path.splitext(spec)
    if not ext:
        spec = spec + default_ext
    return spec

def resolve_detail_path(spec: str, place_name: str, default_ext: str = ".json") -> str:
    """--detail-out 규칙 처리 (* → 상호명 슬러그)"""
    spec = os.path.expanduser(os.path.expandvars(spec or ""))
    token = slugify(place_name) or "place"
    if "*" in spec:
        path = spec.replace("*", token, 1)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        root, ext = os.path.splitext(path)
        if not ext:
            path = path + default_ext
        return path
    if spec.endswith(os.sep) or (os.path.exists(spec) and os.path.isdir(spec)):
        os.makedirs(spec, exist_ok=True)
        return os.path.join(spec, token + default_ext)
    d = os.path.dirname(spec)
    if d:
        os.makedirs(d, exist_ok=True)
    root, ext = os.path.splitext(spec)
    if not ext:
        spec = spec + default_ext
    return spec

# ──────────────────────────────────────────────────────────────────────────────
# Browser helpers
# ──────────────────────────────────────────────────────────────────────────────

def launch_browser(p, headless: bool, slow_mo: int = 0):
    try:
        return p.chromium.launch(
            channel="chrome",
            headless=headless,
            slow_mo=slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-dev-shm-usage",
            ],
        )
    except PWError:
        return p.chromium.launch(
            headless=headless,
            slow_mo=slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-dev-shm-usage",
            ],
        )

def wait_search_box(page, timeout_ms: int, verbose: bool):
    debug_print(verbose, "네이버맵 진입 및 검색 수행")
    candidates = [
        'input.input_search',
        'input[id^="input_search"]',
        'input[role="combobox"].input_search',
    ]
    deadline = time.time() + timeout_ms / 1000
    last_err = None
    while time.time() < deadline:
        for sel in candidates:
            try:
                el = page.query_selector(sel)
                if el:
                    return el
            except Exception as e:
                last_err = e
        page.wait_for_timeout(200)
    if last_err:
        debug_print(verbose, "search box error:", last_err)
    raise PWTimeout(f"Timeout {timeout_ms}ms: 검색창을 찾지 못했습니다.")

def wait_results_ready(page, timeout_ms: int, verbose: bool):
    debug_print(verbose, "search iframe/목록 대기")
    deadline = time.time() + timeout_ms / 1000
    sframe_loc = page.frame_locator("iframe#searchIframe")
    item_candidates = [
        '#_pcmap_list_scroll_container ul > li:has(a.place_bluelink)',
        '#_pcmap_list_scroll_container ul > li:has(span.TYaxT)',
        'ul > li:has(a.place_bluelink)',
        'li:has(span.TYaxT)',
    ]
    while time.time() < deadline:
        try:
            page.wait_for_selector("iframe#searchIframe", timeout=1000)
            break
        except PWTimeout:
            pass
    last_c = 0
    while time.time() < deadline:
        try:
            for sel in item_candidates:
                items = sframe_loc.locator(sel)
                c = items.count()
                if c > 0:
                    debug_print(verbose, f"results ready: {c} items via {sel}")
                    return sframe_loc
        except Exception:
            pass
        try:
            c0 = sframe_loc.locator(item_candidates[0]).count()
            if c0 != last_c:
                last_c = c0
                debug_print(verbose, f"loaded items: {c0}")
        except Exception:
            pass
        page.evaluate("window.scrollBy(0, 200)")
        page.wait_for_timeout(250)
    raise PWTimeout(f"Timeout {timeout_ms}ms: 목록 컨테이너/아이템을 찾지 못했습니다.")

def get_frame_by_selector(page, selector: str, timeout_ms: int):
    page.wait_for_selector(selector, timeout=timeout_ms)
    el = page.query_selector(selector)
    if not el:
        raise PWTimeout(f"iframe not found: {selector}")
    fr = el.content_frame()
    if not fr:
        raise PWTimeout(f"content_frame() failed: {selector}")
    return fr

# ──────────────────────────────────────────────────────────────────────────────
# List scraping
# ──────────────────────────────────────────────────────────────────────────────

def extract_images_from_list_li(li, per_item_timeout: int, max_images: int) -> List[str]:
    urls: List[str] = []
    try:
        scope = li.locator("a.place_thumb img")
        cnt = scope.count()
        for j in range(min(cnt, max_images)):
            img = scope.nth(j)
            src = None
            for attr in ("src", "data-src", "data-lazy", "data-original"):
                try:
                    src = img.get_attribute(attr, timeout=per_item_timeout)
                except Exception:
                    src = src or None
                if src:
                    break
            if src:
                urls.append(src.split("?")[0])
    except Exception:
        pass
    return urls

def extract_place_from_li(li, per_item_timeout: int, verbose: bool, max_images: int) -> Optional[Place]:
    try:
        try:
            li.scroll_into_view_if_needed(timeout=per_item_timeout)
        except Exception:
            pass

        name = ""
        if li.locator(".TYaxT").count():
            name = safe_text(li.locator(".TYaxT"), per_item_timeout)
        if not name and li.locator("a.place_bluelink").count():
            name = safe_text(li.locator("a.place_bluelink"), per_item_timeout)
        if not name and li.locator("a.place_bluelink").count():
            try:
                name = clean_text(li.locator("a.place_bluelink").first.get_attribute("title") or "")
            except Exception:
                pass
        if not name:
            return None

        # 스팸/광고용 항목 스킵
        bad = ["내 업체 등록", "업체 등록", "장소 등록"]
        if any(b in name for b in bad):
            return None

        category = ""
        if li.locator(".KCMnt").count():
            category = safe_text(li.locator(".KCMnt"), per_item_timeout)
        if category and name.endswith(category):
            name = clean_text(name.replace(category, ""))

        promo = ""
        if li.locator("a.nyHXH").count():
            promo = safe_text(li.locator("a.nyHXH"), per_item_timeout)

        rating = ""
        reviews = ""
        status = ""
        tagline = ""
        mv_scope = li.locator(".Dr_06 .MVx6e")
        if mv_scope.count():
            try:
                parts = [clean_text(t) for t in mv_scope.locator(".h69bs").all_text_contents()]
            except Exception:
                parts = []
            status = next((t for t in parts if ("영업" in t or "휴무" in t)), "")
            rating = next((t for t in parts if re.search(r"\b\d+(\.\d+)?\b", t) and "리뷰" not in t and "영업" not in t), "")
            reviews = next((t for t in parts if "리뷰" in t), "")
            reviews = only_digits(reviews)
            tagline_candidates = [t for t in parts if t not in {status} and "리뷰" not in t and t != rating and t]
            tagline = tagline_candidates[-1] if tagline_candidates else ""

        ad = False
        if li.locator(".gU6bV .place_blind").count():
            try:
                ad = any("광고" in t for t in li.locator(".gU6bV .place_blind").all_text_contents())
            except Exception:
                ad = False

        images = extract_images_from_list_li(li, per_item_timeout=per_item_timeout, max_images=max_images)
        image = images[0] if images else ""

        return Place(
            rank=0,
            name=name,
            category=category,
            rating=rating,
            reviews=reviews,
            status=status,
            tagline=tagline,
            ad=ad,
            promo=promo,
            image=image,
            images=images,
        )
    except Exception as e:
        debug_print(verbose, f"extract error: {e}")
        return None

def scroll_and_collect(page, sframe_loc, max_items: int, max_scrolls: int, pause_ms: int,
                       verbose: bool, per_item_timeout: int, max_images: int) -> List[Place]:
    debug_print(verbose, "스크롤 수집 시작")
    item_selectors = [
        '#_pcmap_list_scroll_container ul > li:has(a.place_bluelink)',
        '#_pcmap_list_scroll_container ul > li:has(span.TYaxT)',
        'ul > li:has(a.place_bluelink)',
        'li:has(span.TYaxT)',
    ]
    container_selectors = [
        "#_pcmap_list_scroll_container",
        "div.Ryr1F#_pcmap_list_scroll_container",
    ]

    cont = None
    for csel in container_selectors:
        node = sframe_loc.locator(csel)
        if node.count():
            cont = node
            break

    def get_items_locator():
        for sel in item_selectors:
            loc = sframe_loc.locator(sel)
            if loc.count() > 0:
                return loc, sel
        return sframe_loc.locator(item_selectors[0]), item_selectors[0]

    places: List[Place] = []
    seen_names = set()
    last_len = 0

    for s in range(max_scrolls):
        items, using_sel = get_items_locator()
        count_now = items.count()
        debug_print(verbose, f"scan[{s}] items={count_now} via {using_sel}")

        for i in range(count_now):
            if len(places) >= max_items:
                break
            li = items.nth(i)
            pz = extract_place_from_li(
                li, per_item_timeout=per_item_timeout, verbose=verbose, max_images=max_images
            )
            if not pz:
                continue
            if pz.name in seen_names:
                continue
            pz.rank = len(places) + 1
            places.append(pz)
            seen_names.add(pz.name)

        if len(places) >= max_items:
            break

        if count_now == last_len:
            debug_print(verbose, f"no more new items after scroll {s}")
            break
        last_len = count_now

        if cont and cont.count():
            try:
                cont.evaluate("el => el.scrollTo({top: el.scrollHeight, behavior: 'instant'})")
            except Exception:
                pass
        else:
            try:
                page.evaluate("window.scrollBy(0, 1000)")
            except Exception:
                pass

        page.wait_for_timeout(pause_ms)

    return places

# ──────────────────────────────────────────────────────────────────────────────
# Detail scraping (설명/카테고리/주소/편의시설/메뉴/사진)
# ──────────────────────────────────────────────────────────────────────────────

def click_place_on_list(sframe_loc, name: str, per_item_timeout: int, verbose: bool):
    """좌측 목록에서 상호명을 클릭"""
    anchor = sframe_loc.locator('a.place_bluelink', has_text=re.compile(re.escape(name), re.I))
    if anchor.count() == 0:
        target = sframe_loc.locator('li:has(span.TYaxT)', has_text=re.compile(re.escape(name), re.I)).first
        target.scroll_into_view_if_needed(timeout=per_item_timeout)
        target.click(timeout=per_item_timeout)
        return
    el = anchor.first
    try:
        el.scroll_into_view_if_needed(timeout=per_item_timeout)
    except Exception:
        pass
    el.click(timeout=per_item_timeout)

def wait_entry_for_name(page, name: str, timeout_ms: int, verbose: bool):
    """entryIframe 로드 및 상호명 확인"""
    deadline = time.time() + timeout_ms / 1000
    last = ""
    while time.time() < deadline:
        try:
            fr = get_frame_by_selector(page, "iframe#entryIframe", 2_000)
        except Exception as e:
            last = f"no entryIframe yet: {e}"
            page.wait_for_timeout(200)
            continue
        try:
            cand = [
                f':is(h1,h2,span).Fc1rA:has-text("{name}")',
                f':is(h1,h2,span):has-text("{name}")',
                "h1, h2, .Fc1rA, ._3XamX, .IH7VW"
            ]
            for sel in cand:
                loc = fr.locator(sel)
                if loc.count():
                    texts = [clean_text(t) for t in loc.all_text_contents()][:10]
                    if any(name in t for t in texts):
                        return fr
        except Exception as e:
            last = f"entry check err: {e}"
        page.wait_for_timeout(250)
    raise PWTimeout(f"Timeout {timeout_ms}ms: entryIframe not matched for name. last={last}")

# ===== 상단(홈) 카드의 소개문 추출 =====
def extract_intro_description(fr, verbose: bool) -> str:
    """
    기본 상세(홈) 상단 카드에 노출되는 소개 문구(예: '시그니처 음료와 함께하는 제주 여행')를 수집.
    """
    for sel in [
        "div.XtBbS",
        "div._3Vx7s",
        "section:has(a[role='button']):nth-of-type(1) div"
    ]:
        try:
            loc = fr.locator(sel).first
            if loc and loc.count():
                t = clean_text(loc.inner_text(timeout=600))
                if 8 <= len(t) <= 180 and not re.search(r"(리뷰|영업|쿠폰|예약|주문|길찾기|출발|도착|공유|전화|네이버)", t):
                    debug_print(verbose, f"description(found@{sel}): {t[:40]}{'...' if len(t)>40 else ''}")
                    return t
        except Exception:
            pass

    try:
        nodes = fr.locator("div")
        n = min(120, nodes.count())
        for i in range(n):
            try:
                t = clean_text(nodes.nth(i).inner_text(timeout=150))
            except Exception:
                continue
            if not t or len(t) < 8 or len(t) > 180:
                continue
            if re.search(r"(리뷰|영업|쿠폰|예약|주문|저장|길찾기|출발|도착|공유|전화|네이버|원)", t):
                continue
            if re.fullmatch(r"[가-힣A-Za-z0-9\s\.,·\u00B7\-!~]+", t):
                debug_print(verbose, f"description(heuristic): {t[:40]}{'...' if len(t)>40 else ''}")
                return t
    except Exception:
        pass
    return ""

# ===== 상단(홈) 영역 카테고리 추출 =====
_CAT_STOP = {"리뷰","영업","쿠폰","예약","주문","길찾기","출발","도착","공유","전화","네이버","사진","메뉴","정보","홈"}
_CAT_HINT = re.compile(
    r"(카페|디저트|한식|중식|일식|양식|분식|고기|흑돼지|해산물|횟집|치킨|술집|펍|바|뷔페|국수|칼국수|떡볶이|제과|베이커리|빵|피자|파스타|샌드위치|버거|스테이크|아시아|인도|태국|베트남|멕시코|브런치|포차|호프|초밥|스시|라멘|돈까스|구이|국밥|보쌈|족발|찜닭|오마카세|카레|샤브샤브)",
    re.I,
)

def _normalize_category_text(t: str) -> str:
    t = clean_text(t)
    parts = re.split(r"[,\u00B7·/|]+|\s{2,}", t)
    out, seen = [], set()
    for p in parts:
        p = clean_text(p)
        if not p or p in _CAT_STOP:
            continue
        if not _CAT_HINT.search(p):
            if not (re.fullmatch(r"[가-힣]{1,8}", p) and len(p) >= 2):
                continue
        if p not in seen:
            out.append(p)
            seen.add(p)
    return ", ".join(out)

def extract_category_detail(fr, verbose: bool) -> str:
    # 0) 타이틀 영역 고정 패턴 (#_title 내부)
    title_scope = fr.locator('#_title, [id="_title"]')
    try:
        if title_scope and title_scope.count():
            # a) 가장 신뢰도 높은 전용 클래스(소문자 l, 대문자 i 모두 지원)
            for sel in (".lnJFt", ".InJfT", 'span[class*="lnJFt"]', 'span[class*="InJfT"]'):
                loc = title_scope.locator(sel).first
                if loc and loc.count():
                    raw = clean_text(loc.inner_text(timeout=500))
                    cat = _normalize_category_text(raw)
                    if cat:
                        debug_print(verbose, f"category(found@#_title {sel}): {cat}")
                        return cat
            # b) GHAhO(상호) 다음에 오는 스팬 텍스트를 후보로
            spans = title_scope.locator("span")
            gh_idx = -1
            for i in range(min(12, spans.count())):
                clz = spans.nth(i).get_attribute("class") or ""
                if "GHAhO" in clz:
                    gh_idx = i
                    break
            if gh_idx >= 0 and spans.count() > gh_idx + 1:
                try:
                    raw = clean_text(spans.nth(gh_idx + 1).inner_text(timeout=400))
                    cat = _normalize_category_text(raw)
                    if cat:
                        debug_print(verbose, f"category(neighbor#_title): {cat}")
                        return cat
                except Exception:
                    pass
    except Exception:
        pass

    # 1) 기타 대표 클래스들
    for sel in [
        'div[class*="title"] span.lnJFt',
        'div[class*="title"] span.InJfT',
        'span.lnJFt', 'span.InJfT',
        'div[class*="title"] .KCMnt', '.KCMnt',
    ]:
        try:
            loc = fr.locator(sel).first
            if loc and loc.count():
                raw = clean_text(loc.inner_text(timeout=500))
                cat = _normalize_category_text(raw)
                if cat:
                    debug_print(verbose, f"category(found@{sel}): {cat}")
                    return cat
        except Exception:
            pass

    # 2) 상단 영역 휴리스틱
    try:
        head = fr.locator('div[class*="title"]').first
        nodes = (head.locator("span,div,a") if head and head.count() else fr.locator("span,div,a"))
        n = min(80, nodes.count())
        for i in range(n):
            try:
                t = clean_text(nodes.nth(i).inner_text(timeout=120))
            except Exception:
                continue
            if not t or len(t) > 40:
                continue
            if any(s in t for s in _CAT_STOP):
                continue
            cat = _normalize_category_text(t)
            if cat:
                debug_print(verbose, f"category(heuristic): {cat}")
                return cat
    except Exception:
        pass
    return ""

# ===== 상단 탭 클릭 =====
def click_top_tab(fr, label_regex: str, verbose: bool, timeout: int = 2000) -> bool:
    try:
        cands = [
            fr.locator('.place_fixed_maintab.place_stuck :is(button,a,[role="tab"])', has_text=re.compile(label_regex)),
            fr.locator(':is(button,a,[role="tab"])', has_text=re.compile(label_regex)),
        ]
        for c in cands:
            if c.count():
                btn = c.first
                try:
                    btn.scroll_into_view_if_needed(timeout=800)
                except Exception:
                    pass
                btn.click(timeout=1000)
                debug_print(verbose, f"상단 탭 클릭: {label_regex}")
                try:
                    fr.wait_for_timeout(200)
                    fr.wait_for_selector(":is(section,div)", timeout=timeout)
                except Exception:
                    pass
                return True
    except Exception:
        pass
    return False

def click_menu_tab(fr, verbose: bool):
    return click_top_tab(fr, r"메뉴", verbose)

def click_info_tab(fr, verbose: bool):
    ok = click_top_tab(fr, r"(정보|상세정보|소개)", verbose)
    try:
        if ok:
            fr.wait_for_selector("div.place_section_content", timeout=1500)
    except Exception:
        pass
    return ok

def click_photos_tab(fr, verbose: bool):
    return click_top_tab(fr, r"(사진|갤러리)", verbose)

# ===== 주소 펼치기(아코디언) & 주소 파트 =====
def _expand_address_if_collapsed(fr, verbose: bool):
    try:
        btn = fr.locator(':is(a,button).PkgBl[aria-expanded="false"], :is(a,button)[aria-expanded="false"]:has(.LDgIH)')
        if btn and btn.count():
            try:
                btn.first.scroll_into_view_if_needed(timeout=600)
            except Exception:
                pass
            btn.first.click(timeout=1000)
            fr.wait_for_selector(':is(a,button)[aria-expanded="true"]', timeout=1500)
            debug_print(verbose, "주소 아코디언 펼침")
    except Exception:
        pass

def _harvest_address_rows_by_label(fr) -> List[Tuple[str, str]]:
    rows = []
    try:
        for row in fr.locator("div.nQ7Lh").all():
            try:
                label = clean_text(row.locator("span.TjXg1").first.text_content())
            except Exception:
                label = ""
            body = clean_text(row.inner_text())
            if label:
                val = clean_text(body.replace(label, "", 1))
            else:
                val = body
            val = strip_after_copy_word(val)
            if label and val:
                rows.append((label, val))
    except Exception:
        pass
    return rows

def _harvest_address_rows_label_agnostic(fr) -> List[Tuple[str, str]]:
    pairs = []
    try:
        nodes = fr.locator(
            ':is(div,li,section,p):has(span:has-text("도로명")), '
            ':is(div,li,section,p):has(span:has-text("지번"))'
        )
        for i in range(min(nodes.count(), 20)):
            node = nodes.nth(i)
            txt = clean_text(node.inner_text())
            txt = strip_after_copy_word(txt)
            for lab, val, _next in re.findall(r"(도로명|지번)\s*([^\s].+?)(?=\s*(도로명|지번|$))", txt):
                pairs.append((lab, clean_text(val)))
            if not pairs:
                for lab in ("도로명", "지번"):
                    if txt.startswith(lab):
                        pairs.append((lab, clean_text(txt.replace(lab, "", 1))))
    except Exception:
        pass
    return pairs

def _harvest_address_from_amen_block(fr) -> List[Tuple[str, str]]:
    pairs = []
    try:
        secs = fr.locator(
            ':is(div,section,li).O8qbU:has(.place_blind:has-text("주소")), '
            ':is(div,section,li).O8qbU:has(.place_blind:has-text("도로명")), '
            ':is(div,section,li).O8qbU:has(.place_blind:has-text("지번"))'
        )
        for i in range(min(secs.count(), 6)):
            sc = secs.nth(i)
            txt = clean_text(sc.inner_text())
            txt = strip_after_copy_word(txt)
            for line in [clean_text(x) for x in txt.splitlines()]:
                if not line:
                    continue
                if line.startswith("도로명"):
                    pairs.append(("도로명", clean_text(line.replace("도로명", "", 1))))
                elif line.startswith("지번"):
                    pairs.append(("지번", clean_text(line.replace("지번", "", 1))))
                elif "도로명" in line or "지번" in line:
                    m = re.findall(r"(도로명|지번)\s*([^\s].+?)(?=\s*(도로명|지번|$))", line)
                    for (lab, val, _next) in m:
                        pairs.append((lab, clean_text(val)))
    except Exception:
        pass
    return pairs

def _fallback_address_from_body(fr) -> str:
    try:
        body_text = fr.evaluate("() => document.body.innerText")
        lines = [clean_text(x) for x in (body_text or "").splitlines()]
        addr_like = [
            ln for ln in lines
            if (("제주" in ln) or re.search(r"(로|길|번길)\s*\d", ln)) and 8 <= len(ln) <= 120
        ]
        addr_like = [strip_after_copy_word(a) for a in addr_like]
        addr_like.sort(key=lambda x: (0 if ("로 " in x or "길 " in x or "번길" in x) else 1, len(x)))
        return addr_like[0] if addr_like else ""
    except Exception:
        return ""

def extract_address_block(fr, timeout_ms: int, verbose: bool):
    _expand_address_if_collapsed(fr, verbose)

    road = ""
    jibun = ""
    tel = ""

    for lab, val in _harvest_address_rows_by_label(fr):
        if lab.startswith("도로명") and not road:
            road = val
        elif lab.startswith("지번") and not jibun:
            jibun = val

    if not (road and jibun):
        for lab, val in _harvest_address_rows_label_agnostic(fr):
            if lab.startswith("도로명") and not road:
                road = val
            elif lab.startswith("지번") and not jibun:
                jibun = val

    if not (road and jibun):
        for lab, val in _harvest_address_from_amen_block(fr):
            if lab.startswith("도로명") and not road:
                road = val
            elif lab.startswith("지번") and not jibun:
                jibun = val

    # 전화번호
    try:
        tnode = fr.locator('div.O8qbU.nbXkr .xlx7Q').first
        if tnode and tnode.count():
            tel = clean_text(tnode.inner_text())
    except Exception:
        pass
    if not tel:
        try:
            tel_a = fr.locator('a[href^="tel:"]').first
            if tel_a and tel_a.count():
                href = tel_a.get_attribute("href") or ""
                m = re.search(r"tel:(.+)$", href)
                if m:
                    tel = m.group(1)
        except Exception:
            pass

    final = road or jibun
    if not final:
        final = _fallback_address_from_body(fr)

    road = strip_after_copy_word(road)
    jibun = strip_after_copy_word(jibun)
    final = strip_after_copy_word(final)

    debug_print(verbose, f"addr: road='{road}', jibun='{jibun}', final='{final}', tel='{tel}'")
    return final, road, jibun, tel

# ──────────────────────────────────────────────────────────────────────────────
# 편의시설(정보 탭) 수집 — 노이즈 제거/중복 제거
# ──────────────────────────────────────────────────────────────────────────────

_AMEN_STOPWORDS = {
    "이전", "다음", "이전 페이지", "다음 페이지", "페이지 닫기", "닫기",
    "이전페이지", "다음페이지", "펼쳐보기", "접기", "업주", "업주에게 문의",
    "찾아오는길", "이전/다음", "추천순", "최신순", "필터", "업체 정보 수정",
}

_AMEN_MAP = {
    "주차": r"(주차|발레|발렛)",
    "예약": r"(예약)",
    "단체 이용 가능": r"(단체.?이용|단체석|단체 가능)",
    "남녀 화장실 구분": r"(남.?녀\s*화장실|남녀.?화장실|화장실 구분)",
    "무선 인터넷": r"(무선.?인터넷|와이.?파이|wifi|Wi-?Fi)",
    "반려동물 동반": r"(반려동물|애견|펫|강아지)",
    "유아의자": r"(유아의자|아기의자|키즈의자)",
    "키즈존": r"(키즈.?존|놀이공간|놀이방)",
    "포장": r"(포장)",
    "배달": r"(배달)",
    "룸": r"(룸|개별실|개인실|프라이빗)",
    "좌식": r"(좌식|좌탁)",
    "흡연": r"(흡연)",
    "엘리베이터": r"(엘리베이터|리프트|승강기)",
}

def _is_noise_token(token: str) -> bool:
    t = clean_text(token)
    if not t:
        return True
    if t in _AMEN_STOPWORDS:
        return True
    if re.search(r"(이전|다음|펼쳐보기|접기|닫기|찾아오는길|페이지)", t):
        return True
    if len(t) > 40 or "원" in t:
        return True
    return False

def _amen_normalize(token: str) -> Optional[str]:
    t = clean_text(token)
    if _is_noise_token(t):
        return None
    for key, pattern in _AMEN_MAP.items():
        if re.search(pattern, t, re.I):
            return key
    if re.search(r"[가-힣]", t) and 2 <= len(t) <= 12:
        return t
    return None

def _amen_harvest_from_section_content(fr) -> List[str]:
    """정보 탭 내부 place_section_content > ul > li 및 li 하위 div 텍스트만 수집."""
    vals: List[str] = []
    try:
        container = fr.locator('div.place_section_content')
        if not container or not container.count():
            return vals

        li_nodes = container.locator('ul li')
        for i in range(min(400, li_nodes.count())):
            t = clean_text(li_nodes.nth(i).inner_text())
            if t:
                vals.append(t)

        div_nodes = container.locator('ul li div')
        for i in range(min(800, div_nodes.count())):
            t = clean_text(div_nodes.nth(i).inner_text())
            if t:
                vals.append(t)
    except Exception:
        pass
    return vals

def extract_amenities(fr, verbose: bool) -> List[str]:
    click_info_tab(fr, verbose)

    raw: List[str] = _amen_harvest_from_section_content(fr)

    norm: List[str] = []
    seen = set()
    for tok in raw:
        parts = re.split(r"[,\u00B7·/;，、]+|\s{2,}", tok)
        for it in parts:
            nt = _amen_normalize(it)
            if not nt:
                continue
            if nt not in seen:
                norm.append(nt)
                seen.add(nt)

    debug_print(verbose, f"amenities ({len(norm)}): {', '.join(norm[:10])}{'...' if len(norm)>10 else ''}")
    return norm

# ──────────────────────────────────────────────────────────────────────────────
# 메뉴/사진
# ──────────────────────────────────────────────────────────────────────────────

def _menu_name_meaningless(name: str) -> bool:
    n = clean_text(name)
    if not n or len(n) < 2:
        return True
    if re.fullmatch(r"0+|\d{1,3}", n):
        return True
    if n in {"메뉴", "기본", "선택", "선택1", "선택2"}:
        return True
    return False

def extract_menu(fr, limit: int, verbose: bool) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    click_menu_tab(fr, verbose)
    try:
        scope = fr.locator(':is(section,div):has-text("메뉴")')
        if not scope or not scope.count():
            scope = fr
        blocks = scope.locator("li, div")
        n = min(1500, max(0, blocks.count()))
        seen = set()
        for i in range(n):
            try:
                t = clean_text(blocks.nth(i).inner_text(timeout=300))
            except Exception:
                continue
            if ("원" not in t) or len(t) > 160 or len(t) < 2:
                continue
            m = re.search(r"(.{2,80}?)\s*([0-9][0-9,]*)\s*원", t)
            if not m:
                continue
            name = clean_text(m.group(1))
            price_str = clean_text(m.group(2)).replace(",", "")
            try:
                price_val = int(re.sub(r"\D", "", price_str))
            except Exception:
                continue
            if price_val <= 0:
                continue
            if _menu_name_meaningless(name):
                continue
            key = (name, price_val)
            if key in seen:
                continue
            items.append({"name": name, "price": str(price_val), "raw": t})
            seen.add(key)
            if len(items) >= limit:
                break
    except Exception as e:
        debug_print(verbose, f"menu parse err: {e}")
    return items

def extract_detail_photos(fr, max_cnt: int, verbose: bool) -> List[str]:
    """사진 탭에서 상세 이미지 URL을 수집한다.
    - place_section_content 내부 a.place_thumb > img 를 1순위로 탐색
    - https://search.pstatic.net/common/ 등 검색 CDN 이미지는 제외
    - data:, blob: 등 비정상 스킴 제외
    - 중복 제거, 최대 max_cnt(기본 5)
    """
    click_photos_tab(fr, verbose)
    urls: List[str] = []

    def pick_src(img):
        try:
            srcset = img.get_attribute("srcset")
            if srcset:
                first = srcset.split(",")[0].strip().split(" ")[0]
                if first:
                    return first
        except Exception:
            pass
        for attr in ("src", "data-src", "data-lazy", "data-original"):
            try:
                v = img.get_attribute(attr)
                if v:
                    return v
            except Exception:
                pass
        return None

    def consider(src: Optional[str]):
        if not src:
            return
        base = src.split("?")[0]
        if base.startswith(("data:", "blob:")):
            return
        if "search.pstatic.net/common/" in base:
            return
        if base not in urls:
            urls.append(base)

    # 1) 사진 탭의 섹션 컨텐츠 안에서 a.place_thumb > img 우선
    try:
        cont = fr.locator("div.place_section_content")
        if cont and cont.count():
            imgs = cont.locator("a.place_thumb img")
            for i in range(min(imgs.count(), max_cnt * 3)):
                consider(pick_src(imgs.nth(i)))
                if len(urls) >= max_cnt:
                    return urls
    except Exception:
        pass

    # 2) 보조: 사진/갤러리 표식이 있는 구역에서 img 수집
    try:
        scopes = fr.locator(':is(section,div,ul):has-text("사진"), :is(section,div,ul):has-text("갤러리")')
        for i in range(min(3, scopes.count())):
            imgs = scopes.nth(i).locator("img")
            for j in range(imgs.count()):
                consider(pick_src(imgs.nth(j)))
                if len(urls) >= max_cnt:
                    return urls
    except Exception:
        pass

    # 3) 최후: 프레임 전체에서 img 훑기
    try:
        imgs = fr.locator("img")
        for i in range(imgs.count()):
            consider(pick_src(imgs.nth(i)))
            if len(urls) >= max_cnt:
                return urls
    except Exception:
        pass

    return urls

def scrape_detail_for_place(page, name: str, timeout_ms: int, per_item_timeout: int,
                            menu_max: int, verbose: bool, max_detail_images: int) -> PlaceDetail:
    try:
        fr = wait_entry_for_name(page, name, timeout_ms, verbose)
    except PWTimeout:
        debug_print(verbose, "entry not ready, retry after small wait...")
        page.wait_for_timeout(600)
        fr = wait_entry_for_name(page, name, timeout_ms, verbose)

    desc = extract_intro_description(fr, verbose)
    cat  = extract_category_detail(fr, verbose)
    addr, road, jibun, tel = extract_address_block(fr, timeout_ms, verbose)
    ams = extract_amenities(fr, verbose)                     # 정보 탭 기반, 노이즈 제외
    menus = extract_menu(fr, menu_max, verbose)
    photos = extract_detail_photos(fr, max_detail_images, verbose)
    return PlaceDetail(
        name=name,
        description=desc,
        category=cat,
        address=addr,
        road_address=road,
        jibun_address=jibun,
        tel=tel,
        amenities=ams,
        menu=menus,
        photos=photos,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Crawl orchestration
# ──────────────────────────────────────────────────────────────────────────────

def crawl(query: str,
          headless: bool,
          max_items: int,
          out_spec: str,
          timeout_ms: int,
          verbose: bool,
          pause_ms: int,
          max_scrolls: int,
          slow_mo: int,
          screenshot_on_error: bool,
          per_item_timeout: int,
          max_images: int,
          detail_out_spec: Optional[str],
          menu_max: int,
          click_delay_ms: int,
          max_detail_images: int):

    list_out_path = resolve_out_path(out_spec, query, default_ext=".json")

    with sync_playwright() as p:
        browser = launch_browser(p, headless=headless, slow_mo=slow_mo)
        context = browser.new_context(
            locale="ko-KR",
            viewport={"width": 1440, "height": 900},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"),
        )
        page = context.new_page()
        page.set_default_timeout(max(10_000, timeout_ms))

        try:
            page.goto("https://map.naver.com", wait_until="load", timeout=60_000)

            search_input = wait_search_box(page, timeout_ms, verbose)
            search_input.click()
            search_input.fill(query)
            search_input.press("Enter")
            debug_print(verbose, "검색창 입력 성공")

            sframe_loc = wait_results_ready(page, timeout_ms, verbose)

            places = scroll_and_collect(
                page=page,
                sframe_loc=sframe_loc,
                max_items=max_items,
                max_scrolls=max_scrolls,
                pause_ms=pause_ms,
                verbose=verbose,
                per_item_timeout=per_item_timeout,
                max_images=max_images,
            )

            rows = [asdict(x) for x in places]
            if list_out_path.lower().endswith(".json"):
                with open(list_out_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
            else:
                fieldnames = [
                    "rank", "name", "category", "rating", "reviews", "status",
                    "tagline", "ad", "promo", "image", "images"
                ]
                with open(list_out_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in rows:
                        r = r.copy()
                        r["images"] = ";".join(r.get("images") or [])
                        writer.writerow(r)

            print(f"[nm] 목록 저장 완료: {len(rows)}개 -> {list_out_path}")

            # 상세 수집
            if detail_out_spec:
                print(f"[nm] 상세 수집 시작 ({len(places)}곳)")
                collected = 0
                for idx, pz in enumerate(places, start=1):
                    try:
                        if any(b in pz.name for b in ["내 업체 등록", "업체 등록", "장소 등록"]):
                            debug_print(verbose, f"detail skip(by name): {pz.name}")
                            continue
                        debug_print(verbose, f"detail[{idx}/{len(places)}] 클릭: {pz.name}")
                        click_place_on_list(sframe_loc, pz.name, per_item_timeout, verbose)
                        page.wait_for_timeout(max(150, click_delay_ms))
                        detail = scrape_detail_for_place(
                            page=page,
                            name=pz.name,
                            timeout_ms=timeout_ms,
                            per_item_timeout=per_item_timeout,
                            menu_max=menu_max,
                            verbose=verbose,
                            max_detail_images=max_detail_images,
                        )
                        dpath = resolve_detail_path(detail_out_spec, pz.name, default_ext=".json")
                        with open(dpath, "w", encoding="utf-8") as f:
                            json.dump(asdict(detail), f, ensure_ascii=False, indent=2)
                        collected += 1
                        debug_print(verbose, f"detail saved: {dpath}")
                    except Exception as de:
                        debug_print(verbose, f"detail fail [{pz.name}]: {de}")
                print(f"[nm] 상세 수집 완료: {collected}/{len(places)} 저장")

        except Exception as e:
            if screenshot_on_error:
                try:
                    page.screenshot(path="nm_error.png", full_page=True)
                    print("[nm] 에러 스크린샷 저장: nm_error.png")
                except Exception:
                    pass
            print(f"[nm] 에러 발생: {e}")
            raise
        finally:
            browser.close()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(prog="nm-목록수집.py")
    # 목록
    ap.add_argument("--query", required=True, help="네이버맵 검색어")
    ap.add_argument("--max", type=int, default=150, dest="max_items", help="최대 수집 개수(목록)")
    ap.add_argument("--out", "--out-list", dest="out", default="awol.json",
                    help="목록 저장 경로(.json/.csv). 디렉터리 또는 * 패턴 허용 (예: data/list/jeju/*.json)")
    ap.add_argument("--headless", action="store_true", help="브라우저 창 숨김")
    ap.add_argument("--show", action="store_true", help="브라우저 창 표시(headless 해제)")
    ap.add_argument("--timeout", type=int, default=32000, help="주요 대기 타임아웃(ms)")
    ap.add_argument("--per-item-timeout", type=int, default=3000, help="아이템 단위 텍스트/속성 추출 타임아웃(ms)")
    ap.add_argument("--delay", type=int, default=700, help="스크롤 간 대기(ms)")
    ap.add_argument("--max-scrolls", type=int, default=80, help="최대 스크롤 시도 횟수")
    ap.add_argument("--slowmo", type=int, default=0, help="Playwright slow_mo(ms)")
    ap.add_argument("--screenshot-on-error", action="store_true", help="에러 발생 시 스크린샷 저장")
    ap.add_argument("--max-images", type=int, default=3, help="목록 아이템별 최대 썸네일 수집 개수")
    ap.add_argument("--verbose", action="store_true", help="로그 상세 출력(진행 상황 표시)")

    # 상세
    ap.add_argument("--detail-out", "--out-detail", dest="detail_out", default="",
                    help="상세 저장 경로(.json). * 패턴 권장 (예: data/detail/jeju/* → *가 상호명으로 치환)")
    ap.add_argument("--menu-max", type=int, default=80, help="메뉴 최대 파싱 개수(휴리스틱)")
    ap.add_argument("--click-delay", type=int, default=400, help="목록 클릭 후 상세 로딩 대기(ms)")
    ap.add_argument("--max-detail-images", type=int, default=5, help="상세 사진 최대 수집 개수(기본 5장)")

    args = ap.parse_args()

    headless = args.headless
    if args.show:
        headless = False

    crawl(
        query=args.query,
        headless=headless,
        max_items=args.max_items,
        out_spec=args.out,
        timeout_ms=args.timeout,
        verbose=args.verbose,
        pause_ms=args.delay,
        max_scrolls=args.max_scrolls,
        slow_mo=args.slowmo,
        screenshot_on_error=args.screenshot_on_error,
        per_item_timeout=args.per_item_timeout,
        max_images=args.max_images,
        detail_out_spec=(args.detail_out or None),
        menu_max=args.menu_max,
        click_delay_ms=args.click_delay,
        max_detail_images=args.max_detail_images,
    )

if __name__ == "__main__":
    main()
