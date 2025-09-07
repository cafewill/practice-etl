#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DiningCode AJAX 크롤러 (isearch API)
- 브라우저가 보내는 폼데이터/헤더를 그대로 재현
- 사전 GET으로 쿠키/토큰 확보 후 POST
- 페이지네이션/재시도/백오프/로깅/저장
- 요청 간 딜레이: 기본 3.0초 + 매 요청마다 ±0~jitter(기본 2초) 랜덤 가감

Usage:
  python dc_isearch.py --query "제주도" --pages 5 --delay 3.0 --jitter 2.0 --out data/jeju --size 20 --order r_score
"""

import argparse
import json
import os
import re
import sys
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

LIST_URL = "https://www.diningcode.com/list.dc"
API_URL  = "https://im.diningcode.com/API/isearch/"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "origin": "https://www.diningcode.com",
    "referer": "https://www.diningcode.com/",
    "content-type": "application/x-www-form-urlencoded",
    "user-agent": UA,
    "connection": "keep-alive",
}

TOKEN_PATTERNS = [
    r'"token"\s*:\s*"(?P<val>[^"]+)"',
    r"token\s*=\s*'(?P<val>[^']+)'",
    r"token\s*=\s*\"(?P<val>[^\"]+)\"",
    r"data-token=\s*\"(?P<val>[^\"]+)\"",
]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def extract_token_from_html(html: str) -> Optional[str]:
    for pat in TOKEN_PATTERNS:
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            return m.group("val")
    return None

def get_cookies_and_token(session: requests.Session, query: str) -> Tuple[Dict[str, str], Optional[str]]:
    """검색 페이지를 한 번 열어 쿠키/토큰 확보"""
    params = {"query": query}
    r = session.get(
        LIST_URL,
        params=params,
        headers={"user-agent": UA, "referer": "https://www.diningcode.com/"},
        timeout=15
    )
    r.raise_for_status()
    token = extract_token_from_html(r.text)
    return session.cookies.get_dict(), token

def build_payload(
    query: str,
    page: int,
    size: int,
    order: str,
    token: Optional[str],
    lat: Optional[str],
    lng: Optional[str],
    rect: Optional[str],
    addr: str,
    keyword: str,
    bhour: str,
    mode: str,
    dc_flag: str,
    search_type: str,
    distance: str,
) -> Dict[str, str]:
    # 브라우저의 폼데이터를 그대로 따름. 빈값들은 빈 문자열로 전달.
    return {
        "query": query,
        "addr": addr or "",
        "keyword": keyword or "",
        "order": order or "r_score",
        "distance": distance or "",
        "search_type": search_type or "poi_search",
        "lat": lat or "",
        "lng": lng or "",
        "rect": rect or "",
        "bhour": bhour or "",
        "token": token or "",
        "mode": mode or "poi",
        "dc_flag": dc_flag or "1",
        "page": str(page),
        "size": str(size),
    }

def robust_post(session: requests.Session, url: str, data: Dict[str, str], headers: Dict[str, str], max_retries=4) -> requests.Response:
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, data=data, headers=headers, timeout=20)
            # 재시도 기준: 429/5xx
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"Transient HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            # 일부 케이스에서 JSON이 아닌 HTML이 오면 파싱 실패 → 재시도
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" not in ctype.lower():
                try:
                    _ = resp.json()
                except Exception:
                    raise requests.HTTPError("Expected JSON but got non-JSON", response=resp)
            return resp
        except Exception as e:
            if attempt >= max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("robust_post: unexpected fallthrough")

def normalize_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    응답 JSON의 스키마가 변할 수 있으니, 가능한 경로를 순서대로 시도.
    """
    if not isinstance(payload, dict):
        return []
    candidates = [
        ("result_data", "poi_section", "list"),  # 최근 구조 우선
        ("result", "list"),
        ("data", "list"),
        ("list",),
        ("results",),
        ("items",),
    ]
    for path in candidates:
        cur = payload
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list):
            return cur
    # 최후: dict 값들 중 list 하나 찾아서 리턴
    for v in payload.values():
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, list):
                    return vv
    return []

def pick_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    흔히 보이는 필드 이름 위주로 안전 추출.
    """
    def first(*names, default=None):
        for n in names:
            if n in item:
                return item[n]
        return default
    return {
        "id": first("id", "poi_id", "res_id", "v_rid", "rid", "poiId"),
        "name": first("name", "res_name", "title", "nm", "storeName"),
        "category": first("category", "cate", "type", "category_name"),
        "rating": first("rating", "score", "r_score", "user_score"),
        "reviews": first("review_count", "reviews", "cnt_review", "review_cnt"),
        "addr": first("addr", "address", "road_addr"),
        "lat": first("lat", "latitude"),
        "lng": first("lng", "longitude"),
        "url": first("url", "link"),
    }

def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_ndjson(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sleep_with_jitter(base_delay: float, jitter: float = 2.0) -> float:
    """
    base_delay(초)에 대해 [-jitter, +jitter] 범위의 랜덤 오프셋을 적용해 sleep.
    결과가 음수면 0으로 보정. 실제 대기 시간을 반환.
    """
    actual = base_delay + random.uniform(-jitter, jitter)
    if actual < 0:
        actual = 0.0
    time.sleep(actual)
    return actual

def crawl(query: str, pages: int, delay: float, jitter: float, out_dir: str, size: int, order: str):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ensure_dir(out_dir)
    session = requests.Session()
    # 1) 쿠키/토큰 확보
    try:
        cookies, token = get_cookies_and_token(session, query)
        print(f"[INIT] cookies={bool(cookies)} token={'***' if token else ''}")
    except Exception as e:
        print(f"[WARN] 초기 GET 실패: {e} (토큰 없이 진행)")
        token = None

    # 고정 헤더
    headers = dict(HEADERS)

    all_rows: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        payload = build_payload(
            query=query,
            page=page,
            size=size,
            order=order,
            token=token,
            lat=None, lng=None, rect=None,
            addr="",
            keyword="",
            bhour="",
            mode="poi",
            dc_flag="1",
            search_type="poi_search",
            distance="",
        )
        print(f"[API] page={page} payload={payload}")

        try:
            resp = robust_post(session, API_URL, data=payload, headers=headers, max_retries=4)
        except Exception as e:
            print(f"[ERROR] API 실패 (page={page}): {e}")
            if isinstance(e, requests.HTTPError) and e.response is not None:
                raw_path = os.path.join(out_dir, f"raw_page{page}_{ts}.html")
                try:
                    with open(raw_path, "wb") as f:
                        f.write(e.response.content)
                    print(f"[DUMP] 비정상 응답 저장: {raw_path}")
                except Exception:
                    pass
            # 실패했어도 다음 페이지 시도 전 슬립 적용(서버 보호용)
            if page < pages:
                actual = sleep_with_jitter(delay, jitter=jitter)
                print(f"[SLEEP] base={delay:.2f}s, jittered={actual:.2f}s")
            continue

        # JSON 파싱
        try:
            payload_json = resp.json()
        except Exception:
            raw_path = os.path.join(out_dir, f"raw_page{page}_{ts}.html")
            with open(raw_path, "wb") as f:
                f.write(resp.content)
            print(f"[WARN] JSON 아님 → 원본 저장: {raw_path}")
            if page < pages:
                actual = sleep_with_jitter(delay, jitter=jitter)
                print(f"[SLEEP] base={delay:.2f}s, jittered={actual:.2f}s")
            continue

        # 원본 JSON 항상 저장(디버깅용)
        save_json(os.path.join(out_dir, f"api_page{page}_{ts}.json"), payload_json)

        items = normalize_items(payload_json)
        print(f"[INFO] page={page} 수집 {len(items)}건")

        for it in items:
            try:
                all_rows.append(pick_fields(it))
            except Exception:
                all_rows.append({"_raw": it})

        if page < pages:
            actual = sleep_with_jitter(delay, jitter=jitter)
            print(f"[SLEEP] base={delay:.2f}s, jittered={actual:.2f}s")

    # 결과 저장
    ndjson_path = os.path.join(out_dir, f"result_{ts}.ndjson")
    save_ndjson(ndjson_path, all_rows)
    print(f"[DONE] 총 {len(all_rows)}건 저장 → {ndjson_path}")

def main():
    p = argparse.ArgumentParser(description="DiningCode isearch AJAX 크롤러")
    p.add_argument("--query", required=True, help="검색어 (예: 제주도)")
    p.add_argument("--pages", type=int, default=3, help="가져올 페이지 수 (default: 3)")
    p.add_argument("--delay", type=float, default=3.0, help="요청 간 딜레이(초) (default: 3.0)")
    p.add_argument("--jitter", type=float, default=2.0, help="지터(±초) (default: 2.0, 매 요청마다 랜덤 가감)")
    p.add_argument("--out", required=True, help="저장 폴더")
    p.add_argument("--size", type=int, default=20, help="페이지당 건수 (default: 20)")
    p.add_argument("--order", default="r_score", help="정렬 (기본 r_score)")
    args = p.parse_args()

    try:
        crawl(
            query=args.query,
            pages=args.pages,
            delay=args.delay,
            jitter=args.jitter,
            out_dir=args.out,
            size=args.size,
            order=args.order,
        )
    except KeyboardInterrupt:
        print("\n[ABORT] 사용자 중단")
        sys.exit(130)

if __name__ == "__main__":
    main()
