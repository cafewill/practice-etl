#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DiningCode 수집 json 병합 스크립트
- 대상: data/jeju/api_page*.json
- 구조: result_data.poi_section.list 에서 항목 추출 (첨부 파일 구조 기준)
- 중복 제거: v_rid 우선, 없으면 (nm, road_addr|addr) 조합
- 출력: {"list": [...]} 형태로 저장
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple, Set

def safe_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_list_from_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    첨부 구조 우선:
      obj["result_data"]["poi_section"]["list"]
    보강:
      obj["poi_section"]["list"]
      obj["list"]
    """
    # 1) result_data.poi_section.list (우선)
    rd = obj.get("result_data")
    if isinstance(rd, dict):
        ps = rd.get("poi_section")
        if isinstance(ps, dict):
            lst = ps.get("list")
            if isinstance(lst, list):
                return [x for x in lst if isinstance(x, dict)]

    # 2) poi_section.list
    ps = obj.get("poi_section")
    if isinstance(ps, dict):
        lst = ps.get("list")
        if isinstance(lst, list):
            return [x for x in lst if isinstance(x, dict)]

    # 3) 최상위 list
    lst = obj.get("list")
    if isinstance(lst, list):
        return [x for x in lst if isinstance(x, dict)]

    return []

def make_dedupe_key(item: Dict[str, Any]) -> Tuple[str, str]:
    v_rid = str(item.get("v_rid") or "").strip()
    if v_rid:
        return ("v_rid", v_rid)
    nm = str(item.get("nm") or "").strip()
    road = str(item.get("road_addr") or "") or str(item.get("addr") or "")
    return ("fallback", f"{nm}||{road}")

def main():
    ap = argparse.ArgumentParser(description="DiningCode JSON 'list' 병합기")
    ap.add_argument("--pattern", default="data/jeju/api_page*.json",
                    help="입력 파일 글롭 패턴 (기본: data/jeju/api_page*.json)")
    ap.add_argument("--out", default="merged_list.json",
                    help="출력 파일 경로 (기본: merged_list.json)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(f"[WARN] 패턴과 일치하는 파일이 없습니다: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    merged: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    print(f"[INFO] 병합 대상 파일 수: {len(paths)}")
    for p in paths:
        added = 0
        try:
            obj = safe_load(p)
        except Exception as e:
            print(f"[WARN] JSON 로드 실패: {p} ({e}) — 건너뜀", file=sys.stderr)
            continue

        items = get_list_from_obj(obj)
        if not items:
            # 상단 몇 키만 보여주며 힌트
            top_keys = ", ".join(list(obj.keys())[:8]) if isinstance(obj, dict) else str(type(obj))
            print(f"[WARN] list 영역을 찾지 못함: {p} (top: {top_keys}) — 건너뜀", file=sys.stderr)
            continue

        for it in items:
            key = make_dedupe_key(it)
            if key in seen:
                continue
            seen.add(key)
            merged.append(it)
            added += 1

        print(f"[INFO] {os.path.basename(p)}: +{added}개, 누적 {len(merged)}개")

    out_obj = {"list": merged}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 병합 완료 → {args.out} (총 {len(merged)}개)")

if __name__ == "__main__":
    main()
