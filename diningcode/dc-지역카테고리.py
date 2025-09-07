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

화면 출력:
- 처리 진행 상황: --log-every N (기본 100건마다)
- name / name_json(ko,en,cn) 프리뷰: --show-name-json true
- 지역 리포트: --show-region-report true

번역 옵션(지역적재와 동일):
  --translate {auto|romanize|off} (default: romanize)
  --translate-timeout 2.0
  --translate-max 200
  --translate-provider {auto_chain|googletrans|deep} (default: auto_chain)
  --zh-variant {cn|tw}
  --cache-file <path>

카테고리 번역 품질 향상:
- CATEGORY_GLOSSARY_EN/CN 사전 우선 적용
- 영어는 항상 Title Case로 정규화 (BBQ 등 전부 대문자 토큰 보존)

부트스트랩 친절 로그:
- 시드가 없으면 "skip augment" 안내를 출력 (첫 실행에 유용)
- 시드가 있으면 top-k / min-count 기준으로 사전 보강 후 추가/스킵 건수 출력
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

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

# ====== 번역/로마자 — 지역적재와 동일 + 제주 지명/카테고리 보강 ===================

_RE_HANGUL = re.compile(r"[가-힣]")
_L = ["g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"]
_V = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo","u","wo","we","wi","yu","eu","ui","i"]
_T = ["","k","k","ks","n","nj","nh","t","l","lk","lm","lb","ls","lt","lp","lh","m","p","ps","t","t","ng","t","t","k","t","p","t"]

_roman_cache: Dict[str, str] = {}
_trans_cache: Dict[Tuple[str, str], str] = {}   # (dest, norm_text) -> translated

def _norm_key(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip()).lower()

def has_hangul(s: str) -> bool:
    return bool(_RE_HANGUL.search(s or ""))

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
    def __init__(self, provider: str, timeout: float, max_calls: int, zh_variant: str,
                 trace: bool = False):
        self.provider = provider  # auto_chain | googletrans | deep
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
            if self.trace: print(f"[TX] googletrans → {dest}: '{text}'")
            out = _run_with_timeout(_try_googletrans, self.timeout, text, dest, "ko")
        if not out and self.provider in ("deep", "auto_chain"):
            if not self._can_call():
                return None
            self.calls_used += 1
            if self.trace: print(f"[TX] deep_translator → {dest}: '{text}'")
            out = _run_with_timeout(_try_deeptranslator, self.timeout, text, dest, "ko")
        if out:
            _trans_cache[key] = out
        return out

def load_trans_cache(path: Optional[str]):
    if not path: return
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cnt = 0
        if isinstance(d, dict):
            for dest, m in d.items():
                if not isinstance(m, dict): continue
                for norm_text, val in m.items():
                    _trans_cache[(dest, norm_text)] = val
                    cnt += 1
        print(f"[CACHE] 번역 캐시 로드: {cnt}건 from {path}")
    except FileNotFoundError:
        print(f"[CACHE] 파일 없음(새로 생성 예정): {path}")
    except Exception as e:
        print(f"[CACHE] 로드 실패({e}) — 무시하고 진행")

def save_trans_cache(path: Optional[str]):
    if not path: return
    out: Dict[str, Dict[str, str]] = {}
    for (dest, norm_text), val in _trans_cache.items():
        out.setdefault(dest, {})[norm_text] = val
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

    if translate_mode == "off":
        en = ""
    elif translate_mode == "romanize" or transctl is None:
        en = romanize_korean(ko) if has_hangul(ko) else ko
    else:
        if has_hangul(ko):
            t_en = transctl.translate(ko, "en")
            en = (t_en or "").strip() or romanize_korean(ko)
        else:
            en = ko
    en = titlecase_en(en)

    if translate_mode == "off":
        cn = ""
    else:
        jeju_cn = jeju_ko_to_cn(ko, zh_variant, transctl if translate_mode == "auto" else None)
        if jeju_cn:
            cn = jeju_cn
        else:
            if translate_mode == "romanize" or transctl is None:
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
        elif translate_mode == "romanize" or transctl is None:
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

# ---------------- 시드 보강 (친절 로그 포함) ----------------

def _pretty_seed(path: Optional[str]) -> str:
    return path if path else "(none)"

def augment_glossary_from_category_seed(seed_path: str,
                                        translate_mode: str,
                                        transctl: Optional[TransCtl],
                                        zh_variant: str,
                                        top_k: int,
                                        min_count: int) -> Tuple[int,int]:
    """categories.json 시드에서 상위 토큰을 사전에 주입."""
    data = read_json(seed_path)
    if not isinstance(data, list):
        print(f"[SEED] category-seed 형식이 리스트가 아님 → skip", file=sys.stderr)
        return (0,0)
    # 정렬(빈도 내림차순) 후 top-k + min-count
    rows = [x for x in data if isinstance(x, dict) and x.get("category")]
    rows.sort(key=lambda x: (-int(x.get("count", 0)), str(x.get("category"))))
    rows = [r for r in rows if int(r.get("count", 0)) >= int(min_count)]
    rows = rows[: int(top_k)]
    added = skipped = 0
    for r in rows:
        ko = str(r["category"]).strip()
        if ko in CATEGORY_GLOSSARY_EN and ko in CATEGORY_GLOSSARY_CN:
            skipped += 1
            continue
        nj = build_name_obj_category(ko, translate_mode, transctl, zh_variant)
        if ko not in CATEGORY_GLOSSARY_EN and nj.get("en"):
            CATEGORY_GLOSSARY_EN[ko] = nj["en"]
        if ko not in CATEGORY_GLOSSARY_CN and nj.get("cn"):
            CATEGORY_GLOSSARY_CN[ko] = nj["cn"]
        added += 1
    return (added, skipped)

def augment_jeju_from_region_seed(seed_path: str,
                                  translate_mode: str,
                                  transctl: Optional[TransCtl],
                                  zh_variant: str,
                                  top_k: int,
                                  min_count: int) -> Tuple[int,int]:
    """
    regions-addr-unique.json 시드에서 l1/l2/l3 텍스트를 스캔해
    JEJU_ZH_GLOSSARY_FULL_CN에 없으면 런타임 보강(속도/일관성 향상).
    """
    data = read_json(seed_path)
    if not isinstance(data, list):
        print(f"[SEED] region-seed 형식이 리스트가 아님 → skip", file=sys.stderr)
        return (0,0)
    rows = [x for x in data if isinstance(x, dict)]
    rows.sort(key=lambda x: (-int(x.get("count", 0)), str(x.get("l1","")), str(x.get("l2","")), str(x.get("l3",""))))
    rows = [r for r in rows if int(r.get("count", 0)) >= int(min_count)]
    rows = rows[: int(top_k)]

    uniq_terms: List[str] = []
    seen = set()
    for r in rows:
        for key in ("l1","l2","l3"):
            t = (r.get(key) or "").strip()
            if t and t not in seen:
                seen.add(t); uniq_terms.append(t)

    added = skipped = 0
    for ko in uniq_terms:
        if ko in JEJU_ZH_GLOSSARY_FULL_CN:
            skipped += 1
            continue
        nj = build_name_obj_region(ko, translate_mode, transctl, zh_variant)
        cn = nj.get("cn","")
        if cn:
            JEJU_ZH_GLOSSARY_FULL_CN[ko] = cn
            added += 1
        else:
            skipped += 1
    return (added, skipped)

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

            done += 1
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

    # 저장 옵션
    ap.add_argument("--save-region", default="true", choices=["true","false"], help="지역 결과 저장 (기본 true)")
    ap.add_argument("--save-category", default="true", choices=["true","false"], help="카테고리 결과 저장 (기본 true)")
    ap.add_argument("--out-addr-triplets", default="data/region/regions-addr-triplets.json")
    ap.add_argument("--out-road-triplets", default="data/region/regions-road-addr-triplets.json")
    ap.add_argument("--out-addr-unique", default="data/region/regions-addr-unique.json")
    ap.add_argument("--out-road-unique", default="data/region/regions-road-addr-unique.json")
    ap.add_argument("--out-category", default="data/category/categories.json")

    # 번역 옵션(지역적재와 동일)
    ap.add_argument("--translate", choices=["auto","romanize","off"], default="romanize")
    ap.add_argument("--translate-timeout", type=float, default=2.0)
    ap.add_argument("--translate-max", type=int, default=200)
    ap.add_argument("--translate-provider", choices=["auto_chain","googletrans","deep"], default="auto_chain")
    ap.add_argument("--zh-variant", choices=["cn","tw"], default="cn")
    ap.add_argument("--trace-translate", action="store_true")
    ap.add_argument("--cache-file", default=None, help="번역 결과 캐시 JSON 경로")

    # 🔹 시드 보강 옵션 (친절 로그 포함)
    ap.add_argument("--region-seed", default=None, help="regions-addr-unique.json 경로 (옵션)")
    ap.add_argument("--category-seed", default=None, help="categories.json 경로 (옵션)")
    ap.add_argument("--seed-top-k", type=int, default=400, help="시드 상위 k개만 보강")
    ap.add_argument("--seed-min-count", type=int, default=2, help="시드 최소 빈도")
    ap.add_argument("--no-seed-augment", action="store_true", help="시드 보강 비활성화")

    args = ap.parse_args()
    show = (args.show.lower() == "true")
    show_name = (args.show_name_json.lower() == "true")
    show_report = (args.show_region_report.lower() == "true")
    do_save_region = (args.save_region.lower() == "true")
    do_save_category = (args.save_category.lower() == "true")

    # 번역 컨트롤러 + 캐시
    transctl = None
    if args.translate == "auto":
        transctl = TransCtl(
            provider=args.translate_provider,
            timeout=args.translate_timeout,
            max_calls=args.translate_max,
            zh_variant=args.zh_variant,
            trace=args.trace_translate,
        )
    load_trans_cache(args.cache_file)

    # 🔹 부트스트랩 친절 로그 & 시드 보강
    if args.no_seed_augment:
        print("[SEED] augmentation disabled (--no-seed-augment)")
    else:
        # Region seed
        if not (args.region_seed and os.path.exists(args.region_seed)):
            print(f"[SEED] region-seed not found -> skip augment ({_pretty_seed(args.region_seed)})")
        else:
            a, s = augment_jeju_from_region_seed(
                args.region_seed, args.translate, transctl, args.zh_variant,
                args.seed_top_k, args.seed_min_count
            )
            print(f"[SEED] region-seed applied: added={a}, skipped={s} (src={args.region_seed})")
        # Category seed
        if not (args.category_seed and os.path.exists(args.category_seed)):
            print(f"[SEED] category-seed not found -> skip augment ({_pretty_seed(args.category_seed)})")
        else:
            a, s = augment_glossary_from_category_seed(
                args.category_seed, args.translate, transctl, args.zh_variant,
                args.seed_top_k, args.seed_min_count
            )
            print(f"[SEED] category-seed applied: added={a}, skipped={s} (src={args.category_seed})")

    # 입력 파일
    files = gather_files(args.files or [])
    if not files:
        print("[ERR] 입력 파일이 없습니다.", file=sys.stderr)
        end = time.perf_counter()
        print(f"[TIME] 종료: {now_local_str()}")
        print(f"[TIME] 총 소요: {end-start:.3f}s")
        save_trans_cache(args.cache_file)
        sys.exit(2)

    # 집계 (진행 로그 포함)
    (regions_addr_per_shop, regions_road_per_shop,
     addr_triplet_counter, road_triplet_counter, cat_counter) = process_files(
        files, args.path, args.log_every, transctl
    )

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

    save_trans_cache(args.cache_file)
    end = time.perf_counter()
    print(f"[TIME] 종료: {now_local_str()}")
    print(f"[TIME] 총 소요: {end-start:.3f}s")

if __name__ == "__main__":
    main()
