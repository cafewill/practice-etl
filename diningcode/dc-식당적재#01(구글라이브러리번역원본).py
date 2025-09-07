#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jeju-Yeora — dc-식당적재.py (detail 파일 연동)
- 입력 JSON(list)로부터 restaurants, restaurant_menus, restaurant_category_mappings 적재
- 지역/카테고리 테이블은 최초 1회 로딩 후 메모리 캐시 사용
- 다국어 JSON 키 통일: {"ko","en","cn"}
- LV1 '제주'/'제주특별자치도' -> '제주도' 표준화, 주소 앞 3단어로 지역 매핑
- 상세 설명은 data/detail/[rid].json 에서 읽어옴
  - description  ← detail/[rid].json 의 description_json
  - short_description ← detail/[rid].json 의 short_description_json

사용 예:
python3 dc-식당적재.py \
  --files "20250823-merged-list-whole.json" \
  --config config.json --profile local \
  --translate auto --translate-provider auto_chain \
  --translate-timeout 2.0 --translate-max 200 \
  --zh-variant cn \
  --cache-file .cache/restaurant-trans.json \
  --region-table region \
  --owner-id 0 \
  --limit 0  # 0 또는 음수면 무제한 적재
"""

import argparse, glob, json, os, re, sys
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime

import pymysql

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


def now_local_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z")


def parse_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        return float(m.group(0)) if m else None


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
    # 안전한 토큰치환: '제주특별자치도'와 순수 토큰 '제주'만
    s = re.sub(r"제주특별자치도(?=\s|$)", "제주도", s)
    s = re.sub(r"^제주(?=\s)", "제주도", s)
    s = re.sub(r"(?<=\s)제주(?=\s)", "제주도", s)
    s = re.sub(r"(제주도)(\s+\1)+", r"\1", s)
    return _clean_ws(s)


def split_addr_3(addr: str) -> Tuple[str, str, str, str]:
    """
    addr → (L1, L2, L3, rest)
    L1/L2/L3은 주소 앞 3단어(제주도/제주시/우도면 등) 탐지, rest는 그 이후 상세
    """
    s = normalize_jeju(addr)
    if not s:
        return ("", "", "", "")
    toks = s.split()

    # L1
    l1 = ""
    for t in toks[:3]:
        if t == "제주도" or (t.endswith("도") and "제주" in t):
            l1 = "제주도"
            break
    # L2
    l2 = ""
    for t in toks:
        if RE_L2.search(t):
            l2 = t
            break
    # L3
    l3 = ""
    start = toks.index(l2) + 1 if l2 in toks else 1
    for t in toks[start : start + 6]:
        if RE_L3.search(t):
            l3 = t
            break
    if not l3:
        for t in toks[start : start + 6]:
            if RE_L3_ALT.search(t):
                l3 = t
                break

    if not l1 and len(toks) >= 1:
        l1 = toks[0]
    if not l2 and len(toks) >= 2:
        l2 = toks[1]
    if not l3 and len(toks) >= 3:
        l3 = toks[2]

    l1 = normalize_lv1(l1)
    prefix_cnt = 0
    if l1:
        prefix_cnt += 1
    if l2:
        prefix_cnt += 1
    if l3:
        prefix_cnt += 1
    rest = " ".join(toks[prefix_cnt:]) if prefix_cnt < len(toks) else ""

    return (_clean_ws(l1), _clean_ws(l2), _clean_ws(l3), _clean_ws(rest))


# ===================== 파일 탐색 =====================

def gather_files(patterns: List[str]) -> List[str]:
    out, seen = [], set()
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if p not in seen:
                out.append(p)
                seen.add(p)
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
    # DFS backup
    found: List[Dict[str, Any]] = []

    def dfs(x: Any):
        nonlocal found
        if found:
            return
        if isinstance(x, dict):
            for k, vv in x.items():
                if k in ("list", "items", "pois", "restaurants") and _looks_like_item_list(vv):
                    found = vv  # type: ignore
                    return
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
        if table_exists(cur, preferred):
            return preferred
        alt = "regions" if preferred == "region" else "region"
        if table_exists(cur, alt):
            eprint(f"[INFO] '{preferred}' 미존재 → '{alt}'로 자동 폴백")
            return alt
        raise RuntimeError(f"region(s) 테이블을 찾을 수 없습니다. 시도: '{preferred}', '{alt}'")


# ===================== 번역/캐시 =====================

_L = [
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
]
_V = [
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
]
_T = [
    "",
    "k",
    "k",
    "ks",
    "n",
    "nj",
    "nh",
    "t",
    "l",
    "lk",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "p",
    "ps",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "t",
]
_roman_cache: Dict[str, str] = {}
_trans_cache_mem: Dict[Tuple[str, str], str] = {}


def romanize_korean(text: str) -> str:
    key = norm_key(text)
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


class DiskCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.data: Dict[str, Dict[str, str]] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, text: str, dest: str) -> Optional[str]:
        if not self.path:
            return None
        key = norm_key(text)
        ent = self.data.get(key)
        if not ent:
            return None
        if dest == "cn":
            return ent.get("cn") or ent.get("zh")
        return ent.get(dest)

    def set(self, text: str, dest: str, value: str):
        if not self.path:
            return
        key = norm_key(text)
        ent = self.data.get(key) or {}
        ent[dest] = value
        if dest == "cn":
            ent["zh"] = value
        self.data[key] = ent

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


class TranslatorCtl:
    def __init__(
        self,
        mode: str,
        provider: str,
        timeout: float,
        max_calls: Optional[int],
        zh_variant: str,
        cache: DiskCache,
    ):
        self.mode = mode  # none | auto
        self.provider = provider  # auto_chain | googletrans | deep
        self.timeout = timeout
        self.max_calls = max_calls
        self.used_calls = 0
        self.zh_variant = "cn" if (zh_variant or "cn").lower().startswith("cn") else "tw"
        self.cache = cache

    def _can_call(self) -> bool:
        return self.max_calls is None or self.used_calls < self.max_calls

    def _google_dest(self) -> str:
        return "zh-cn" if self.zh_variant == "cn" else "zh-tw"

    def _deep_dest(self) -> str:
        return "zh-CN" if self.zh_variant == "cn" else "zh-TW"

    def _translate_googletrans(self, text: str, dest: str, src: str = "ko") -> Optional[str]:
        try:
            from googletrans import Translator  # type: ignore

            tr = Translator()
            res = tr.translate(text, src=src, dest=dest)
            return res.text if getattr(res, "text", None) else None
        except Exception:
            return None

    def _translate_deep(self, text: str, dest: str, src: str = "ko") -> Optional[str]:
        try:
            from deep_translator import GoogleTranslator  # type: ignore

            tr = GoogleTranslator(source=src, target=dest)
            return tr.translate(text)
        except Exception:
            return None

    def translate_once(self, text: str, dest_lang: str) -> str:
        if self.mode == "none" or not text:
            return text
        key = (dest_lang, norm_key(text))
        if key in _trans_cache_mem:
            return _trans_cache_mem[key]
        from_disk = self.cache.get(text, dest_lang)
        if from_disk:
            _trans_cache_mem[key] = from_disk
            return from_disk
        if not self._can_call():
            ans = romanize_korean(text) if (dest_lang == "en" and RE_ADDR_KO.search(text or "")) else text
            _trans_cache_mem[key] = ans
            self.cache.set(text, dest_lang, ans)
            return ans
        out: Optional[str] = None
        if self.provider == "auto_chain":
            if dest_lang == "en" and RE_ADDR_KO.search(text or ""):
                out = romanize_korean(text)
            if not out:
                if dest_lang == "cn":
                    out = self._translate_googletrans(text, self._google_dest())
                    if not out:
                        out = self._translate_deep(text, self._deep_dest())
                elif dest_lang == "en":
                    out = self._translate_googletrans(text, "en")
                    if not out:
                        out = self._translate_deep(text, "en")
        elif self.provider == "googletrans":
            out = self._translate_googletrans(text, "en" if dest_lang == "en" else self._google_dest())
        else:
            out = self._translate_deep(text, "en" if dest_lang == "en" else self._deep_dest())
        self.used_calls += 1
        if not out:
            out = romanize_korean(text) if (dest_lang == "en" and RE_ADDR_KO.search(text or "")) else text
        _trans_cache_mem[key] = out
        self.cache.set(text, dest_lang, out)
        return out

    def to_multi(self, text: str) -> Dict[str, str]:
        src = text or ""
        return {"ko": src, "en": self.translate_once(src, "en"), "cn": self.translate_once(src, "cn")}


# ===================== 지역/카테고리 로더 (캐시) =====================

def detect_name_column(cur, table: str) -> str:
    cur.execute(f"SHOW COLUMNS FROM `{table}`")
    cols = [r["Field"] for r in cur.fetchall()]
    for c in ("name", "name_ko", "ko_name", "label", "names", "nm", "title_ko"):
        if c in cols:
            return c
    return "name"


def extract_ko_from_value(val: Any) -> str:
    """DB에서 읽은 name 컬럼이 JSON 문자열/텍스트 모두 대응하여 ko 이름 추출."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # JSON?
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("ko", "KO", "kr", "KR"):
                    if k in obj and isinstance(obj[k], str):
                        return _clean_ws(obj[k])
                for v in obj.values():
                    if isinstance(v, str):
                        return _clean_ws(v)
        except Exception:
            pass
    return _clean_ws(s)


def load_region_map(conn, region_table_preferred: str) -> Dict[Tuple[str, str, str], int]:
    """
    region(s) 전체를 읽고 level=3 노드 기준으로 (l1,l2,l3)->id 맵 생성.
    이름은 ko 기준으로 비교. l1은 '제주도' 표준화.
    """
    table = resolve_region_table_name(conn, region_table_preferred)
    with conn.cursor() as cur:
        name_col = detect_name_column(cur, table)
        cur.execute(f"SELECT id, parent_id, level, `{name_col}` AS name FROM `{table}`")
        rows = cur.fetchall()
    nodes: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        nodes[r["id"]] = {
            "parent_id": r["parent_id"],
            "level": int(r["level"]),
            "name": extract_ko_from_value(r["name"]),
        }
    # parent chain
    triple_to_id: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        if int(r["level"]) != 3:
            continue
        n3 = nodes[r["id"]]
        p2_id = n3["parent_id"]
        n2 = nodes.get(p2_id)
        if not n2:
            continue
        p1_id = n2["parent_id"]
        n1 = nodes.get(p1_id)
        if not n1:
            continue
        l1 = normalize_lv1(n1["name"])  # 제주/제주특별자치도 → 제주도
        l2 = n2["name"]
        l3 = n3["name"]
        triple_to_id[(l1, l2, l3)] = r["id"]
    eprint(f"[INFO] region lv3 loaded(map size): {len(triple_to_id)} rows (table={table})")
    return triple_to_id


def load_categories(conn) -> Dict[str, int]:
    """
    restaurant_categories → {토큰(ko/en/cn/소문자/공백제거): id}
    텍스트/JSON name 모두 지원.
    """
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
        # ko/en/cn 다 찾아서 토큰화
        try:
            obj = json.loads(raw) if raw.startswith("{") else {}
        except Exception:
            obj = {}
        for k in ("ko", "KO", "en", "EN", "cn", "CN"):
            v = obj.get(k) if isinstance(obj, dict) else None
            if isinstance(v, str) and v.strip():
                names.add(_clean_ws(v))
        if ko:
            names.add(ko)
        if not names and raw:
            names.add(_clean_ws(raw))
        # 토큰 키 생성
        for name in names:
            key = norm_key(name)
            m[key] = int(r["id"])
    eprint(f"[INFO] restaurant_categories loaded: {len(m)} tokens")
    return m


def match_categories_for_item(item: Dict[str, Any], cat_map: Dict[str, int]) -> List[int]:
    fields = []
    for k in [
        "category",
        "categories",
        "category_names",
        "cat",
        "cats",
        "cat_big",
        "cat_mid",
        "cat_small",
        "mainCategory",
        "subCategory",
    ]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            fields.extend(re.split(r"[\/\|,;>\]\)\(]+|\s{2,}", v))
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, str):
                    fields.append(it)
    out: List[int] = []
    seen: Set[int] = set()
    for token in fields:
        tk = norm_key(token)
        if not tk:
            continue
        # 완전일치 우선
        if tk in cat_map:
            cid = cat_map[tk]
            if cid not in seen:
                out.append(cid)
                seen.add(cid)
        else:
            # 부분일치(짧은 토큰 제외)
            if len(tk) >= 2:
                for key, cid in cat_map.items():
                    if key in tk or tk in key:
                        if cid not in seen:
                            out.append(cid)
                            seen.add(cid)
    return out


# ===================== detail/menus 파일 로더 =====================

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
    """
    data/detail/[rid].json에서 description_json / short_description_json 을 읽고
    (description_json, short_description_json) 튜플로 반환.
    파일이 없거나 파싱 실패하면 빈 다국어 딕셔너리 반환.
    """
    path = os.path.join(detail_dir, f"{rid}.json")
    if not os.path.exists(path):
        return ({"ko": "", "en": "", "cn": ""}, {"ko": "", "en": "", "cn": ""})
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return ({"ko": "", "en": "", "cn": ""}, {"ko": "", "en": "", "cn": ""})

    # 루트에 없을 경우 대비하여 몇 가지 흔한 경로도 확인
    desc = doc.get("description_json") or doc.get("description") or get_by_path(doc, "data.description_json")
    short = doc.get("short_description_json") or doc.get("short_description") or get_by_path(doc, "data.short_description_json")

    return (_normalize_ml_dict(desc), _normalize_ml_dict(short))


# --- 가격 파서: 문자열/숫자 모두 견고하게 정수 KRW로 변환 ---
_K_PAT = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[kK]\s*$")


def coerce_price_krw(v) -> Optional[int]:
    """
    다양한 포맷을 int KRW로 변환:
    - 12000, "12000", "12,000", "₩12,000", "12,000원", "  12000  "
    - "9.9k" → 9900 (반올림)
    실패 시 None
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            n = int(round(float(v)))
            return n if n >= 0 else None
        except Exception:
            return None
    s = str(v).strip()
    if not s:
        return None
    m = _K_PAT.match(s)
    if m:
        try:
            n = float(m.group(1))
            n = int(round(n * 1000))
            return n if n >= 0 else None
        except Exception:
            return None
    # 일반 숫자 추출: 콤마/통화기호/원 제거
    s_norm = s.replace(",", "")
    s_norm = re.sub(r"[^\d.]", "", s_norm)  # 숫자/점 이외 제거
    if not s_norm:
        return None
    try:
        # 소수점이 있으면 반올림
        n = int(round(float(s_norm)))
        return n if n >= 0 else None
    except Exception:
        return None


def read_menus(menu_dir: str, rid: str) -> List[Dict[str, Any]]:
    """
    data/menu/[rid].json 형식 유연 처리:
    - 루트: list | {menus|list|items: []}
    - 각 항목: { name_json: {ko,en,cn}, price_krw, (optional) discount_price, image_url }
    """
    path = os.path.join(menu_dir, f"{rid}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return []
    items_src = []
    if isinstance(doc, list):
        items_src = doc
    elif isinstance(doc, dict):
        for k in ("menus", "list", "items"):
            v = doc.get(k)
            if isinstance(v, list):
                items_src = v
                break
    if not items_src:
        return []

    out: List[Dict[str, Any]] = []
    for it in items_src:
        if not isinstance(it, dict):
            continue
        name_json = it.get("name_json") or it.get("name") or it.get("title")
        name_norm = _normalize_ml_dict(name_json)

        price_val = coerce_price_krw(it.get("price_krw") or it.get("price") or it.get("priceKRW"))
        disc_val = coerce_price_krw(it.get("discount_price") or it.get("discountPrice"))
        img = it.get("image_url") or it.get("image") or None

        out.append(
            {
                "name_json": name_norm,
                "price_krw": price_val,  # None 일 수도 있음 → insert에서 보정
                "discount_price": disc_val,  # None이면 insert에서 price로 보정
                "image_url": img.strip() if isinstance(img, str) and img.strip() else None,
            }
        )
    return out


# ===================== 비즈니스 시간 고정값 =====================

BUSINESS_HOURS = {
    "regular": {
        "monday": "10:00-21:00",
        "tuesday": "10:00-21:00",
        "wednesday": "10:00-21:00",
        "thursday": "10:00-21:00",
        "friday": "10:00-21:00",
        "saturday": "10:00-21:00",
        "sunday": "10:00-21:00",
    }
}
BREAK_TIME = {
    "regular": {
        "monday": "15:00-17:00",
        "tuesday": "15:00-17:00",
        "wednesday": "15:00-17:00",
        "thursday": "15:00-17:00",
        "friday": "15:00-17:00",
        "saturday": "15:00-17:00",
        "sunday": "15:00-17:00",
    }
}
LAST_ORDER = {
    "regular": {
        "monday": "20:00",
        "tuesday": "20:00",
        "wednesday": "20:00",
        "thursday": "20:00",
        "friday": "20:00",
        "saturday": "20:00",
        "sunday": "20:00",
    }
}


# ===================== INSERTS =====================

def insert_restaurant(
    cur,
    item: Dict[str, Any],
    owner_id: int,
    region_map: Dict[Tuple[str, str, str], int],
    translator: TranslatorCtl,
    detail_dir: str,
) -> Optional[int]:
    nm = _clean_ws(item.get("nm") or item.get("name") or "")
    if not nm:
        eprint(f"[WARN] nm 없음 → skip (rid={item.get('v_rid') or item.get('rid')})")
        return None
    ml_name = translator.to_multi(nm)

    addr = _clean_ws(item.get("addr") or item.get("address") or "")
    l1, l2, l3, rest = split_addr_3(addr)
    region_id = None
    if l1 and l2 and l3:
        region_id = region_map.get((l1, l2, l3))
        if not region_id:
            eprint(f"[WARN] region 매핑 실패: addr='{addr}' → (l1={l1}, l2={l2}, l3={l3})")
    else:
        eprint(f"[WARN] 주소 파싱 실패: '{addr}'")

    # address/addr_detail 다국어
    addr_prefix = " ".join([p for p in [l1, l2, l3] if p])
    addr_detail = rest
    ml_addr = translator.to_multi(addr_prefix) if addr_prefix else {"ko": "", "en": "", "cn": ""}
    ml_addr_detail = translator.to_multi(addr_detail) if addr_detail else {"ko": "", "en": "", "cn": ""}

    # desc / short_desc from detail/[rid].json
    rid_str = str(item.get("v_rid") or item.get("rid") or "").strip()
    desc_json, short_json = read_detail_json(detail_dir, rid_str) if rid_str else (
        {"ko": "", "en": "", "cn": ""},
        {"ko": "", "en": "", "cn": ""},
    )

    phone = _clean_ws(item.get("tel") or item.get("phone") or "")
    lat = parse_float(item.get("lat") or item.get("y"))
    lng = parse_float(item.get("lng") or item.get("x"))
    image = _clean_ws(item.get("image") or item.get("img") or "")

    cur.execute(
        """
        INSERT INTO restaurants
        (name, description, short_description, phone, address, address_detail,
         latitude, longitude, business_hours, last_order_time, break_time,
         waiting_enabled, reservation_enabled, status, main_image_url,
         owner_id, region_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, 1, 1, 'PENDING', %s, %s, %s)
        """,
        (
            json.dumps(ml_name, ensure_ascii=False),
            json.dumps(desc_json, ensure_ascii=False),
            json.dumps(short_json, ensure_ascii=False),
            phone or None,
            json.dumps(ml_addr, ensure_ascii=False),
            json.dumps(ml_addr_detail, ensure_ascii=False),
            lat,
            lng,
            json.dumps(BUSINESS_HOURS, ensure_ascii=False),
            json.dumps(LAST_ORDER, ensure_ascii=False),
            json.dumps(BREAK_TIME, ensure_ascii=False),
            image or None,
            int(owner_id),
            int(region_id or 0),
        ),
    )
    return int(cur.lastrowid)


def insert_category_mappings(cur, restaurant_id: int, category_ids: List[int]) -> int:
    if not category_ids:
        return 0
    inserted = 0
    for idx, cid in enumerate(category_ids):
        cur.execute(
            """
            INSERT INTO restaurant_category_mappings (restaurant_id, category_id, is_primary)
            VALUES (%s, %s, %s)
            """,
            (restaurant_id, cid, 1 if idx == 0 else 0),
        )
        inserted += 1
    return inserted


def insert_menus(cur, restaurant_id: int, menus: List[Dict[str, Any]]) -> int:
    """
    - price NOT NULL 스키마 준수:
      * price_krw가 None이면 0으로 보정하고 WARN 로그
      * discount_price가 None이거나 price보다 크면 price로 세팅
    - description = name_json 동일 저장
    """
    inserted = 0
    for m in menus:
        name_json = m.get("name_json") or {"ko": "", "en": "", "cn": ""}
        price = m.get("price_krw")
        if price is None:
            eprint(
                f"[WARN] 메뉴 가격 없음 → 0원으로 보정 (restaurant_id={restaurant_id}, name_ko='{name_json.get('ko','')}')"
            )
            price = 0
        discount = m.get("discount_price")
        if discount is None or (isinstance(discount, int) and discount > price):
            discount = price
        image_url = m.get("image_url") or None

        cur.execute(
            """
            INSERT INTO restaurant_menus
              (restaurant_id, name, description, price, discount_price, image_url,
               is_signature, is_popular, is_new, status, display_order)
            VALUES (%s, %s, %s, %s, %s, %s, 0, 0, 0, 'ACTIVE', 0)
            """,
            (
                restaurant_id,
                json.dumps(name_json, ensure_ascii=False),
                json.dumps(name_json, ensure_ascii=False),
                int(price),
                int(discount),
                image_url,
            ),
        )
        inserted += 1
    return inserted


# ===================== 메인 =====================

def main():
    ap = argparse.ArgumentParser(description="restaurants / restaurant_menus / restaurant_category_mappings 적재기 (detail 연동)")
    ap.add_argument("--files", nargs="+", required=True, help="입력 리스트 JSON 파일/패턴")
    ap.add_argument("--config", required=True, help="DB config.json")
    ap.add_argument("--profile", default="local")
    ap.add_argument("--limit", type=int, default=100, help="처리 개수 제한 (<=0 이면 무제한)")
    ap.add_argument("--owner-id", type=int, required=True)
    ap.add_argument("--region-table", default="region", help="region 테이블명 (region | regions)")

    # 부가 경로
    ap.add_argument("--menu-dir", default="data/menu")
    ap.add_argument("--detail-dir", default="data/detail")

    # 번역 옵션
    ap.add_argument("--translate", choices=["none", "auto"], default="auto")
    ap.add_argument(
        "--translate-provider", choices=["auto_chain", "googletrans", "deep"], default="auto_chain"
    )
    ap.add_argument("--translate-timeout", type=float, default=2.0)
    ap.add_argument("--translate-max", type=int, default=200)
    ap.add_argument("--zh-variant", choices=["cn", "tw"], default="cn")
    ap.add_argument("--cache-file", default=".cache/restaurant-trans.json")

    args = ap.parse_args()

    cfg = load_config(args.config)
    mysql_cfg = resolve_mysql_cfg(cfg, args.profile)
    files = gather_files(args.files)
    if not files:
        eprint("[ERR] --files 패턴에 해당하는 파일이 없습니다.")
        sys.exit(2)

    # 번역 캐시 준비
    os.makedirs(os.path.dirname(args.cache_file) or ".", exist_ok=True)
    disk_cache = DiskCache(args.cache_file)
    translator = TranslatorCtl(
        mode=args.translate,
        provider=args.translate_provider,
        timeout=args.translate_timeout,
        max_calls=args.translate_max,
        zh_variant=args.zh_variant,
        cache=disk_cache,
    )

    conn = connect_mysql(mysql_cfg)
    total_inserted = 0
    total_catmap = 0
    total_menus = 0

    limit = args.limit if args.limit is not None else 100
    unlimited = limit <= 0

    try:
        with conn.cursor() as cur:
            # 지역/카테고리 캐시 로딩
            region_map = load_region_map(conn, args.region_table)
            cat_map = load_categories(conn)

            # 입력 파싱 & 적재
            picked = 0
            for fp in files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                except Exception as ex:
                    eprint(f"[WARN] 파일 로드 실패: {fp}: {ex}")
                    continue

                items = find_items_auto(doc)
                eprint(f"[INFO] {os.path.basename(fp)} 항목 수: {len(items)}")
                for it in items:
                    if (not unlimited) and picked >= limit:
                        break
                    rid = str(it.get("v_rid") or it.get("rid") or "").strip()

                    rest_id = insert_restaurant(
                        cur, it, args.owner_id, region_map, translator, args.detail_dir
                    )
                    if not rest_id:
                        continue

                    # 카테고리 매핑
                    cids = match_categories_for_item(it, cat_map)
                    total_catmap += insert_category_mappings(cur, rest_id, cids)

                    # 메뉴 적재
                    menus = read_menus(args.menu_dir, rid) if rid else []
                    total_menus += insert_menus(cur, rest_id, menus)

                    picked += 1
                    total_inserted += 1
                    if (picked % 10) == 0:
                        conn.commit()
                        disk_cache.save()
                        eprint(
                            f"[INFO] 진행: inserted={total_inserted}, cat_mapped(+{len(cids)}) total={total_catmap}, menus(+{len(menus)}) total={total_menus}"
                        )

            conn.commit()
            disk_cache.save()
            eprint(
                f"[DONE] restaurants inserted={total_inserted}, category mappings inserted={total_catmap}, menus inserted={total_menus} (limit={'unlimited' if unlimited else limit})"
            )

    except KeyboardInterrupt:
        eprint("\n[INTERRUPTED] Ctrl+C → 진행분 커밋 시도")
        try:
            conn.commit()
            disk_cache.save()
            eprint("[DONE] 부분 커밋 완료")
        except Exception:
            conn.rollback()
            eprint("[ROLLBACK] 커밋 실패 → 롤백")
        sys.exit(1)
    except Exception as e:
        conn.rollback()
        eprint(f"[ERR] 롤백: {e}")
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
