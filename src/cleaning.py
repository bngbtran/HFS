"""Các hàm làm sạch dữ liệu cửa hàng F&B (dùng trong notebook 01)."""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from unidecode import unidecode

TEXT_COLS = ["storeName", "ward", "city", "district", "address",
             "category", "productType", "storeId"]

DISTRICT_CANONICAL = "Quận Hải Châu"

# 13 phường của Q. Hải Châu (địa giới trước sáp nhập 2025), dạng ASCII để so khớp.
HAI_CHAU_WARDS = {
    "binh hien": "Bình Hiên", "binh thuan": "Bình Thuận",
    "hai chau 1": "Hải Châu 1", "hai chau 2": "Hải Châu 2",
    "hoa cuong bac": "Hòa Cường Bắc", "hoa cuong nam": "Hòa Cường Nam",
    "hoa thuan dong": "Hòa Thuận Đông", "hoa thuan tay": "Hòa Thuận Tây",
    "nam duong": "Nam Dương", "phuoc ninh": "Phước Ninh",
    "thach thang": "Thạch Thang", "thanh binh": "Thanh Bình",
    "thuan phuoc": "Thuận Phước",
}

# Quận khác (Đà Nẵng + Hải An - Hải Phòng). Chỉ so khớp khi có tiền tố "quận/q."
# vì "Hải Phòng", "Nguyễn Sơn Trà" là TÊN ĐƯỜNG ở Hải Châu.
OTHER_DISTRICTS = ["son tra", "lien chieu", "thanh khe", "cam le",
                   "ngu hanh son", "hoa vang", "hai an"]

# Nhãn huy hiệu của Grab, không phải loại hình sản phẩm.
BADGE_LABELS = {"Tạp Dề Bạc", "Tạp Dề Vàng", "Deluxe by Grab", "Official Brand Store"}

# Gộp nhãn đồng nghĩa / nhãn con về nhãn gốc, dịch nhãn tiếng Việt.
LABEL_MAP = {
    "Noodles": "Noodles & Congee",
    "Congee": "Noodles & Congee",
    "Chè": "Dessert",
    "Cake": "Dessert",
    "Healthy": "Healthy Food",
    "Healthy - Salad": "Healthy Food",
    "Vegan - Vegetarian": "Healthy Food",
    "Fish-seafood": "Seafood",
    "Sushi": "Japanese",
    "Bánh Truyền Thống": "Snack",
    "Bánh Tráng": "Snack",
    "Món Nhậu": "Hotpot & Grill",
    "Vietnamese cuisine": "Others",
    "Convenience Store": "Grocery",
    "Supermarket": "Grocery",
    "Packaged food-dried food": "Grocery",
    "Bakery & Baking Supplies": "Grocery",
    "Vegetable": "Grocery",
    "Fresh Fruits": "Grocery",
    "Organic": "Grocery",
    "Beverage-Dairy": "Grocery",
    "Alcohol-Beer": "Grocery",
    "Flower": "Other Store",
    "Health & Beauty": "Other Store",
}


# ---------------------------------------------------------------- tải & chuỗi
def load_raw(path: str | Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f)["DaNangData"])


def normalize_text(s):
    """Unicode NFC, strip, gộp khoảng trắng, bỏ khoảng trắng trước dấu phẩy."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+,", ",", s)
    return s if s else np.nan


def to_ascii(s) -> str:
    """Bỏ dấu + lowercase để so khớp (đ -> d)."""
    if not isinstance(s, str):
        return ""
    return unidecode(s).lower()


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in TEXT_COLS:
        df[c] = df[c].map(normalize_text)
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng tóm tắt: kiểu dữ liệu, missing, rỗng, số giá trị unique."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_empty": df.apply(lambda s: (s.astype(str).str.strip() == "").sum()),
        "n_unique": df.nunique(),
        "example": df.iloc[0],
    })


# ---------------------------------------------------------------- quận
def is_hai_chau_text(s) -> bool:
    a = to_ascii(s)
    return bool(re.search(r"hai\s*chau", a))


def mentions_other_district(addr) -> bool:
    a = to_ascii(addr)
    pat = r"(?:quan|q\.)\s*(?:" + "|".join(OTHER_DISTRICTS) + r")\b"
    return bool(re.search(pat, a))


def clean_district(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa district về 1 giá trị; đánh cờ bản ghi ngoài Hải Châu.

    is_district_inconsistent = district không phải Hải Châu
                               HOẶC address ghi rõ "Quận <quận khác>".
    """
    df = df.copy()
    df["district_raw"] = df["district"]
    ok_text = df["district"].map(is_hai_chau_text)
    other_addr = df["address"].map(mentions_other_district)
    df["is_district_inconsistent"] = (~ok_text) | other_addr
    df["district"] = np.where(ok_text, DISTRICT_CANONICAL, df["district"])
    return df


# ---------------------------------------------------------------- phường
_WARD_RE = re.compile(r"^Da Nang - (.+?) Ward of (.+?) District$")
_ROMAN = {" ii": " 2", " i": " 1"}


def _ward_key(name: str) -> str:
    k = to_ascii(name).strip()
    for r, n in _ROMAN.items():
        if k.endswith(r):
            k = k[: -len(r)] + n
    return k


def parse_ward_field(w):
    """'Da Nang - Hai Chau II Ward of Hai Chau District' -> ('hai chau 2', 'Hai Chau')."""
    if not isinstance(w, str):
        return np.nan, np.nan
    m = _WARD_RE.match(w)
    if not m:
        return np.nan, np.nan
    return _ward_key(m.group(1)), m.group(2)


_ADDR_WARD_RE = re.compile(
    r"(?:phuong|p\.)\s*([a-z0-9 ]+?)\s*(?:,|$|quan\b|q\.|thanh pho|tp\b|da nang)")


def parse_ward_address(addr):
    """Lấy tên phường trong address ('Phường X' / 'P. X'), so khớp mờ với 13 phường.

    Trả về khóa ASCII của phường Hải Châu, hoặc chuỗi thô nếu là phường khác,
    hoặc NaN nếu không có / không xác định được (vd 'Hải Châu', 'Hòa Cường').
    """
    m = _ADDR_WARD_RE.search(to_ascii(addr))
    if not m:
        return np.nan
    raw = _ward_key(m.group(1))
    if raw in HAI_CHAU_WARDS:
        return raw
    if raw in {"hai chau", "hoa cuong", "hoa thuan", "hoa thuan nam", "1"}:
        return np.nan  # mơ hồ
    close = difflib.get_close_matches(raw, HAI_CHAU_WARDS.keys(), n=1, cutoff=0.85)
    return close[0] if close else raw


def clean_ward(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các cột phường.

    ward_name      : tên phường tách từ trường `ward` (có vẻ là reverse-geocode)
    ward_district  : quận tương ứng trong trường `ward`
    ward_addr      : phường đọc từ address (do chủ quán nhập)
    ward_final     : ưu tiên ward_addr nếu là phường Hải Châu hợp lệ,
                     ngược lại dùng ward_name nếu là phường Hải Châu,
                     còn lại 'Ngoài Hải Châu' / 'Không rõ'.
    """
    df = df.copy()
    parsed = df["ward"].map(parse_ward_field)
    df["ward_name"] = parsed.str[0]
    df["ward_district"] = parsed.str[1]
    df["ward_addr"] = df["address"].map(parse_ward_address)

    def _final(r):
        if r.ward_addr in HAI_CHAU_WARDS:
            return HAI_CHAU_WARDS[r.ward_addr]
        if r.ward_name in HAI_CHAU_WARDS and r.ward_district == "Hai Chau":
            return HAI_CHAU_WARDS[r.ward_name]
        if isinstance(r.ward_district, str):
            return "Ngoài Hải Châu"
        return "Không rõ"

    df["ward_final"] = df.apply(_final, axis=1)
    both = df["ward_addr"].notna() & df["ward_name"].notna()
    df["is_ward_conflict"] = both & (df["ward_addr"] != df["ward_name"])
    return df


# ---------------------------------------------------------------- productType
def split_labels(s) -> list[str]:
    if not isinstance(s, str):
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


def clean_product_types(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """Tách productType thành list, tách huy hiệu Grab, gộp nhãn đồng nghĩa, khử trùng.

    Nhãn còn lại có tần suất < min_count được gộp vào 'Others' (FOOD) / 'Other Store' (SHOP).
    """
    df = df.copy()
    df["pt_raw"] = df["productType"].map(split_labels)
    df["grab_badges"] = df["pt_raw"].map(lambda L: [x for x in L if x in BADGE_LABELS])
    df["has_badge"] = df["grab_badges"].map(bool)

    def _map(L):
        out = []
        for x in L:
            if x in BADGE_LABELS:
                continue
            y = LABEL_MAP.get(x, x)
            if y not in out:
                out.append(y)
        return out

    mapped = df["pt_raw"].map(_map)
    counts = pd.Series([x for L in mapped for x in L]).value_counts()
    rare = set(counts[counts < min_count].index)

    def _fold(L, cat):
        other = "Other Store" if cat == "SHOP" else "Others"
        out = []
        for x in L:
            y = other if x in rare else x
            if y not in out:
                out.append(y)
        return out or [other]

    df["product_types"] = [_fold(L, c) for L, c in zip(mapped, df["category"])]
    df["n_types"] = df["product_types"].map(len)
    return df


# ---------------------------------------------------------------- rating
def add_rating(df: pd.DataFrame) -> pd.DataFrame:
    """GIẢ ĐỊNH: flag = rating × 10, flag = 0 nghĩa là chưa có đánh giá."""
    df = df.copy()
    df["rating"] = np.where(df["flag"] > 0, df["flag"] / 10.0, np.nan)
    df["has_rating"] = df["flag"] > 0
    return df


# ---------------------------------------------------------------- khử trùng
def _addr_key(addr) -> str:
    first = str(addr).split(",")[0]
    return re.sub(r"[^a-z0-9/]", "", to_ascii(first))


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gộp các listing trùng của cùng 1 cửa hàng.

    Khóa trùng: tên (ASCII) + tọa độ (5 chữ số thập phân, ~1m) + phần đầu địa chỉ
    (số nhà + đường). Sạp khác lô trong cùng chợ có địa chỉ khác nên không bị gộp.
    Khi gộp: hợp nhất product_types, rating = trung bình các rating khác 0.
    Trả về (df đã gộp, bảng các nhóm bị gộp).
    """
    df = df.copy()
    df["_key"] = (df["storeName"].map(to_ascii) + "|" + df["latitude"].round(5).astype(str)
                  + "|" + df["longitude"].round(5).astype(str) + "|" + df["address"].map(_addr_key))
    grp = df.groupby("_key", sort=False)
    df["n_listings"] = grp["storeId"].transform("size")
    dups = df[df["n_listings"] > 1].sort_values("_key")

    def _merge(g: pd.DataFrame) -> pd.Series:
        r = g.iloc[0].copy()
        if len(g) == 1:
            return r
        types = []
        for L in g["product_types"]:
            for x in L:
                if x not in types:
                    types.append(x)
        r["product_types"] = types
        r["n_types"] = len(types)
        r["grab_badges"] = sorted({b for L in g["grab_badges"] for b in L})
        r["has_badge"] = bool(r["grab_badges"])
        rated = g.loc[g["flag"] > 0, "flag"]
        r["flag"] = int(round(rated.mean())) if len(rated) else 0
        r["rating"] = rated.mean() / 10.0 if len(rated) else np.nan
        r["has_rating"] = bool(len(rated))
        r["merged_ids"] = ";".join(g["storeId"])
        return r

    multi = df[df["n_listings"] > 1]
    merged = [_merge(g) for _, g in multi.groupby("_key", sort=False)]
    out = pd.concat([df[df["n_listings"] == 1], pd.DataFrame(merged)])
    out = out.sort_index().drop(columns="_key")
    out["merged_ids"] = out.get("merged_ids", pd.Series(dtype=str)).fillna("")
    return out, dups.drop(columns="_key")


# ---------------------------------------------------------------- chain
_BRAND_SPLIT = re.compile(r"\s+[-–]\s*|\s*[-–]\s+|\(")


def brand_of(name) -> str:
    """Chuẩn hóa tên -> brand: bỏ dấu, lowercase, cắt phần sau ' - ' / '(' .

    Chỉ cắt khi phần còn lại có ≥2 từ, tránh brand quá chung chung (vd 'banh mi').
    Gạch nối không có khoảng trắng (vd 'Tea-Rex') được giữ nguyên.
    """
    a = to_ascii(name)
    parts = _BRAND_SPLIT.split(a, maxsplit=1)
    head = parts[0]
    if len(head.split()) < 2:
        head = a
    head = re.sub(r"[^a-z0-9& ]", " ", head)
    return re.sub(r"\s+", " ", head).strip()


# Tên chỉ là tên món ăn chung chung -> nhiều quán độc lập trùng tên, KHÔNG phải chuỗi.
# Lập thủ công từ danh sách brand có ≥2 địa điểm (xem notebook 01, bước 9).
GENERIC_BRANDS = {
    "bun bo", "bun bo hue", "bun long bo", "bun mam", "bun hai san", "bun cha ca",
    "tra sua", "an vat", "an vat sai gon", "banh mi", "banh trang", "banh trang tron",
    "banh trang kep", "pho", "bo ne", "pha lau", "lau bo", "nuoc ep", "pizza",
    "pizza pasta", "takoyaki", "tiramisu", "trung ga nuong", "com hen", "quan com tam",
    "hu tieu nam vang", "dac san lao", "dac san 3 mien", "diem tam sang", "ca phe muoi",
    "mart", "banh canh ca loc", "tiem tra chanh", "bubble tea & fast food",
    "bubble tea & fast foods",
}
# Sạp trong chợ đặt tên theo chợ -> không phải chuỗi.
MARKET_BRANDS = {"cho con", "cho han"}


def add_chain(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """is_chain = brand xuất hiện ≥ min_count lần và không phải tên chung chung / tên chợ."""
    df = df.copy()
    df["brand"] = df["storeName"].map(brand_of)
    df["brand_count"] = df.groupby("brand")["brand"].transform("size")
    df["is_generic_name"] = df["brand"].isin(GENERIC_BRANDS)
    df["is_market_stall"] = df["brand"].isin(MARKET_BRANDS)
    df["is_chain"] = ((df["brand_count"] >= min_count)
                      & ~df["is_generic_name"] & ~df["is_market_stall"])
    return df


# ---------------------------------------------------------------- log
class CleaningLog:
    """Ghi lại từng bước: số bản ghi trước/sau, số bản ghi bị ảnh hưởng."""

    def __init__(self):
        self.rows = []

    def add(self, step: str, n_before: int, n_after: int, n_affected: int, note: str = ""):
        self.rows.append(dict(step=step, n_before=n_before, n_after=n_after,
                              n_affected=n_affected, note=note))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
