"""Sinh notebooks/01_preprocessing.ipynb (chạy: python scripts/build_nb01.py)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 01 – Tiền xử lý dữ liệu cửa hàng F&B quận Hải Châu

**Mục tiêu:** từ file thô `data/raw/DaNangData.json` (3.256 bản ghi crawl từ GrabFood/GrabMart) tạo ra bảng sạch
`data/processed/stores_clean.csv` dùng cho các notebook sau.

Các bước:
1. Load & khảo sát (missing, kiểu dữ liệu, unique)
2. Chuẩn hóa chuỗi (Unicode NFC, strip, gộp khoảng trắng)
3. Chuẩn hóa `district`, đánh cờ bản ghi ngoài Hải Châu
4. Tách tên phường từ `ward`, đối chiếu với `address`
5. Làm sạch tọa độ (bounding box Hải Châu + 1 km)
6. Tách `productType` đa nhãn → one-hot `pt_*`, `n_types`
7. `flag` → `rating`, `has_rating` (**giả định**)
8. Khử trùng listing
9. Phát hiện chuỗi cửa hàng (`brand`, `is_chain`)
10. Xuất dữ liệu + bảng log

Toàn bộ logic nằm trong `src/cleaning.py`, `src/geo.py`, `src/features.py`, `src/viz.py`; notebook chỉ gọi hàm và trình bày.
""")

code(r"""
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import folium

from src import cleaning as cl, geo, features as ft, viz

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
viz.set_style()

RAW_PATH = ROOT / "data" / "raw" / "DaNangData.json"
OUT_PATH = ROOT / "data" / "processed" / "stores_clean.csv"
log = cl.CleaningLog()
""")

# ------------------------------------------------------------------ 1
md(r"""
## 1. Load dữ liệu và khảo sát ban đầu
""")
code(r"""
raw = cl.load_raw(RAW_PATH)
print("Kích thước:", raw.shape)
cl.profile(raw)
""")
code(r"""
for c in ["category", "city", "flag"]:
    print(f"--- {c} ---")
    print(raw[c].value_counts(dropna=False).to_string(), "\n")
print("Số tổ hợp productType khác nhau:", raw["productType"].nunique())
print("Số storeName xuất hiện ≥2 lần:", (raw["storeName"].value_counts() >= 2).sum())
print("Số storeId trùng:", raw["storeId"].duplicated().sum())
print("Số dòng trùng hoàn toàn:", raw.duplicated().sum())
""")
md(r"""
**Kiểm chứng các quan sát ban đầu** (xem output phía trên):
- Không có dòng trùng hoàn toàn và không có `storeId` trùng. Chỉ `ward` có 3 giá trị null; `storeName` không có giá trị rỗng.
- `category` mất cân bằng: FOOD chiếm áp đảo so với SHOP.
- `flag` có rất nhiều giá trị 0; các giá trị khác nằm trong đoạn 30–45.
""")

# ------------------------------------------------------------------ 2
md(r"""
## 2. Chuẩn hóa chuỗi

Dữ liệu crawl chứa ký tự Unicode **tổ hợp** (NFD, vd "a" + dấu huyền rời) lẫn **dựng sẵn** (NFC), nhìn giống nhau nhưng là chuỗi khác nhau,
khoảng trắng thừa ở đầu/cuối hoặc giữa các từ, và khoảng trắng trước dấu phẩy. Hàm `cl.normalize_text` lần lượt:
NFC → gộp khoảng trắng → strip → bỏ khoảng trắng trước dấu phẩy. Chuỗi rỗng được chuyển thành NaN.
""")
code(r"""
df = cl.normalize_strings(raw)
changed = {c: int((raw[c].fillna("") != df[c].fillna("")).sum()) for c in cl.TEXT_COLS}
print("Số giá trị bị thay đổi theo cột:", changed)
print("Số giá trị district unique: trước =", raw["district"].nunique(), "| sau =", df["district"].nunique())
log.add("2. Chuẩn hóa chuỗi (NFC, strip, khoảng trắng)", len(raw), len(df),
        int((raw[cl.TEXT_COLS].fillna("") != df[cl.TEXT_COLS].fillna("")).any(axis=1).sum()),
        "; ".join(f"{k}={v}" for k, v in changed.items() if v))
""")

# ------------------------------------------------------------------ 3
md(r"""
## 3. Chuẩn hóa `district`

Quy tắc:
- Nếu chuỗi (sau khi bỏ dấu, lowercase) chứa `hai chau` thì gán `"Quận Hải Châu"`. Cách này bao được "Q.Hải Châu", "Quận Hải châu", "Quận Hải Châu 1", "Phường Hải Châu 1", ...
- `is_district_inconsistent = True` nếu `district` **không** phải Hải Châu, **hoặc** `address` ghi rõ "Quận/Q. <quận khác>".
  - Chỉ so khớp khi có tiền tố "Quận/Q.", vì **"Hải Phòng" và "Nguyễn Sơn Trà" là tên đường ở Hải Châu**. Nếu so khớp chuỗi trần sẽ đánh cờ nhầm hàng chục bản ghi.
""")
code(r"""
print("Các biến thể district (repr để thấy khoảng trắng / ký tự lạ) – dữ liệu thô:")
print(raw["district"].map(repr).value_counts().to_string())
""")
code(r"""
df = cl.clean_district(df)
print(df["district"].value_counts().to_string())
inc = df[df["is_district_inconsistent"]]
print(f"\nSố bản ghi bị đánh cờ is_district_inconsistent: {len(inc)}")
inc[["storeName", "district_raw", "address", "latitude", "longitude"]]
""")
code(r"""
log.add("3. Chuẩn hóa district", len(df), len(df),
        int((df["district_raw"] != df["district"]).sum() + df["is_district_inconsistent"].sum()),
        f"{raw['district'].nunique()} biến thể -> 1 giá trị chuẩn; {int(df['is_district_inconsistent'].sum())} bản ghi bị đánh cờ ngoài Hải Châu")
""")
md(r"""
**Nhận xét:** các bản ghi ghi "Quận Hải An" và địa chỉ "Phường Đằng Hải", "Phường Cát Bi" thực chất thuộc **quận Hải An, TP. Hải Phòng**. Đây không phải lỗi gõ "Hải Châu" mà là cửa hàng của tỉnh khác bị lẫn vào khi crawl.
"An Hải Bắc/Đông" là phường của Q. Sơn Trà, bên kia sông Hàn. Các bản ghi này sẽ bị loại ở bước 5.
""")

# ------------------------------------------------------------------ 4
md(r"""
## 4. Tách tên phường

Có **hai nguồn** thông tin phường:
- Trường `ward`, dạng `"Da Nang - Hai Chau II Ward of Hai Chau District"`. Ta tách được `ward_name` và `ward_district`.
- Phần "Phường X" / "P. X" trong `address` do chủ quán tự nhập, ta tách thành `ward_addr`. Phần này được so khớp mờ (difflib, cutoff 0.85) với danh sách 13 phường để sửa lỗi gõ như "Phước Ninp", "Hoàng Cường Bắc", "Thạch Trang".
""")
code(r"""
df = cl.clean_ward(df)
print("ward_district (theo trường ward):")
print(df["ward_district"].value_counts(dropna=False).to_string())
""")
code(r"""
cmp = pd.DataFrame({
    "theo_truong_ward": df.loc[df["ward_district"] == "Hai Chau", "ward_name"].value_counts(),
    "theo_address": df["ward_addr"].value_counts(),
}).reindex(sorted(cl.HAI_CHAU_WARDS)).fillna(0).astype(int)
cmp.index = [cl.HAI_CHAU_WARDS[k] for k in cmp.index]
print("Số cửa hàng theo phường – so sánh hai nguồn:")
cmp
""")
code(r"""
both = df["ward_addr"].notna() & df["ward_name"].notna()
agree = (df.loc[both, "ward_addr"] == df.loc[both, "ward_name"]).mean()
print(f"Bản ghi có cả 2 nguồn: {both.sum()} | tỉ lệ khớp: {agree:.1%} | mâu thuẫn: {df['is_ward_conflict'].sum()}")
print(f"Bản ghi address không có phường đọc được: {df['ward_addr'].isna().sum()}")
print("\nCác cặp mâu thuẫn phổ biến (address -> ward):")
print(df[df["is_ward_conflict"]].groupby(["ward_addr", "ward_name"]).size().sort_values(ascending=False).head(12).to_string())
""")
md(r"""
**Quyết định chọn phường cho phân tích (`ward_final`):**
- Trường `ward` hoàn toàn **không có** phường Phước Ninh và Bình Hiên, trong khi `address` có hàng trăm bản ghi thuộc hai phường này. Trường `ward` còn gán hàng trăm điểm cho các phường của Q. Thanh Khê.
  Như vậy `ward` nhiều khả năng được **reverse-geocode** từ tọa độ theo một bộ ranh giới khác hoặc cũ, nên kém tin cậy hơn.
- Vì vậy ta **ưu tiên `ward_addr`** khi nó là một phường Hải Châu hợp lệ. Nếu không, dùng `ward_name` khi nó thuộc Hải Châu. Còn lại gán `"Ngoài Hải Châu"` hoặc `"Không rõ"`.
- Giữ cờ `is_ward_conflict` để minh bạch.
""")
code(r"""
print(df["ward_final"].value_counts().to_string())
log.add("4. Tách phường (ward_name, ward_addr, ward_final)", len(df), len(df),
        int(df["is_ward_conflict"].sum()),
        f"tỉ lệ khớp 2 nguồn {agree:.1%}; ưu tiên phường trong address")
""")

# ------------------------------------------------------------------ 5
md(r"""
## 5. Làm sạch tọa độ

Bounding box Hải Châu (xấp xỉ): lat 16.00–16.09, lon 108.18–108.26. Ta **nới thêm 1 km mỗi phía**
(1 km ≈ 0.0090° vĩ độ ≈ 0.0093° kinh độ ở vĩ độ 16°) để không loại nhầm quán nằm sát ranh giới.
Lý do: các thuật toán mật độ ở notebook 03 cần "ngữ cảnh" quanh rìa quận.

Bản ghi bị loại ở bước này:
1. Tọa độ nằm ngoài bbox đã nới, ví dụ điểm ở TP.HCM, Hải Phòng/Quảng Ninh, Liên Chiểu.
2. Bản ghi `is_district_inconsistent`, tức địa chỉ ghi rõ quận/tỉnh khác. Đề tài chỉ xét Q. Hải Châu.
""")
code(r"""
bbox_buf = geo.buffered_bbox(geo.HAI_CHAU_BBOX, buffer_m=1000)
print("BBox gốc :", geo.HAI_CHAU_BBOX)
print("BBox +1km:", {k: round(v, 4) for k, v in bbox_buf.items()})

inside_core = geo.in_bbox(df, geo.HAI_CHAU_BBOX)
inside_buf = geo.in_bbox(df, bbox_buf)
df["is_coord_outlier"] = ~inside_buf
drop_mask = df["is_coord_outlier"] | df["is_district_inconsistent"]

print(f"\nNgoài bbox gốc: {(~inside_core).sum()} | ngoài bbox +1km: {(~inside_buf).sum()}")
print(f"Được giữ nhờ biên 1km: {(inside_buf & ~inside_core).sum()}")
print(f"District/address không nhất quán: {df['is_district_inconsistent'].sum()} "
      f"(trong đó đã ngoài bbox: {(df['is_district_inconsistent'] & ~inside_buf).sum()})")
print(f"=> Tổng số bản ghi bị loại: {drop_mask.sum()}")
df.loc[drop_mask, ["storeName", "address", "latitude", "longitude", "is_coord_outlier", "is_district_inconsistent"]]
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.scatter(df.loc[~drop_mask, "longitude"], df.loc[~drop_mask, "latitude"], s=6, c=viz.PALETTE[0], label="Giữ lại")
ax.scatter(df.loc[drop_mask, "longitude"], df.loc[drop_mask, "latitude"], s=30, c=viz.PALETTE[7], marker="x", label="Bị loại")
for _, r in df[df["is_coord_outlier"] & (~df["latitude"].between(15.5, 16.5))].iterrows():
    ax.annotate(r["storeName"], (r["longitude"], r["latitude"]), fontsize=8, xytext=(5, 3),
                textcoords="offset points", color=viz.TEXT_2)
ax.set_title("Trước khi lọc: toàn bộ 3.256 điểm")
ax.set_xlabel("Kinh độ (°E)"); ax.set_ylabel("Vĩ độ (°N)")
ax.legend(loc="center left")

ax = axes[1]
kept = df[~drop_mask]
ax.scatter(kept["longitude"], kept["latitude"], s=4, c=viz.PALETTE[0], alpha=0.6, label="Giữ lại")
near = df[drop_mask & df["latitude"].between(15.9, 16.2)]
ax.scatter(near["longitude"], near["latitude"], s=40, c=viz.PALETTE[7], marker="x", label="Bị loại (gần Đà Nẵng)")
for b, ls, lab in [(geo.HAI_CHAU_BBOX, "-", "BBox Hải Châu"), (bbox_buf, "--", "BBox + 1 km")]:
    ax.add_patch(patches.Rectangle((b["lon_min"], b["lat_min"]), b["lon_max"] - b["lon_min"],
                                   b["lat_max"] - b["lat_min"], fill=False, ls=ls, ec=viz.TEXT_2, lw=1.2, label=lab))
ax.set_xlim(bbox_buf["lon_min"] - 0.015, bbox_buf["lon_max"] + 0.01)
ax.set_ylim(bbox_buf["lat_min"] - 0.01, bbox_buf["lat_max"] + 0.01)
ax.set_aspect(1 / np.cos(np.radians(16.05)))
ax.set_title("Sau khi lọc: phóng to khu vực Hải Châu")
ax.set_xlabel("Kinh độ (°E)"); ax.set_ylabel("Vĩ độ (°N)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
fig.suptitle("Làm sạch tọa độ: loại điểm ngoài bounding box Hải Châu (+1 km) và địa chỉ quận khác", fontweight="bold")
fig.tight_layout()
viz.save_fig(fig, "01_coord_cleaning_before_after")
plt.show()
""")
code(r"""
m = viz.base_map(zoom=13)
viz.add_points(m, df[~drop_mask], viz.PALETTE[0], "Giữ lại", radius=2)
viz.add_points(m, df[drop_mask], viz.PALETTE[7], "Bị loại", popup_cols=("storeName", "address", "latitude", "longitude"), radius=6)
for b, dash, lab in [(geo.HAI_CHAU_BBOX, None, "BBox Hải Châu"), (bbox_buf, "6", "BBox + 1 km")]:
    folium.Rectangle([[b["lat_min"], b["lon_min"]], [b["lat_max"], b["lon_max"]]], color="#52514e",
                     weight=2, fill=False, dash_array=dash, tooltip=lab).add_to(m)
folium.LayerControl().add_to(m)
map_path = viz.save_map(m, "01_coord_cleaning")
print("Đã lưu:", map_path.relative_to(ROOT))
viz.show_map(map_path)
""")
code(r"""
n0 = len(df)
dropped = df[drop_mask].copy()
df = df[~drop_mask].copy()
log.add("5. Loại tọa độ ngoài bbox+1km & quận khác", n0, len(df), int(drop_mask.sum()),
        f"ngoài bbox: {int(dropped['is_coord_outlier'].sum())}; quận/tỉnh khác (trong bbox): "
        f"{int((dropped['is_district_inconsistent'] & ~dropped['is_coord_outlier']).sum())}")
print("Còn lại:", len(df))
print("\nward_final của các bản ghi còn lại:")
print(df["ward_final"].value_counts().to_string())
""")
md(r"""
**Lưu ý:** sau khi lọc vẫn còn một số điểm có `ward_final = "Ngoài Hải Châu"`. Những điểm này nằm trong bbox nới rộng, sát ranh giới Thanh Khê / Cẩm Lệ / Sơn Trà, và địa chỉ không ghi phường Hải Châu.
Ta **giữ** chúng vì hai lý do. Thứ nhất, bbox chỉ là xấp xỉ. Thứ hai, chúng cung cấp ngữ cảnh mật độ cho phân cụm không gian. Khi phân tích theo phường, nhóm này được tách riêng.
""")

# ------------------------------------------------------------------ 6
md(r"""
## 6. `productType` đa nhãn

Khảo sát thực tế cho thấy có **51 nhãn gốc** chứ không phải khoảng 20. Nguyên nhân:
- **Nhãn huy hiệu Grab** không phải loại hình: "Tạp Dề Bạc", "Tạp Dề Vàng", "Deluxe by Grab", "Official Brand Store". Các nhãn này được tách ra cột `grab_badges` / `has_badge`.
- **Nhãn đồng nghĩa hoặc nhãn con**: "Noodles" và "Congee" là con của "Noodles & Congee", nên chuỗi "Noodles, Noodles & Congee" thực chất là trùng nghĩa. "Chè" gộp vào "Dessert", "Sushi" gộp vào "Japanese", ... (xem `cl.LABEL_MAP`).
- **Nhãn hiếm** (< 10 cửa hàng sau khi gộp) được gom vào "Others" (FOOD) hoặc "Other Store" (SHOP). Nhãn quá hiếm không đủ support cho luật kết hợp và chỉ làm tăng chiều cho mô hình.
""")
code(r"""
df = cl.clean_product_types(df, min_count=10)
raw_labels = df["pt_raw"].explode().value_counts()
final_labels = df["product_types"].explode().value_counts()
print(f"Số nhãn gốc: {len(raw_labels)}  ->  số nhãn sau chuẩn hóa: {len(final_labels)}")
print(f"Số bản ghi có huy hiệu Grab: {df['has_badge'].sum()}")
print(f"Số tổ hợp productType: trước {df['productType'].nunique()} -> sau {df['product_types'].map(lambda L: tuple(sorted(L))).nunique()}")

mapping = (df[["pt_raw", "product_types"]].explode("pt_raw").dropna()
           .assign(mapped=lambda d: d["pt_raw"].map(lambda x: "(huy hiệu Grab)" if x in cl.BADGE_LABELS else cl.LABEL_MAP.get(x, x))))
label_map_tbl = (raw_labels.rename("n_raw").to_frame()
                 .join(mapping.drop_duplicates("pt_raw").set_index("pt_raw")["mapped"]))
label_map_tbl["final"] = label_map_tbl["mapped"].where(
    label_map_tbl["mapped"].isin(final_labels.index) | (label_map_tbl["mapped"] == "(huy hiệu Grab)"), "Others / Other Store (hiếm)")
label_map_tbl.to_csv(viz.TABLE_DIR / "01_product_type_label_map.csv", encoding="utf-8-sig")
label_map_tbl[label_map_tbl.index != label_map_tbl["final"]]
""")
code(r"""
onehot = ft.multilabel_onehot(df["product_types"])
print("Các cột one-hot:", list(onehot.columns))
print("\nPhân phối n_types:")
print(df["n_types"].value_counts().sort_index().to_string())

fig, ax = plt.subplots(figsize=(8, 6.5))
viz.barh_counts(final_labels, "Tần suất loại hình (productType) sau chuẩn hóa",
                "Số cửa hàng (một cửa hàng có thể có nhiều nhãn)", "Loại hình", ax=ax)
fig.tight_layout(); viz.save_fig(fig, "01_product_type_freq"); plt.show()
log.add("6. Tách productType đa nhãn", len(df), len(df), int((df["pt_raw"].map(len) != df["n_types"]).sum()),
        f"{len(raw_labels)} nhãn gốc -> {len(final_labels)} nhãn; {int(df['has_badge'].sum())} bản ghi có huy hiệu Grab")
""")

# ------------------------------------------------------------------ 7
md(r"""
## 7. `flag` → `rating`

**Giả thuyết:** `flag = rating × 10` và `flag = 0` nghĩa là *chưa có đánh giá*. Ta kiểm chứng bằng phân phối:
""")
code(r"""
vc = df["flag"].value_counts().sort_index()
print(vc.to_string())
nz = df.loc[df["flag"] > 0, "flag"]
print(f"\nflag=0: {(df['flag'] == 0).sum()} ({(df['flag'] == 0).mean():.1%}) | flag>0: min={nz.min()}, max={nz.max()}, "
      f"median={nz.median()}, mean={nz.mean():.2f}")
print("Mọi giá trị đều là số nguyên:", (df["flag"] % 1 == 0).all())
print("\nTỉ lệ flag=0 theo category:")
print(df.groupby("category")["flag"].apply(lambda s: (s == 0).mean()).map("{:.1%}".format).to_string())
""")
code(r"""
df = cl.add_rating(df)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
r = df["rating"].dropna()
bins = np.arange(r.min() - 0.05, r.max() + 0.1, 0.1)
ax.hist(r, bins=bins, color=viz.PALETTE[0], edgecolor="white", linewidth=1.5)
ax.axvline(r.mean(), color=viz.TEXT_2, ls="--", lw=1)
ax.annotate(f"TB = {r.mean():.2f}", (r.mean(), ax.get_ylim()[1] * 0.97), xytext=(-60, 0), textcoords="offset points",
            color=viz.TEXT_2, fontsize=9, va="center", arrowprops=dict(arrowstyle="->", color=viz.TEXT_2, lw=0.8))
ax.set_title("Phân phối rating (= flag / 10, chỉ cửa hàng có đánh giá)")
ax.set_xlabel("Rating"); ax.set_ylabel("Số cửa hàng")
ax = axes[1]
share = df.groupby("category")["has_rating"].value_counts(normalize=True).unstack().reindex(columns=[True, False])
share.columns = ["Có đánh giá", "Chưa có (flag=0)"]
share.plot(kind="bar", stacked=True, ax=ax, color=[viz.PALETTE[0], "#c9c8c2"], width=0.6, edgecolor="white", linewidth=2)
ax.set_title("Tỉ lệ có đánh giá theo category")
ax.set_xlabel("Category"); ax.set_ylabel("Tỉ lệ"); ax.tick_params(axis="x", rotation=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
fig.tight_layout(); viz.save_fig(fig, "01_rating_distribution"); plt.show()
log.add("7. flag -> rating, has_rating (giả định)", len(df), len(df), int((~df["has_rating"]).sum()),
        "rating = flag/10; flag=0 -> NaN (chưa có đánh giá)")
""")
md(r"""
**Đánh giá giả thuyết:**
- Ủng hộ: các giá trị khác 0 là số nguyên trong đoạn 30–45, tức rating 3.0–4.5 với bước 0.1. Đây đúng là định dạng sao của Grab (1 chữ số thập phân). Phân phối lệch về phía cao, mode khoảng 4.0, đúng kiểu rating trên nền tảng giao đồ ăn.
  `flag = 0` chiếm gần một nửa và tập trung ở SHOP, là nhóm ít được đánh giá. Điều này hợp lý với nghĩa "chưa có đánh giá".
- Điểm bất thường: **không có giá trị nào trên 45 (4.5 sao)** và không có giá trị dưới 30. Rating thật trên Grab có thể đạt 4.6–5.0. Có thể dữ liệu đã bị cắt (clip), hoặc `flag` là một chỉ số khác có thang tương tự.
- **Kết luận: đây là GIẢ ĐỊNH, sẽ được nêu rõ trong phần Hạn chế của báo cáo.** Mọi phân tích về rating ở notebook 05 chỉ dùng `has_rating = True`.
""")

# ------------------------------------------------------------------ 8
md(r"""
## 8. Khử trùng listing

Một số cửa hàng được đăng **hai lần** trên Grab: cùng tên, cùng tọa độ, cùng địa chỉ, nhưng khác `storeId` (vd một listing có huy hiệu "Tạp Dề", một listing không).
Nếu giữ nguyên, mật độ không gian và luật co-location sẽ bị thổi phồng.

**Khóa trùng:** tên (bỏ dấu) + tọa độ làm tròn 5 chữ số (~1 m) + phần đầu địa chỉ (số nhà + đường).
Các sạp trong cùng một chợ (vd Chợ Cồn) có **cùng tọa độ nhưng khác số lô**, nên vẫn được giữ là các cửa hàng riêng.
Khi gộp: hợp nhất danh sách loại hình; rating lấy trung bình các listing có đánh giá.
""")
code(r"""
n0 = len(df)
df, dup_groups = cl.deduplicate(df)
print(f"Số listing thuộc nhóm trùng: {len(dup_groups)} | trước: {n0} -> sau: {len(df)} (gộp {n0 - len(df)})")
print(f"Số cặp trùng tọa độ còn lại (khác cửa hàng, vd sạp chợ / trung tâm thương mại): "
      f"{df.duplicated(['latitude', 'longitude']).sum()}")
dup_groups[["storeName", "address", "flag", "productType", "storeId"]].head(12)
""")
code(r"""
log.add("8. Khử trùng listing", n0, len(df), n0 - len(df),
        f"{len(dup_groups)} listing thuộc nhóm trùng; gộp product_types, rating TB")
""")

# ------------------------------------------------------------------ 9
md(r"""
## 9. Phát hiện chuỗi cửa hàng (chain)

`brand` = tên → bỏ dấu → lowercase → cắt phần sau `" - "` hoặc `"("`. Phần sau thường là tên đường hoặc chi nhánh, vd "Cơm Bếp Ngọ- Lê Duẩn" → "com bep ngo".
Hai quy tắc bảo vệ:
- Chỉ cắt khi phần đầu còn **≥ 2 từ**, để tránh brand chung chung kiểu "banh mi" gom nhầm các quán độc lập.
- Gạch nối **không có khoảng trắng** ("Tea-Rex", "Family1-Shop") là một phần của tên nên được giữ nguyên.

`is_chain = True` nếu `brand` xuất hiện ≥ 2 lần (sau khi khử trùng listing, nên đây là ≥ 2 **địa điểm**), **trừ hai trường hợp**:
- **Tên chung chung** (`cl.GENERIC_BRANDS`, vd "bún bò", "trà sữa", "ăn vặt", "phở", "bánh mì"): đây là nhiều quán độc lập trùng tên món chứ không phải một chuỗi. Danh sách được lập thủ công sau khi rà toàn bộ brand có ≥ 2 địa điểm.
- **Sạp trong chợ** (`cl.MARKET_BRANDS`: "Chợ Cồn", "Chợ Hàn"): các sạp đặt tên theo chợ.
""")
code(r"""
df = cl.add_chain(df, min_count=2)
ex = df.loc[df["storeName"].str.contains(r" - |\(|– |- | -", regex=True), ["storeName", "brand"]].head(10)
print("Ví dụ chuẩn hóa tên -> brand:")
print(ex.to_string(index=False))
print(f"\nSố brand: {df['brand'].nunique()} | số brand là chain: {df.loc[df['is_chain'], 'brand'].nunique()} "
      f"| số cửa hàng thuộc chain: {df['is_chain'].sum()} ({df['is_chain'].mean():.1%})")
print(f"Bị loại khỏi chain vì tên chung chung: {(df['is_generic_name'] & (df['brand_count'] >= 2)).sum()} cửa hàng | "
      f"sạp chợ: {df['is_market_stall'].sum()}")
top_brands = df.loc[df["is_chain"], "brand"].value_counts().head(15)
top_brands
""")
code(r"""
fig, ax = plt.subplots(figsize=(8, 5.5))
viz.barh_counts(top_brands, "15 chuỗi cửa hàng có nhiều địa điểm nhất", "Số địa điểm", "Brand (đã chuẩn hóa)", ax=ax)
fig.tight_layout(); viz.save_fig(fig, "01_top_chains"); plt.show()
log.add("9. Phát hiện chain (brand, is_chain)", len(df), len(df), int(df["is_chain"].sum()),
        f"{df.loc[df['is_chain'], 'brand'].nunique()} brand có ≥2 địa điểm")
""")
md(r"""
**Lưu ý:** việc phát hiện chain dựa trên tên nên vẫn còn nhiễu. Có thể bỏ sót chuỗi viết tên khác nhau (vd "Milano Coffee" và "Milano Coffee Premium" đang là hai brand), hoặc gộp nhầm các quán trùng tên ngắn. Cột `brand_count` được giữ để mô hình có thể dùng thay cho cờ nhị phân.
""")

# ------------------------------------------------------------------ 10
md(r"""
## 10. Xuất dữ liệu sạch và bảng log
""")
code(r"""
onehot = ft.multilabel_onehot(df["product_types"])
keep_cols = ["storeId", "storeName", "brand", "brand_count", "is_chain", "is_generic_name", "is_market_stall", "category",
             "address", "district", "ward_final", "ward_name", "ward_district", "ward_addr", "is_ward_conflict",
             "latitude", "longitude", "flag", "rating", "has_rating",
             "productType", "n_types", "has_badge", "n_listings", "merged_ids"]
out = df[keep_cols].copy()
out["product_types"] = df["product_types"].map("|".join)
out["grab_badges"] = df["grab_badges"].map("|".join)
out = pd.concat([out.reset_index(drop=True), onehot.reset_index(drop=True)], axis=1)
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print("Đã ghi:", OUT_PATH.relative_to(ROOT), out.shape)

log_df = log.to_frame()
log_df.to_csv(viz.TABLE_DIR / "01_cleaning_log.csv", index=False, encoding="utf-8-sig")
dropped[["storeId", "storeName", "address", "district_raw", "latitude", "longitude",
         "is_coord_outlier", "is_district_inconsistent"]].to_csv(viz.TABLE_DIR / "01_dropped_records.csv",
                                                                index=False, encoding="utf-8-sig")
log_df
""")
code(r"""
# Kiểm tra lại file đã xuất
chk = pd.read_csv(OUT_PATH)
assert chk["storeId"].is_unique
assert chk["latitude"].between(bbox_buf["lat_min"], bbox_buf["lat_max"]).all()
assert (chk["district"] == cl.DISTRICT_CANONICAL).all()
assert chk.loc[chk["has_rating"], "rating"].notna().all() and chk.loc[~chk["has_rating"], "rating"].isna().all()
print("OK –", chk.shape, "| category:", chk["category"].value_counts().to_dict(),
      "| has_rating:", int(chk["has_rating"].sum()))
print("ward_final:", chk["ward_final"].value_counts().to_dict())
""")

md(r"""
## Tóm tắt notebook 01

| Hạng mục | Kết quả |
|---|---|
| Số bản ghi | 3.256 → **3.208** (loại 16 bản ghi sai vị trí, gộp 32 listing trùng) |
| Tọa độ | 6 điểm ngoài bbox+1km (TP.HCM, Hải Phòng/Quảng Ninh, Liên Chiểu); 8 điểm được giữ nhờ biên 1 km |
| district | 20 biến thể → 1 giá trị; 12 bản ghi địa chỉ thuộc quận/tỉnh khác (gồm 2 cửa hàng ở Hải An – **Hải Phòng**) |
| Phường | trường `ward` và phường trong `address` chỉ khớp **48,5%**; `ward` thiếu hẳn Phước Ninh, Bình Hiên → ưu tiên `address` |
| productType | 51 nhãn gốc → **20 nhãn** (tách 4 nhãn huy hiệu Grab, gộp đồng nghĩa, gom nhãn hiếm) |
| Rating | 1.661 cửa hàng (51,8%) có đánh giá, rating 3.0–4.5; `flag = rating×10` là **giả định** |
| Chain | 274 brand có ≥ 2 địa điểm, 705 cửa hàng (22,0%) |

**Đầu ra:** `data/processed/stores_clean.csv`, `outputs/tables/01_cleaning_log.csv`, `01_dropped_records.csv`, `01_product_type_label_map.csv`,
`outputs/figures/01_*.png`, `outputs/maps/01_coord_cleaning.html`.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
Path("notebooks/01_preprocessing.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print("written")
