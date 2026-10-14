# HaiChau F&B Spatial Mining

Khai phá dữ liệu không gian các cửa hàng F&B quận Hải Châu (Đà Nẵng): phân cụm điểm nóng, luật kết hợp đồng vị trí và dự đoán mức đánh giá.

## Cài đặt (Windows)
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name python3
```

## Chạy
Mở các notebook trong `notebooks/` và chạy **theo thứ tự** 01 → 06. Có thể chạy không giao diện:
```powershell
cd notebooks
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 01_preprocessing.ipynb
```
Notebook được sinh từ `scripts/build_nbXX.py`. Nếu sửa nội dung notebook, nên sửa trong script rồi chạy lại `python scripts/build_nbXX.py`.

## Cấu trúc
```
data/raw/            dữ liệu gốc (KHÔNG sửa)
data/processed/      stores_clean.csv, stores_features.csv, ...
notebooks/           01_preprocessing → 06_outliers_and_recommendation
src/                 cleaning.py, geo.py, features.py, viz.py (logic dùng chung)
scripts/             script sinh notebook
outputs/figures/     hình PNG (dpi 150)
outputs/maps/        bản đồ HTML (folium) / GeoJSON
outputs/tables/      bảng CSV
reports/             báo cáo tổng hợp
```

## Tiến độ
- [x] 01 – Tiền xử lý
- [ ] 02 – EDA
- [ ] 03 – Phân cụm điểm nóng
- [ ] 04 – Luật kết hợp
- [ ] 05 – Dự đoán rating
- [ ] 06 – Ngoại lai & gợi ý vị trí
- [ ] Báo cáo

## Ghi chú phương pháp
- `random_state = 42` ở mọi chỗ có ngẫu nhiên.
- Tọa độ được chiếu sang **UTM zone 49N (EPSG:32649)**: Đà Nẵng nằm ở kinh độ ~108.2°E, thuộc zone 49 (108°–114°E).
- `rating = flag / 10`, `flag = 0` ⇒ chưa có đánh giá: đây là **giả định**, xem notebook 01 mục 7.
