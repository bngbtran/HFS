# HaiChau F&B Spatial Mining

Khai phá dữ liệu không gian các cửa hàng F&B quận Hải Châu (Đà Nẵng): phân cụm điểm nóng, luật kết hợp đồng vị trí và dự đoán loại hình F&B phù hợp với vị trí. Báo cáo: `reports/report.md`.

## Cài đặt (Windows)
```powershell
conda create -n dam python=3.13 -y
conda activate dam
pip install -r requirements.txt
python -m ipykernel install --user --name dam --display-name "Python 3 (dam)"
```
Các lệnh bên dưới đều chạy trong env `dam` (đã `conda activate dam`).

## Chạy
Mở các notebook trong `notebooks/` và chạy **theo thứ tự** 00 → 06. Có thể chạy không giao diện:
```powershell
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_preprocessing.ipynb
```
Notebook được sinh từ `scripts/build_nbXX.py`. Nếu sửa nội dung notebook, nên sửa trong script rồi chạy lại `python scripts/build_nbXX.py`.

## Cấu trúc
```
data/raw/            dữ liệu gốc (KHÔNG sửa)
data/processed/      stores_clean.csv, stores_clusters.csv, stores_features.csv, stores_predictions.csv, stores_analyzed.csv
notebooks/           00_problem_and_data → 06_outliers_and_recommendation
src/                 cleaning, geo, features, viz, clustering, assoc, modeling, recommend, scrape_gmaps
scripts/             script sinh notebook
outputs/figures/     hình PNG (dpi 150)
outputs/maps/        bản đồ HTML (folium) / GeoJSON
outputs/tables/      bảng CSV
outputs/models/      mô hình đã huấn luyện (joblib)
reports/             báo cáo tổng hợp
```

## Quy trình (Data Science Methodology – DAM501)

| Bước | Notebook / file |
|---|---|
| 1. Problem definition · 2. Analytic approach · 3. Data requirements · 4. Data collection | `00_problem_and_data` (+ `src/scrape_gmaps.py`, tùy chọn) |
| 5. Data understanding | `01_preprocessing` (mục 1), `02_eda` |
| 6. Data preprocessing | `01_preprocessing`, đặc trưng không gian trong `05_classification` |
| 7. Model | `03_clustering` (hotspot), `04_association_rules` (co-location), `05_classification` (dự đoán loại hình) |
| 8. Evaluation | `05_classification` (StratifiedKFold vs GroupKFold không gian) |
| 9. Deployment | `06_outliers_and_recommendation`, `src/recommend.py`, `data/processed/stores_analyzed.csv`, `outputs/maps/*.geojson` |
| 10. Feedback | `06_outliers_and_recommendation` (mục 5), `reports/report.md` |

**Sản phẩm chính:** `data/processed/stores_analyzed.csv` (+ `stores_analyzed_dictionary.csv`) – mỗi cửa hàng kèm hotspot, bối cảnh không gian, loại hình dự đoán / mức phù hợp, cờ ngoại lai.

**Công cụ gợi ý:**
```powershell
python -m src.recommend --lat 16.0678 --lon 108.2208      # loại hình nào hợp vị trí này?
python -m src.recommend --type "Trà sữa" --top 5          # nên mở trà sữa ở đâu?
```

## Ghi chú phương pháp
- `random_state = 42` ở mọi chỗ có ngẫu nhiên.
- Tọa độ được chiếu sang **UTM zone 49N (EPSG:32649)**: Đà Nẵng nằm ở kinh độ ~108.2°E, thuộc zone 49 (108°–114°E).
- `rating = flag / 10`, `flag = 0` ⇒ chưa có đánh giá: đây là **giả định**, xem notebook 01 mục 7.
