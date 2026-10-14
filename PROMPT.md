# Dự án: HaiChau F&B Spatial Mining

## Bối cảnh
Tôi là sinh viên môn Data Mining. Đề tài: **"Khai phá dữ liệu không gian các cửa hàng F&B quận Hải Châu: phân cụm điểm nóng, luật kết hợp đồng vị trí và dự đoán mức đánh giá"**.
Hãy giúp tôi xây dựng toàn bộ pipeline phân tích bằng Python, có thể tái lập, trình bày rõ để nộp báo cáo và demo. Viết markdown/giải thích trong notebook bằng **tiếng Việt**, tên biến/hàm bằng tiếng Anh.

## Dữ liệu đầu vào
File: `data/raw/DaNangData.json`, cấu trúc `{"DaNangData": [ ... ]}`, có 3.256 bản ghi, mỗi bản ghi đủ 11 trường:
`storeName, ward, city, district, address, latitude, longitude, flag, category, productType, storeId`

Những gì đã quan sát được (cần kiểm chứng lại bằng code):
- `category`: FOOD 3.018, SHOP 238, **mất cân bằng**.
- `district`: gần như toàn bộ là Q. Hải Châu nhưng viết **khoảng 20 kiểu khác nhau** ("Q.Hải Châu", "Quận Hải châu", có khoảng trắng thừa hoặc ký tự Unicode tổ hợp, "Quận Hải Châu, Đà Nẵng"...). Có vài bản ghi ghi "Quận Hải An", "Phường An Hải Bắc/Đông" là nhập sai.
- `ward`: 31 giá trị dạng "Da Nang - Hai Chau II Ward of Hai Chau District". Cần tách lấy tên phường. Lưu ý `ward` có thể mâu thuẫn với `address`.
- `productType`: **đa nhãn**, phân tách bằng ", " (vd "Rice, Hotpot & Grill, Noodles & Congee"). Có 339 tổ hợp nhưng chỉ khoảng 20 nhãn gốc. Có cả trùng lặp trong cùng chuỗi (vd "Noodles, Noodles & Congee").
- `flag`: 0 chiếm khoảng 1.575 bản ghi, còn lại 30–45. **Giả thuyết: rating × 10, và 0 = chưa có đánh giá.** Phải kiểm chứng bằng phân phối và nêu rõ đây là giả định trong báo cáo.
- Tọa độ có **outlier**: lat 10.6–20.96, tức có điểm nằm ở TP.HCM và Hà Nội. Hải Châu nằm khoảng lat 16.00–16.09, lon 108.18–108.26.
- `storeName` có 460 tên trùng, tức là chuỗi cửa hàng (chain). Một số bản ghi có giá trị rỗng/null.

## Cấu trúc thư mục cần tạo
```
data/raw/            # dữ liệu gốc, KHÔNG sửa
data/processed/      # stores_clean.csv, stores_features.csv, ...
notebooks/           # 01_ → 06_, chạy theo thứ tự
src/                 # module dùng chung: cleaning.py, geo.py, features.py, viz.py
outputs/figures/  outputs/maps/  outputs/tables/
reports/             # báo cáo tổng hợp
requirements.txt  README.md
```

## Môi trường
- Windows, Python 3.10+. Tạo venv `.venv`. Thư viện: pandas, numpy, scikit-learn, mlxtend, folium, matplotlib, seaborn, hdbscan (hoặc `sklearn.cluster.HDBSCAN` nếu sklearn ≥1.3), imbalanced-learn, unidecode, jupyter.
- Kiểm tra Python có sẵn chưa trước khi cài. Ghi đủ phiên bản vào requirements.txt.
- Đặt `random_state=42` ở mọi chỗ có ngẫu nhiên.

## Các bước thực hiện

### 01_preprocessing.ipynb: Tiền xử lý
1. Load JSON thành DataFrame, in thống kê missing, kiểu dữ liệu, giá trị unique của từng cột.
2. Chuẩn hóa chuỗi: Unicode NFC, strip, gộp khoảng trắng.
3. Chuẩn hóa `district` về một giá trị. Đánh cờ `is_district_inconsistent` cho các bản ghi ngoài Hải Châu.
4. Tách `ward_name` từ `ward`.
5. Làm sạch tọa độ: loại điểm ngoài bounding box Hải Châu (có thể nới thêm biên khoảng 1km). Báo cáo số điểm bị loại kèm bản đồ before/after.
6. `productType`: tách thành list, khử trùng, tạo one-hot đa nhãn (`pt_<label>`), tạo thêm `n_types`.
7. `flag`: tạo `rating = flag/10` (NaN nếu flag=0), `has_rating`. Vẽ phân phối.
8. Phát hiện chain: chuẩn hóa tên (lowercase, bỏ dấu, cắt phần sau " - "/"(" ), tạo `brand`, `is_chain` (brand xuất hiện ≥2 lần).
9. Xuất `data/processed/stores_clean.csv` và một bảng log tóm tắt các bước làm sạch (trước/sau, số bản ghi bị ảnh hưởng).

### 02_eda.ipynb: Phân tích khám phá
- Phân phối theo category, productType (sau khi tách), phường, rating, chain/độc lập.
- Crosstab phường × loại hình, rating trung bình theo loại hình và theo phường.
- Bản đồ Folium: điểm theo category, HeatMap mật độ.
- Viết 5–8 nhận xét (insight) bằng tiếng Việt.

### 03_clustering.ipynb: Phân cụm điểm nóng
- **K-Means** trên tọa độ (chiếu sang mét hoặc UTM 48N trước, không dùng độ trực tiếp). Chọn k bằng Elbow và Silhouette.
- **DBSCAN** với `metric='haversine'`, `algorithm='ball_tree'`, `eps = mét / 6371000` (radian). Thử eps 100–300m, min_samples 5–15. Dùng k-distance plot để chọn eps.
- **HDBSCAN** để so sánh.
- Bảng so sánh: số cụm, % nhiễu, Silhouette, Davies-Bouldin. Thảo luận vì sao density-based phù hợp với dữ liệu không gian.
- Mô tả từng hotspot: số cửa hàng, loại hình chủ đạo, rating TB, tên đường tiêu biểu (lấy từ address).
- Phân cụm **phường** theo vector tỉ lệ loại hình (K-Means hoặc Hierarchical kèm dendrogram).
- Xuất `outputs/maps/hotspots.html` và `outputs/tables/cluster_profiles.csv`.

### 04_association_rules.ipynb: Luật kết hợp
- **(a) Trong cửa hàng**: transaction = tập productType của 1 cửa hàng. Chạy Apriori và FP-Growth, so sánh thời gian chạy.
- **(b) Co-location không gian** (trọng tâm): chia lưới khoảng 200m×200m (thử thêm 150m/300m). Transaction = tập các loại hình có trong ô. Cũng thử cách neighborhood: mỗi cửa hàng cùng các loại hình trong bán kính R.
- Chọn ngưỡng min_support, min_confidence hợp lý, giải thích lý do chọn. Lọc theo lift > 1. Xếp hạng theo lift, conviction.
- Diễn giải top 10–15 luật bằng ngôn ngữ kinh doanh. Cảnh báo luật hiển nhiên do "Coffee - Tea - Juice" quá phổ biến.
- Xuất bảng luật ra CSV, vẽ scatter support–confidence–lift.

### 05_classification.ipynb: Dự đoán mức đánh giá
- Chỉ dùng các bản ghi `has_rating`. Nhãn: `high_rating = rating >= 4.2` (kiểm tra phân phối rồi chọn ngưỡng hợp lý, có thể cân nhắc chia 3 lớp).
- Feature engineering không gian (dùng BallTree haversine, **loại chính điểm đó khi đếm**):
  - số cửa hàng trong bán kính 100/300/500m
  - số đối thủ cùng loại hình trong 300m
  - khoảng cách tới cửa hàng gần nhất, tới tâm hotspot gần nhất, tới bờ sông Hàn (xấp xỉ bằng đường Bạch Đằng, lon khoảng 108.224)
  - nhãn cụm DBSCAN, one-hot productType, `is_chain`, `n_types`, phường
- Mô hình: Decision Tree, Naive Bayes, KNN, Random Forest, Gradient Boosting/XGBoost (tùy chọn). Baseline: majority class.
- **Đánh giá chống rò rỉ không gian**: ngoài Stratified K-Fold thường, dùng **GroupKFold theo ô lưới hoặc phường** rồi so sánh hai kết quả.
- Metrics: Accuracy, Precision, Recall, F1-macro, ROC-AUC, confusion matrix. Xử lý mất cân bằng (class_weight, SMOTE trong pipeline).
- Feature importance và permutation importance: yếu tố nào ảnh hưởng tới rating?
- Bài phụ (tùy chọn): phân lớp FOOD/SHOP để minh họa dữ liệu mất cân bằng.

### 06_outliers_and_recommendation.ipynb: Ngoại lai & ứng dụng
- LOF và Isolation Forest trên đặc trưng không gian, tìm cửa hàng "lạc loài". Vẽ lên bản đồ.
- Gợi ý vị trí mở quán: với 1 loại hình đầu vào, chấm điểm từng ô lưới theo (mật độ tổng cao, đối thủ cùng loại thấp, có luật co-location ủng hộ). Xuất top 5 ô kèm bản đồ.
- Xuất `outputs/maps/*.geojson` hoặc JSON để sau này tích hợp vào web React (dự án gốc ở `D:\MapSiteDaNang_FE`, dùng Leaflet/Map).

### reports/
- `report.md`: Giới thiệu, Dữ liệu, Tiền xử lý, Phương pháp, Kết quả từng phần (hình + bảng), Thảo luận, **Hạn chế** (1 quận, không có giá/doanh thu/thời gian, ý nghĩa `flag` là giả định, dữ liệu crawl có nhiễu), Hướng phát triển (thêm mật độ dân số theo phường, dữ liệu review).

## Yêu cầu cách làm việc
- Làm **tuần tự từng notebook**. Xong mỗi bước thì chạy thử end-to-end, tóm tắt kết quả chính và hỏi tôi trước khi sang bước tiếp theo.
- Logic dùng lại phải nằm trong `src/`, notebook chỉ gọi hàm và trình bày.
- Mỗi biểu đồ có tiêu đề, nhãn trục, chú thích tiếng Việt, lưu vào `outputs/figures/` (PNG, dpi 150).
- Không tự bịa số liệu. Mọi nhận xét phải dựa trên output đã chạy thật.
- Khi có lựa chọn tham số (eps, k, support...), giải thích căn cứ chọn, vì đây là phần giảng viên hay hỏi.
- Khởi tạo git repo, commit sau mỗi bước hoàn thành.

Bắt đầu bằng: thiết lập môi trường + cấu trúc thư mục, rồi làm notebook 01.
