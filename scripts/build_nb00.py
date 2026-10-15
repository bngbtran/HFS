"""Sinh notebooks/00_problem_and_data.ipynb – 4 bước đầu của quy trình Data Science."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import NB, SETUP  # noqa: E402

nb = NB()
nb.md(r"""
# 00 – Định nghĩa bài toán, hướng tiếp cận và dữ liệu

Dự án đi theo quy trình **Data Science Methodology** (DAM501):

```
Problem definition → Analytic approach → Data requirements → Data collection
        ↑                                        ↑
        └──────────── Feedback ←── Deployment ←── Evaluation ←── Model ←── Data preprocessing ←── Data understanding
```

| Bước | Nội dung | Notebook / file |
|---|---|---|
| 1. Problem definition | Câu hỏi kinh doanh | **00** (notebook này) |
| 2. Analytic approach | Chọn loại bài toán & kỹ thuật | **00** |
| 3. Data requirements | Cần dữ liệu gì | **00** |
| 4. Data collection | Lấy dữ liệu từ đâu | **00**, `src/scrape_gmaps.py` (tùy chọn) |
| 5. Data understanding | Khảo sát dữ liệu thô, EDA | **01** (mục 1), **02** |
| 6. Data preprocessing | Làm sạch, chuẩn hóa, tạo đặc trưng | **01**, **05** (đặc trưng không gian) |
| 7. Model | Phân cụm, luật kết hợp, phân lớp | **03**, **04**, **05** |
| 8. Evaluation | Đánh giá chống rò rỉ không gian | **05** |
| 9. Deployment | Bảng dữ liệu phân tích, bản đồ, công cụ gợi ý | **06**, `src/recommend.py` |
| 10. Feedback | Bài học & vòng lặp cải tiến | **06** (mục cuối), `reports/report.md` |
""")

nb.md(r"""
## 1. Problem definition – Định nghĩa bài toán

**Bối cảnh.** Quận Hải Châu là trung tâm hành chính, thương mại của Đà Nẵng và có mật độ cửa hàng F&B rất cao.
Người muốn mở quán mới (hộ kinh doanh, chuỗi) thường chọn loại hình và vị trí theo cảm tính.

**Câu hỏi kinh doanh chính:**
> *Tại một vị trí cho trước ở Hải Châu, nên mở **loại hình F&B nào** cho phù hợp với môi trường kinh doanh xung quanh?*

**Các câu hỏi phụ:**
1. Cửa hàng F&B tập trung ở đâu? Có những "điểm nóng" (hotspot) nào, và mỗi hotspot có đặc trưng gì?
2. Những loại hình nào thường mở **cạnh nhau** (đồng vị trí), và những loại nào "né" nhau?
3. Ở mỗi khu vực, loại hình nào đang **thiếu** so với mức mà môi trường xung quanh "gợi ý"?

**Thành công được đo bằng:**
- Mô hình dự đoán loại hình tốt hơn baseline (đoán lớp phổ biến nhất) **khi đánh giá trên khu vực mới**, tức dùng GroupKFold theo không gian.
- Có danh sách vị trí gợi ý cho từng loại hình, kèm căn cứ (mật độ, đối thủ, luật đồng vị trí).
- Sản phẩm cuối là một **tệp dữ liệu đã phân tích** (`stores_analyzed.csv`) và bản đồ / bảng gợi ý.
""")

nb.md(r"""
## 2. Analytic approach – Hướng tiếp cận phân tích

| Câu hỏi | Loại bài toán | Kỹ thuật |
|---|---|---|
| Hotspot ở đâu? | **Mô tả – phân cụm không gian** | K-Means (UTM), DBSCAN (haversine), HDBSCAN |
| Loại hình nào mở cạnh nhau? | **Mô tả – luật kết hợp / co-location** | Apriori, FP-Growth; transaction theo cửa hàng, theo lưới, theo lân cận |
| Vị trí này hợp loại hình nào? | **Dự đoán – phân lớp đa lớp** | Decision Tree, Naive Bayes, KNN, Random Forest, Gradient Boosting; baseline = lớp phổ biến nhất |
| Cửa hàng nào "lạc loài"? | **Phát hiện ngoại lai** | LOF, Isolation Forest |
| Nên mở ở đâu? | **Đề xuất (prescriptive)** | Chấm điểm ô lưới = mô hình phân lớp + mật độ + đối thủ + lift đồng vị trí |

**Biến mục tiêu** của bài toán dự đoán là `type_group`, tức nhóm loại hình chính của cửa hàng (9 nhóm).
**Đặc trưng** chỉ mô tả *môi trường xung quanh vị trí* (số lượng và cơ cấu loại hình lân cận, mật độ, khoảng cách tới sông/hotspot).
Ta không dùng tên hay nhãn của chính cửa hàng, vì khi gợi ý cho một vị trí mới thì chưa có cửa hàng ở đó.

**Vì sao không dự đoán rating?** Chỉ 51,8% cửa hàng có rating, và ý nghĩa của trường `flag` là *giả định* (xem notebook 01).
Bài toán loại hình dùng được toàn bộ dữ liệu và trả lời trực tiếp câu hỏi kinh doanh.
""")

nb.md(r"""
## 3. Data requirements – Yêu cầu dữ liệu

| Nhóm dữ liệu | Thuộc tính cần | Có trong dữ liệu? |
|---|---|---|
| Định danh cửa hàng | tên, mã | ✅ `storeName`, `storeId` |
| Vị trí | tọa độ, địa chỉ, phường | ✅ `latitude`, `longitude`, `address`, `ward` (có nhiễu) |
| Loại hình | nhóm FOOD/SHOP, danh mục món | ✅ `category`, `productType` (đa nhãn) |
| Chất lượng / mức độ quan tâm | rating, số lượt đánh giá | ⚠️ chỉ có `flag` (giả định = rating × 10), **không có số lượt đánh giá** |
| Kinh tế | giá, doanh thu, lượng khách | ❌ không có |
| Thời gian | giờ mở cửa, ngày mở | ❌ không có |
| Bối cảnh đô thị | dân số, văn phòng, trường học, giá thuê | ❌ không có (hướng phát triển) |

Dữ liệu hiện có **đủ cho các bài toán không gian** (phân cụm, đồng vị trí, dự đoán loại hình theo môi trường xung quanh), nhưng **không đủ** để dự đoán doanh thu hay mức độ thành công của quán.
""")

nb.md(r"""
## 4. Data collection – Thu thập dữ liệu

**Nguồn chính:** `data/raw/DaNangData.json`, gồm danh sách cửa hàng trên nền tảng giao đồ ăn / đi chợ (GrabFood, GrabMart) trong phạm vi Q. Hải Châu, được crawl sẵn. Dữ liệu này **không được sửa** trong suốt dự án.

**Nguồn bổ sung (tùy chọn, chưa chạy):** `src/scrape_gmaps.py` cào trang danh sách của Google Maps theo lưới 500 m, để lấy thêm rating thật (thang 1–5) và loại hình theo Google. Lưu ý:
- Việc cào giao diện web trái Điều khoản dịch vụ của Google Maps. Dữ liệu chỉ dùng cho học tập, không công khai.
- Mỗi truy vấn trả tối đa khoảng 120 kết quả; người dùng chưa đăng nhập không xem được số lượt đánh giá.
- Lệnh chạy: `.venv\Scripts\python -m src.scrape_gmaps --max-queries 10 --headful` (thử), rồi bỏ `--max-queries` để chạy đủ.
""")
nb.code(SETUP + r"""
raw = cl.load_raw(ROOT / "data" / "raw" / "DaNangData.json")
print("Số bản ghi:", len(raw), "| số trường:", raw.shape[1])
print("Các trường:", list(raw.columns))
raw.head(3).T
""")
nb.code(r"""
gm = ROOT / "data" / "raw" / "gmaps" / "gmaps_places.csv"
if gm.exists():
    g = pd.read_csv(gm)
    print(f"Có dữ liệu Google Maps (thử nghiệm): {len(g)} địa điểm – chưa dùng trong phân tích chính")
else:
    print("Chưa có dữ liệu Google Maps – phân tích dùng nguồn Grab.")
""")
nb.write("00_problem_and_data.ipynb")
