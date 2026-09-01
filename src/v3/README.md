# v3 — Benchmark đa quy mô trên kênh ECG

> Phiên bản có con số đẹp nhất và sai nhiều nhất. Bốn lỗi chồng lên nhau.

| | |
|---|---|
| **Kênh tín hiệu** | **ECG** ⚠️ (không phải PPG như sản phẩm) |
| **Cửa sổ** | 30 giây; bước 30 / 10 / 5 / 2.5 giây → chồng lấn 0 / 67 / 83 / 92% ❌ |
| **Đặc trưng** | 16 cột |
| **Làm sạch** | IQR ×1.5 tính trên **toàn bộ** bảng ❌ |
| **Chuẩn hóa** | StandardScaler fit **toàn bộ** rồi mới chia ❌ |
| **Chia dữ liệu** | ngẫu nhiên 80/20 ❌ |
| **Mô hình** | LR, RF, XGBoost, Voting, Stacking, MLP |
| **Điểm công bố** | 98.65% (Stacking) · 98.73% / AUC 0.9998 (RF) |

## Ý tưởng

Nếu 1.360 cửa sổ cho 97%, thì nhiều dữ liệu hơn chắc phải tốt hơn. Nhóm tạo
4 quy mô bằng cách cho cửa sổ 30 giây trượt với bước ngắn dần — cùng một
lượng tín hiệu gốc nhưng sinh ra tới 16.358 hàng.

## Bốn chỗ sai

**1. Cửa sổ chồng lấn tới 92%.** Với bước 2.5 giây, hai hàng liên tiếp dùng
chung 27.5/30 giây tín hiệu — gần như là bản sao của nhau. Chia ngẫu nhiên
lúc này gần như chắc chắn đặt bản sao của mẫu test vào train. Đây không còn
là "nhớ mặt bệnh nhân" nữa mà là **nhớ gần đúng chính câu hỏi**.

> Lưu ý quan trọng: chồng lấn **tự nó không phải lỗi**. v4 vẫn dùng bước 10
> giây. Nó chỉ nguy hiểm khi đi kèm chia ngẫu nhiên. Chia theo bệnh nhân thì
> mọi bản sao đều nằm cùng một phía, không thể rò rỉ.

**2. Chuẩn hóa trước khi chia.** `StandardScaler().fit_transform(X)` chạy
trên toàn bộ bảng rồi mới `train_test_split`. Trung bình và độ lệch chuẩn
dùng để chuẩn hóa train đã được tính có phần đóng góp của test.

**3. Lọc IQR toàn cục.** Ngưỡng Q1/Q3 tính trên cả bảng, và những hàng "cực
đoan" bị xóa — kể cả trong test. Mà cửa sổ AFib nặng thì vốn dĩ cực đoan.
Đề thi dễ đi lần nữa, đúng kiểu lỗi của v2 nhưng ở dạng tự động.

**4. Vẫn chia ngẫu nhiên, không theo bệnh nhân.**

## Một cái bẫy khi so sánh

Chấm lại bằng LOSO, v3 cho ~97% ở mức bệnh nhân — **cao hơn v4 (~94%)**. Đừng
vội kết luận v3 tốt hơn: chênh lệch này đến từ **loại tín hiệu**, không phải
phương pháp. ECG có sóng R rất nhọn, dễ dò chính xác; PPG thì sóng tù và dễ
nhiễu. Đo bằng ECG rồi khoe điểm, trong khi sản phẩm chạy bằng PPG, là so sánh
khập khiễng — và đó chính là vấn đề của v3, không phải điểm cộng.

So công bằng thì phải so cùng loại tín hiệu: v1, v2, v4 (đều dùng PPG) cho
~94% mức bệnh nhân, ngang nhau.

## Một chi tiết ít ai để ý

v3 trích đặc trưng từ **kênh ECG** (`sub['ecg'].values`), không phải PPG.
Nghĩa là con số v3 không so sánh trực tiếp được với v1/v2, và một sản phẩm
đeo tay dùng cảm biến quang lại đang dựa trên kết quả đo bằng điện tâm đồ.
Thư mục này giữ nguyên kênh ECG để tái dựng trung thực.

Ngoài ra, cột tên `SampEn` trong v3 **không phải Sample Entropy** — công thức
thật là `std(diff_rr) / (sdnn + 1e-6)`, chỉ là một tỉ số độ tản. Nó vẫn được
báo cáo là đặc trưng quan trọng thứ nhì.

## Điều v3 đã tự nhận ra

Notebook `v3_pipeline/mimic/03_project_report.ipynb`, mục 5C, **tự chẩn đoán
đúng bệnh**: nói rõ rằng với 91% chồng lấn và chia ngẫu nhiên thì mô hình chỉ
đang ghi nhớ bệnh nhân, và cần `GroupKFold` theo mã bệnh nhân. Chẩn đoán đúng
nhưng chưa kịp chữa — việc chữa chính là v4.

Notebook `04_mimic_v3_benchmark.ipynb` đã bước một chân sang hướng đúng: dùng
chia theo khối liên tục cho các quy mô chồng lấn, và điểm rơi ngay xuống
94–95%. Nhưng quy mô 1.360 — cái duy nhất vẫn chia ngẫu nhiên, đạt 98.73% —
lại là cái được chọn làm "kết quả chính".

## Chạy thử

```bash
python src/v3/pipeline.py
```

Script chạy cả 4 quy mô, mỗi quy mô chấm hai lần (cách gốc và LOSO), để thấy
rõ một quy luật: **càng chồng lấn nhiều, điểm gốc càng đẹp và phần điểm ảo
càng lớn**. Kết quả lưu ở `models/v3/results.json`, báo cáo đầy đủ ở
[`src/report/03_v3_report.ipynb`](../report/03_v3_report.ipynb).
