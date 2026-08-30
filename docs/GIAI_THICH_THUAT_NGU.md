# Giải Thích Thuật Ngữ Dễ Hiểu — HealthSense ML

Tài liệu này giải thích mọi thuật ngữ trong project bằng ngôn ngữ đời thường,
theo đúng thứ tự dữ liệu chảy qua pipeline.

---

## 1. Tín hiệu đầu vào

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **PPG** (Photoplethysmography) | Cách đo nhịp tim bằng ánh sáng — LED chiếu vào da, máu chảy qua làm lượng ánh sáng phản xạ thay đổi theo từng nhịp tim. Đây chính là tín hiệu mà cảm biến MAX30102 trên vòng đeo HealthSense thu được. |
| **AFib / Rung Nhĩ** (Atrial Fibrillation) | Bệnh loạn nhịp tim phổ biến nhất: hai buồng tim trên (tâm nhĩ) "rung" hỗn loạn thay vì bóp đều. Dấu hiệu nhận biết: **khoảng cách giữa các nhịp tim trở nên bất thường, không theo quy luật**. Nguy hiểm vì làm tăng nguy cơ đột quỵ gấp ~5 lần. |
| **MIMIC PERform AF** | Bộ dữ liệu y tế công khai: 19 bệnh nhân bị AFib + 16 người nhịp bình thường, mỗi người được ghi 20 phút tín hiệu PPG trong bệnh viện (ICU). Dùng để dạy mô hình phân biệt AFib với nhịp thường. |
| **125 Hz** | Tần số lấy mẫu: cảm biến ghi 125 con số mỗi giây. |
| **Butterworth bandpass 0.5–8 Hz** | Bộ lọc "cắt tỉa" tín hiệu: bỏ phần trôi chậm (< 0.5 Hz, do thở/cử động chậm) và phần rung nhanh (> 8 Hz, nhiễu điện) — chỉ giữ dải tần chứa nhịp tim. |

## 2. Từ tín hiệu đến con số

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **Beat detection / Phát hiện nhịp** | Tìm các "đỉnh núi" trong sóng PPG — mỗi đỉnh là một nhịp tim. |
| **NN / IBI** (khoảng NN) | Khoảng thời gian giữa 2 nhịp tim liên tiếp, tính bằng mili-giây. Ví dụ tim đập 60 lần/phút thì NN ≈ 1000 ms. **Toàn bộ 16 đặc trưng đều tính từ chuỗi số này.** |
| **HRV** (Heart Rate Variability) | Độ biến thiên nhịp tim — nhịp tim khỏe mạnh không đều tăm tắp như máy, mà dao động nhẹ. AFib làm sự dao động này trở nên hỗn loạn bất thường, nên HRV là "vân tay" để phát hiện AFib. |
| **Cửa sổ trượt 30s / bước 10s** | Thay vì phân tích cả 20 phút một lần, ta cắt thành từng đoạn 30 giây, mỗi lần dịch đi 10 giây (các đoạn chồng lên nhau). Mỗi đoạn 30s ➔ 1 hàng dữ liệu với 16 con số đặc trưng. |
| **record_id** | "Số định danh bệnh nhân" gắn vào từng hàng dữ liệu. Nghe đơn giản nhưng là **cột quan trọng nhất của v4** — không có nó thì không thể chia dữ liệu đúng cách (xem Data Leakage bên dưới). |

## 3. 16 đặc trưng HRV (nhóm theo ý nghĩa)

**Nhóm thời gian** — đo trực tiếp trên chuỗi NN:
- **HR_mean**: nhịp tim trung bình (BPM).
- **Mean_NN**: khoảng NN trung bình (ms) — nghịch đảo của HR_mean.
- **SDNN**: độ lệch chuẩn của NN — nhịp càng "loạn" số này càng lớn.
- **RMSSD**: giống SDNN nhưng nhạy với thay đổi **giữa 2 nhịp liền kề** — chỉ số kinh điển để bắt AFib.
- **NN50 / pNN50**: đếm số lần 2 nhịp liền kề lệch nhau > 50 ms (và tỉ lệ %). Người AFib có pNN50 rất cao.
- **CV**: SDNN chia Mean_NN — "độ loạn tương đối" để so sánh công bằng giữa người tim nhanh và tim chậm.

**Nhóm tần số** — phân tích "nhịp điệu của sự dao động":
- **LF** (Low Frequency, 0.04–0.15 Hz): dao động chậm, phản ánh hệ thần kinh giao cảm. ⚠️ **Không đáng tin trên cửa sổ 30 giây** (chuẩn y khoa yêu cầu đo ≥ 2 phút) nên v4 loại bỏ nhóm này khi huấn luyện.
- **HF** (High Frequency, 0.15–0.4 Hz): dao động theo nhịp thở.
- **Total_Power, LF/HF Ratio, LF_norm, HF_norm**: các biến tổ hợp của 2 dải trên.

**Nhóm phi tuyến** — đo mức độ "hỗn loạn":
- **SD1 / SD2 (Poincaré)**: vẽ mỗi cặp nhịp liên tiếp thành 1 điểm trên đồ thị; đám điểm tạo hình elip. SD1 = bề ngang elip (loạn ngắn hạn), SD2 = bề dài (loạn dài hạn). AFib làm elip phình tròn ra.
- **SampEn** (Sample Entropy): đo độ "khó đoán" của chuỗi nhịp. Nhịp đều ➔ entropy thấp; AFib hỗn loạn ➔ entropy cao. Một trong những đặc trưng mạnh nhất cho AFib.

## 4. Huấn luyện mô hình

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **Logistic Regression** | Mô hình đơn giản nhất: cộng trừ có trọng số 13 đặc trưng rồi ra xác suất. Dễ giải thích, làm "mốc chuẩn" so sánh. |
| **Random Forest** | "Hội đồng" hàng trăm cây quyết định, mỗi cây nhìn dữ liệu một góc khác nhau rồi bỏ phiếu. Khỏe, khó overfit. |
| **XGBoost** | Các cây quyết định học nối tiếp nhau — cây sau sửa lỗi của cây trước. Thường mạnh nhất trên dữ liệu dạng bảng. |
| **StandardScaler** | Đưa mọi đặc trưng về cùng thang đo (trung bình 0, độ lệch 1) — để đặc trưng có số lớn (Total_Power hàng chục nghìn) không "lấn át" đặc trưng số bé (CV ~0.05). |
| **Outlier / IQR** | Outlier = giá trị bất thường vượt xa số đông. IQR là cách xác định ngưỡng "xa" dựa trên khoảng tứ phân vị. v4 dùng ngưỡng nới lỏng (×3.0) vì **với AFib, "bất thường" chính là dấu hiệu bệnh** — lọc mạnh tay sẽ vứt nhầm chính các ca cần phát hiện. |
| **Hyperparameter tuning** | Thử nhiều "cấu hình núm vặn" của mô hình (số cây, độ sâu...) và giữ cấu hình tốt nhất. |

## 5. Chống Data Leakage — trái tim của v4

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **Data Leakage** (rò rỉ dữ liệu) | Khi thông tin của phần kiểm tra (test) "lọt" vào quá trình huấn luyện — như học sinh được xem trước đề thi. Điểm cao nhưng ảo. |
| **Subject Leakage** | Dạng leakage nguy hiểm nhất trong dữ liệu y sinh: các đoạn 30s của **cùng một bệnh nhân** nằm cả ở train lẫn test. Mô hình không học "dấu hiệu AFib" mà học "nhận mặt từng bệnh nhân" — vẫn đạt 99% nhưng gặp bệnh nhân mới là sai. Đây là lý do kết quả v1–v3 (98–99%) không dùng được. |
| **LOSO** (Leave-One-Subject-Out) | Cách chia chuẩn vàng: có 35 bệnh nhân thì chạy 35 vòng, mỗi vòng giấu trọn 1 bệnh nhân làm đề thi, train trên 34 người còn lại. Điểm số phản ánh đúng câu hỏi thực tế: *"gặp người mới, mô hình có nhận ra AFib không?"* |
| **Nested CV / GroupKFold** | Ngay cả việc chọn cấu hình mô hình cũng chỉ được dùng dữ liệu train (chia tiếp thành các nhóm bệnh nhân nhỏ) — test tuyệt đối không được đụng vào, kể cả gián tiếp. |
| **Fit train-only** | Mọi bước tiền xử lý (chuẩn hóa, lọc outlier) chỉ được "học thông số" từ train của từng vòng, rồi áp lên test — không bao giờ ngược lại. |
| **Cross-dataset validation** | Bài kiểm tra khắc nghiệt hơn cả LOSO: train trên dataset này, test trên một dataset **hoàn toàn khác** (khác bệnh viện, khác loại cảm biến, khác quần thể). Mô hình giữ được điểm tốt qua bài này = bằng chứng mạnh nhất rằng nó học được "dấu hiệu bệnh" thật. |
| **MIT-BIH AFDB** | Dataset thứ hai của project: 23 bệnh nhân đo ECG ~10 giờ/người, nhiều người bị **AF kịch phát** — cùng một người có cả đoạn AFib lẫn đoạn bình thường được gán nhãn theo thời điểm. |
| **AF kịch phát** (Paroxysmal AF) | Rung nhĩ đến rồi đi từng đợt (vài phút đến vài giờ) thay vì liên tục — dạng khó phát hiện nhất và cũng là lý do vòng đeo theo dõi 24/7 có giá trị. |
| **QRS annotation** | File đánh dấu sẵn vị trí từng nhịp tim trong bản ghi ECG (do chuyên gia/thuật toán chuẩn tạo) — cho phép tính chuỗi NN mà không cần xử lý sóng thô. |
| **Pooled LOSO** | Gộp cả 2 dataset (60 bệnh nhân) rồi vẫn thi kiểu LOSO: 60 vòng, mỗi vòng giấu trọn 1 người. Cách đánh giá của mô hình cuối cùng. |
| **Sample weight (cân bằng nguồn)** | AFDB có 29k cửa sổ vs MIMIC 4k — trộn thô thì AFDB "đè" 7:1. Gán trọng số sao cho mỗi dataset đóng góp tổng ảnh hưởng bằng nhau khi huấn luyện, như 2 lá phiếu ngang giá trị. |
| **Model card** | "Giấy khai sinh" của mô hình (`models/final/model_card.json`): input/output là gì, huấn luyện trên dữ liệu nào, điểm số qua từng bài kiểm định, và giới hạn sử dụng. Chuẩn thực hành tốt khi bàn giao mô hình. |
| **Pipeline (.pkl)** | Gói StandardScaler + mô hình thành 1 file duy nhất — backend chỉ cần `joblib.load` rồi `predict_proba`, không phải tự chuẩn hóa lại (tránh làm sai lệch so với lúc huấn luyện). |

## 6. Đọc kết quả

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **Accuracy** | Tỉ lệ đoán đúng tổng thể. Dễ hiểu nhưng dễ đánh lừa khi 2 lớp không cân bằng. |
| **Recall / Sensitivity (Độ nhạy)** | Trong số người **thực sự bị AFib**, mô hình bắt được bao nhiêu %? **Metric quan trọng nhất với sàng lọc bệnh** — bỏ sót bệnh nhân (FN) nguy hiểm hơn báo nhầm. |
| **Specificity (Độ đặc hiệu)** | Trong số người **khỏe mạnh**, mô hình xác nhận đúng bao nhiêu %? Thấp nghĩa là hay báo động nhầm. |
| **Precision** | Khi mô hình hô "AFib!", bao nhiêu % là đúng thật? |
| **F1-Score** | Trung bình điều hòa của Precision và Recall — cân bằng 2 phía. |
| **ROC-AUC** | Thước đo tổng quát 0.5–1.0: khả năng xếp hạng người bệnh cao hơn người khỏe. 0.5 = đoán mò, 1.0 = hoàn hảo. Không phụ thuộc ngưỡng cắt 0.5. |
| **FN / FP** | FN (False Negative) = người bệnh bị bỏ sót — lỗi nghiêm trọng nhất. FP (False Positive) = người khỏe bị báo nhầm — gây phiền, tốn chi phí kiểm tra lại. |
| **Confusion Matrix** | Bảng 2×2 đếm đủ 4 trường hợp: bắt đúng bệnh (TP), xác nhận đúng khỏe (TN), báo nhầm (FP), bỏ sót (FN). |
| **Window-level vs Subject-level** | *Mức cửa sổ*: chấm điểm từng đoạn đo 30 giây. *Mức bệnh nhân*: gộp trung bình mọi đoạn của một người thành 1 kết luận/người — **đây là con số nên dùng để báo cáo**, vì câu hỏi thực tế là "người này có bị AFib không?", không phải "đoạn 30 giây này thế nào?". |

---

*Xem sơ đồ pipeline tương tác tại [`docs/pipeline_v4.html`](pipeline_v4.html).*
