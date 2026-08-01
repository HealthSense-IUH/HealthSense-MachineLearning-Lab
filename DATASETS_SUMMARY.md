# Tóm Tắt Tập Dữ Liệu Và Kết Quả Đánh Giá Mô Hình HealthSense-ML

---

## 1. Tổng Quan Các Tập Dữ Liệu

- Tập MIMIC-III (1,202 mẫu):
  - Đo bằng ECG Monitor 1-2 chuyển đạo (tương tự Đồng hồ thông minh / Vòng đeo tay).
  - Bệnh nhân nặng nằm phòng Hồi sức Cấp cứu (ICU) và nằm cố định trên giường bệnh.
  - Tín hiệu rất sạch, đã được lọc nhiễu phần cứng từ monitor ICU.
  - Mục tiêu AI: Đánh giá bài toán Theo dõi nhịp tim liên tục 24/7 thời gian thực.
  - Kết quả AI: Accuracy ~97.10%, F1-Score 0.9714.

- Tập PTB-XL (3,028 mẫu - gấp 2.5 lần MIMIC):
  - Đo bằng Máy đo ECG 12 chuyển đạo tiêu chuẩn trong Bệnh viện / Phòng khám.
  - Bệnh nhân đa khoa thực tế (ngoại trú + nội trú), tín hiệu chứa nhiều nhiễu lâm sàng thực tế (nhiễu cơ, nhiễu chuyển động, nhiễu điện cực).
  - Mục tiêu AI: Đánh giá bài toán Tầm soát định kỳ và độ bền bỉ ngoài đời thực.
  - Kết quả AI: Accuracy ~91.58% (Stacking Ensemble), F1-Score 0.9168.

- Tập Gộp Combined (4,230 mẫu - MIMIC + PTB-XL):
  - Gộp cả 2 nguồn dữ liệu MIMIC và PTB-XL.
  - Mục tiêu AI: Huấn luyện bộ não AI nhận diện tốt trên mọi thiết bị và mọi môi trường.
  - Kết quả AI: Accuracy ~95.15% (TabPFN) / ~94.80% (Stacking Ensemble), F1-Score 0.9519.

---

## 2. Bảng So Sánh Chi Tiết MIMIC vs PTB-XL

- Nguồn dữ liệu:
  - MIMIC: Bệnh viện Beth Israel Deaconess / Harvard (Mỹ).
  - PTB-XL: Viện Đo lường Quốc gia PTB & Charité Berlin (Đức).

- Loại thiết bị và số chuyển đạo:
  - MIMIC: Monitor theo dõi tại giường bệnh, 1-2 chuyển đạo (Lead II, V1).
  - PTB-XL: Máy đo ECG tiêu chuẩn phòng khám, 12 chuyển đạo đầy đủ (I, II, III, aVR, aVL, aVF, V1-V6).

- Thời lượng ghi:
  - MIMIC: Liên tục 24/7 theo dạng chuỗi thời gian dài.
  - PTB-XL: Đo định kỳ 10 giây cho mỗi lần khám.

- Ý nghĩa thực tiễn:
  - MIMIC: Kiểm chứng tính năng Theo dõi liên tục 24/7 (Continuous Monitoring).
  - PTB-XL: Kiểm chứng tính năng Tầm soát định kỳ và Thiết bị y tế di động (Spot-check & Portable Devices).

---

## 3. Kết Quả So Sánh Mô Hình Trên Từng Tập Dữ Liệu

- Tập MIMIC-III (1,202 mẫu):
  - Accuracy: 97.10%
  - Precision: 95.20%
  - Recall: 99.17%
  - F1-Score: 97.14%

- Tập PTB-XL (3,028 mẫu):
  - Stacking Ensemble (LGB + ET + SVM + TabPFN): Accuracy 91.58%, F1-Score 0.9168
  - TabPFN: Accuracy 91.42%, F1-Score 0.9153
  - KNN: Accuracy 91.42%, F1-Score 0.9148
  - SVM: Accuracy 91.25%, F1-Score 0.9147

- Tập Gộp Combined (4,230 mẫu):
  - TabPFN: Accuracy 95.15%, F1-Score 0.9519
  - Stacking Ensemble: Accuracy 94.80%, F1-Score 0.9484
  - SVM: Accuracy 93.50%, F1-Score 0.9371

---

## 4. Danh Sách Thiết Lập Trong Dự Án

- File cấu hình môi trường: .env (chứa TABPFN_TOKEN).
- Thư mục notebooks:
  - notebooks/mimic/ (03_model_evaluation.ipynb, 04_stacking_ensemble.ipynb)
  - notebooks/ptbxl/ (01_ptbxl_evaluation.ipynb, 02_ptbxl_stacking_ensemble.ipynb)
  - notebooks/combined/ (01_combined_evaluation.ipynb, 02_combined_stacking_ensemble.ipynb)
- Thư mục models:
  - models/mimic_stacking_pipeline.pkl
  - models/ptbxl_stacking_pipeline.pkl
  - models/combined_stacking_pipeline.pkl
