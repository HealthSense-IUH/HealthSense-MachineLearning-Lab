# Tóm Tắt Tập Dữ Liệu Và Kết Quả Đánh Giá Mô Hình HealthSense-ML

---

## 1. Tổng Quan Các Tập Dữ Liệu

- **Tập MIMIC-III (1,360 mẫu - PERform AFib Dataset):**
  - Đo bằng ECG/PPG Monitor 1-2 chuyển đạo (Tương thích hoàn hảo với Đồng hồ thông minh / Vòng đeo tay HealthSense).
  - Tín hiệu sạch, được gán nhãn chuyên biệt cho bài toán Rung Nhĩ (AFib vs Normal).
  - **Mục tiêu AI:** Phân loại và Cảnh báo Rung Nhĩ (AFib Detection) liên tục 24/7.
  - **Kết quả AI xuất sắc (Soft Voting Ensemble):**
    - **Recall (Sensitivity):** **99.33%** (Chỉ bỏ sót 1 ca trong toàn bộ tập Test)
    - **Accuracy:** **98.53%**
    - **F1-Score:** **98.67%**
    - **ROC-AUC:** **0.9941**

- **Tập PTB-XL (3,028 mẫu):**
  - Đo bằng Máy đo ECG 12 chuyển đạo tiêu chuẩn trong Bệnh viện / Phòng khám.
  - Bệnh nhân đa khoa thực tế với nhiều dạng nhiễu lâm sàng.
  - **Mục tiêu AI:** Tầm soát bệnh lý tim mạch lâm sàng (Nhồi máu cơ tim, Biến đổi ST/T, Rối loạn dẫn truyền).
  - **Kết quả AI (Balanced Weights):** Accuracy ~79.0%, Recall ~63.5%, F1-Score ~50.6%.

- **Link Tải Dữ Liệu Kaggle (Kaggle Datasets):**
  - **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)
  - **PTB-XL Dataset:** [khyeh0719/ptb-xl-dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)

---

## 2. Bảng So Sánh Chi Tiết MIMIC vs PTB-XL

- **Nguồn dữ liệu:**
  - MIMIC: Bệnh viện Beth Israel Deaconess / Harvard (Mỹ).
  - PTB-XL: Viện Đo lường Quốc gia PTB & Charité Berlin (Đức).

- **Loại thiết bị và số chuyển đạo:**
  - MIMIC: Monitor tại giường bệnh, 1-2 chuyển đạo (Lead II, V1 / PPG).
  - PTB-XL: Máy đo ECG tiêu chuẩn phòng khám, 12 chuyển đạo đầy đủ.

- **Ý nghĩa thực tiễn:**
  - MIMIC: Kiểm chứng tính năng Theo dõi Rung Nhĩ liên tục 24/7 (Continuous Monitoring).
  - PTB-XL: Kiểm chứng tính năng Tầm soát định kỳ và Thiết bị y tế lâm sàng.

---

## 3. Danh Sách Thư Mục & Notebooks Trong Dự Án

- **Thư mục `notebooks/general/`:**
  - `00_raw_feature_extraction.ipynb`: Trích xuất 16 đặc trưng HRV từ dữ liệu sóng thô RAW (`data/raw/` ➔ `data/features/`).
  - `01_data_normalization_and_scaling.ipynb`: Chuẩn hóa Z-Score & Min-Max Scaling (`data/features/` ➔ `data/processed/`).
- **Thư mục `notebooks/mimic/`:**
  - `01_eda_and_preprocessing.ipynb`: Phân tích thống kê y tế EDA & Tiền xử lý dữ liệu MIMIC.
  - `02_model_training_and_evaluation.ipynb`: Huấn luyện & Đánh giá mô hình AI phát hiện Rung Nhĩ (Soft Voting, XGBoost, MLP).
- **Thư mục `notebooks/ptbxl/`:**
  - `01_eda_and_preprocessing.ipynb`: Phân tích thống kê y tế EDA & Tiền xử lý dữ liệu PTB-XL.
  - `02_model_training_and_evaluation.ipynb`: Huấn luyện & Đánh giá mô hình AI tầm soát bệnh lý PTB-XL.
