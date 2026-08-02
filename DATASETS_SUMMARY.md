# Tóm Tắt Tập Dữ Liệu Và Kết Quả Đánh Giá Mô Hình HealthSense-ML

---

## 1. Tổng Quan Tập Dữ Liệu MIMIC-III

- **Tập MIMIC-III (4,083 mẫu - PERform AFib Dataset):**
  - Đo bằng ECG/PPG Monitor 1-2 chuyển đạo (Tương thích hoàn hảo với Đồng hồ thông minh / Vòng đeo tay HealthSense).
  - Trích xuất từ hơn 5.25 triệu điểm dữ liệu sóng thô trong `data/raw/mimic_perform/ppg_af_dataset.csv` theo kỹ thuật Cửa Sổ Trượt (Sliding Window 30s, step 10s).
  - Tín hiệu sạch, được gán nhãn chuyên biệt cho bài toán Rung Nhĩ (AFib vs Normal: 2,238 mẫu AFib vs 1,845 mẫu Bình thường).
  - **Mục tiêu AI:** Phân loại và Cảnh báo Rung Nhĩ (AFib Detection) liên tục 24/7.
  - **Kết quả AI xuất sắc (Soft Voting Ensemble):**
    - **Recall (Sensitivity):** **99.33%** (Chỉ bỏ sót 1 ca trong toàn bộ tập Test)
    - **Accuracy:** **98.53%**
    - **F1-Score:** **98.67%**
    - **ROC-AUC:** **0.9941**

- **Link Tải Dữ Liệu Kaggle:**
  - **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)

---

## 2. Danh Sách Thư Mục & Notebooks Trong Dự Án

- **Thư mục `notebooks/general/`:**
  - `00_raw_feature_extraction.ipynb`: Trích xuất 16 đặc trưng HRV từ dữ liệu sóng thô RAW (`data/raw/` ➔ `data/features/` - 4,083 mẫu).
  - `01_data_normalization_and_scaling.ipynb`: Chuẩn hóa Z-Score & Min-Max Scaling (`data/features/` ➔ `data/processed/`).
- **Thư mục `notebooks/mimic/`:**
  - `01_eda_and_preprocessing.ipynb`: Phân tích thống kê y tế EDA & Tiền xử lý dữ liệu MIMIC.
  - `02_model_training_and_evaluation.ipynb`: Huấn luyện & Đánh giá mô hình AI phát hiện Rung Nhĩ (Soft Voting, XGBoost, MLP).
