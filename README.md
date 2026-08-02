# HealthSense ML

## Tiếng Việt

**HealthSense ML** là kho chứa mã nguồn dành riêng cho việc phân tích dữ liệu tín hiệu sinh lý (ECG/PPG), trích xuất đặc trưng biến thiên nhịp tim (HRV) và huấn luyện các mô hình Machine Learning phát hiện Rung Nhĩ (AFib) cho dự án HealthSense dựa trên tập dữ liệu y tế chuẩn **MIMIC-III (PERform AFib Dataset)**.

### Chức năng chính
- Thu thập và xử lý dữ liệu sóng thô ECG/PPG từ cảm biến y tế và dữ liệu thực tế (`data/raw/mimic_perform/ppg_af_dataset.csv` - 5.25 triệu điểm sóng thô).
- Tiền xử lý tín hiệu: loại bỏ nhiễu chuyển động (Motion Artifact), lọc dải thông Butterworth và Baseline Wander.
- Trích xuất **16 đặc trưng biến thiên nhịp tim HRV** y tế chuẩn Task Force 1996 theo 4 quy mô dung lượng bằng kỹ thuật Cửa Sổ Trượt (Sliding Window: 1,360, 4,083, 8,165, 16,358 mẫu).
- Huấn luyện & Đánh giá mô hình AI phát hiện **Rung Nhĩ (AFib)** với hiệu năng xuất sắc (**Recall > 99.3%**, **Accuracy > 98.5%**, **ROC-AUC = 0.9941**).

### Công nghệ
- **Ngôn ngữ:** Python 3.12+
- **Xử lý tín hiệu & Chuẩn hóa:** SciPy (Butterworth Filter, Welch Periodogram, Find Peaks), Scikit-learn (StandardScaler, MinMaxScaler)
- **Phân tích & Quản lý dữ liệu:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning & Ensemble:** Scikit-learn, XGBoost, LightGBM, PyTorch / MLP Neural Network
- **Môi trường thí nghiệm:** Jupyter Notebook

### Cấu trúc dự án
- `data/raw/`: Chứa dữ liệu thô MIMIC-III (`mimic_perform/ppg_af_dataset.csv`).
- `data/features/`: Chứa các bảng đặc trưng HRV đa quy mô đã trích xuất từ dữ liệu thô (`mimic_features_1360.csv`, `mimic_features_4083.csv`, `mimic_features_8165.csv`, `mimic_features_16358.csv`).
- `data/processed/`: Chứa các tập dữ liệu đã chuẩn hóa biên độ Z-Score & Min-Max Scaling cho cả 4 quy mô sẵn sàng cho huấn luyện AI.
- `notebooks/`:
  - `general/`: 
    - `00_raw_feature_extraction.ipynb`: Trích xuất 16 đặc trưng HRV đa quy mô từ sóng thô RAW (`data/raw/` ➔ `data/features/`).
    - `01_data_normalization_and_scaling.ipynb`: Phân tích EDA, Imputation và Chuẩn hóa dữ liệu Z-Score & Min-Max Scaling (`data/features/` ➔ `data/processed/`).
  - `mimic/`:
    - `02_model_training_and_evaluation.ipynb`: Pipeline huấn luyện & Đánh giá Y tế chuyên sâu phát hiện Rung Nhĩ AFib từ tập MIMIC-III (Hỗ trợ tùy chọn quy mô 1360, 4083, 8165, 16358).
- `models/`: Chứa các pipeline mô hình AI đã huấn luyện (`models/mimic/`).

### Cài đặt và Sử dụng
1. Tạo môi trường ảo và cài đặt thư viện:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Khởi động Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Tiến trình nghiên cứu khoa học tinh gọn 3 bước từ dữ liệu thô ➔ huấn luyện mô hình:
   - **Bước 0: Trích xuất đặc trưng thô (`notebooks/general/00_raw_feature_extraction.ipynb`)**: Trích xuất 16 đặc trưng HRV từ tín hiệu sóng thô RAW cho cả 4 quy mô (1360, 4083, 8165, 16358 mẫu).
   - **Bước 1: EDA & Chuẩn hóa dữ liệu (`notebooks/general/01_data_normalization_and_scaling.ipynb`)**: Thống kê phân bố, Imputation & Chuẩn hóa Z-Score / Min-Max Scaling cho cả 4 quy mô.
   - **Bước 2: Huấn luyện & Đánh giá AI (`notebooks/mimic/02_model_training_and_evaluation.ipynb`)**: Huấn luyện & Đánh giá các mô hình AI (XGBoost, Logistic Regression, MLP Neural Network, Soft Voting, Stacking Ensemble) với bộ chọn `DATASET_SCALE`.

### Nguồn Dữ Liệu Kaggle (Datasets)
- **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)

### Tài liệu tham khảo
- [1] Task Force of ESC/NASPE, "Heart rate variability: Standards of measurement, physiological interpretation and clinical use," *European Heart Journal*, vol. 17, pp. 354-381, 1996.
- [2] Schäfer, A. & Vagedes, J., "How accurate is pulse rate variability as an estimate of heart rate variability? A review on studies comparing photoplethysmographic technology with an electrocardiogram," *International Journal of Cardiology*, 2013.
- [3] Perez, M.V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation (Apple Heart Study)," *New England Journal of Medicine*, 2019.
- [4] Peralta, E. et al., "Assessing the Quality of Heart Rate Variability Estimated from Wrist and Finger PPG," *Sensors*, 2019.
- Xem danh sách đầy đủ tại file `REFERENCES.md`.

---

## English

**HealthSense ML** is the dedicated repository for physiological signal processing (ECG/PPG), Heart Rate Variability (HRV) feature extraction, and Machine Learning model training for Atrial Fibrillation (AFib) detection in the HealthSense project, centered around the **MIMIC-III (PERform AFib Dataset)**.

### Key Features
- Processes raw ECG/PPG waveform data from medical sensors and clinical datasets (`data/raw/mimic_perform/ppg_af_dataset.csv` - 5.25M raw points).
- Signal Preprocessing: Bandpass filtering, Motion Artifact removal, and Baseline Wander correction.
- Multi-Scale Feature Extraction: Computes 16 medical-grade HRV features across 4 scales (1,360, 4,083, 8,165, and 16,358 samples).
- High-Performance AFib Detection Training (**Recall > 99.3%**, **Accuracy > 98.5%**, **ROC-AUC = 0.9941**).

### Tech Stack
- **Language:** Python 3.12+
- **Signal Processing & Scaling:** SciPy, Scikit-learn
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Neural Networks (MLP)
- **Experimentation Environment:** Jupyter Notebook

### Project Structure
- `data/raw/`: Raw MIMIC-III waveform dataset (`mimic_perform/ppg_af_dataset.csv`).
- `data/features/`: Extracted multi-scale 16-HRV feature tables (`mimic_features_1360.csv`, `mimic_features_4083.csv`, `mimic_features_8165.csv`, `mimic_features_16358.csv`).
- `data/processed/`: Scaled multi-scale datasets (`mimic_zscore_*.csv`, `mimic_minmax_*.csv`) ready for ML training.
- `notebooks/`:
  - `general/`: 
    - `00_raw_feature_extraction.ipynb`: Multi-scale raw signal feature extraction (`data/raw/` ➔ `data/features/`).
    - `01_data_normalization_and_scaling.ipynb`: Exploratory Data Analysis & Multi-scale Z-Score / Min-Max Scaling (`data/features/` ➔ `data/processed/`).
  - `mimic/`:
    - `02_model_training_and_evaluation.ipynb`: Deep pipeline for AFib Detection using MIMIC-III (Supports multi-scale dataset selection via `DATASET_SCALE`).
- `models/`: Serialized trained AI models (`models/mimic/`).

### Installation and Usage
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Streamlined 3-Step Research Pipeline:
   - **Step 0: Raw Signal Extraction (`notebooks/general/00_raw_feature_extraction.ipynb`)**: Extract 16 HRV metrics for 4 scales (1360, 4083, 8165, 16358 samples).
   - **Step 1: EDA & Multi-Scale Normalization (`notebooks/general/01_data_normalization_and_scaling.ipynb`)**: EDA, Imputation & Z-Score / Min-Max Scaling for all scales.
   - **Step 2: AI Training & Evaluation (`notebooks/mimic/02_model_training_and_evaluation.ipynb`)**: Train and evaluate ML models (XGBoost, Logistic Regression, MLP, Soft Voting, Stacking Ensemble) with `DATASET_SCALE` parameter.

### Kaggle Datasets
- **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)
