# HealthSense ML

## Tiếng Việt

**HealthSense ML** là kho chứa mã nguồn dành riêng cho việc phân tích dữ liệu tín hiệu sinh lý (ECG/PPG), trích xuất đặc trưng biến thiên nhịp tim (HRV) và huấn luyện các mô hình Machine Learning phát hiện Rung Nhĩ (AFib) và Bất thường Tim mạch cho dự án HealthSense.

### Chức năng chính
- Thu thập và xử lý dữ liệu sóng thô ECG/PPG từ cảm biến y tế và dữ liệu thực tế (`data/raw/`).
- Tiền xử lý tín hiệu: loại bỏ nhiễu chuyển động (Motion Artifact), lọc dải thông Butterworth và Baseline Wander.
- Trích xuất **16 đặc trưng biến thiên nhịp tim HRV** y tế chuẩn Task Force 1996 (nội suy 4Hz, FFT phổ tần số LF/HF, Poincaré metrics).
- Huấn luyện & Đánh giá mô hình AI phát hiện **Rung Nhĩ (AFib)** với hiệu năng cao (**Recall > 99.3%**, **Accuracy > 98.5%**).

### Công nghệ
- **Ngôn ngữ:** Python 3.12+
- **Xử lý tín hiệu & Chuẩn hóa:** SciPy (Butterworth Filter, Welch Periodogram, Find Peaks), Scikit-learn (StandardScaler, MinMaxScaler)
- **Phân tích & Quản lý dữ liệu:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning & Ensemble:** Scikit-learn, XGBoost, LightGBM, PyTorch / MLP Neural Network
- **Môi trường thí nghiệm:** Jupyter Notebook

### Cấu trúc dự án
- `data/raw/`: Chứa các tập dữ liệu thô (`mimic_perform/ppg_af_dataset.csv`, `ptbxl/`).
- `data/features/`: Chứa các bảng đặc trưng HRV (16 đặc trưng) đã trích xuất từ dữ liệu thô (`mimic_features.csv`, `ptbxl_features.csv`).
- `data/processed/`: Chứa các tập dữ liệu đã chuẩn hóa biên độ (`*_zscore_scaled.csv`, `*_minmax_scaled.csv`) sẵn sàng cho huấn luyện AI.
- `notebooks/`:
  - `general/`: 
    - `00_raw_feature_extraction.ipynb`: Trích xuất 16 đặc trưng HRV từ sóng thô RAW (`data/raw/` ➔ `data/features/`).
    - `01_data_normalization_and_scaling.ipynb`: Chuẩn hóa dữ liệu Z-Score & Min-Max Scaling (`data/features/` ➔ `data/processed/`).
  - `mimic/`: Pipeline huấn luyện & Đánh giá chuyên sâu phát hiện Rung Nhĩ AFib từ tập MIMIC-III (Recall ~99.3%).
  - `ptbxl/`: Pipeline huấn luyện & Đánh giá tầm soát bệnh lý tim mạch từ tập PTB-XL.
- `models/`: Chứa các pipeline mô hình AI đã huấn luyện (`models/mimic/`, `models/ptbxl/`).

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
3. Tiến trình nghiên cứu khoa học từ dữ liệu thô ➔ huấn luyện mô hình:
   - **Xử lý dữ liệu thô & Chuẩn hóa (`notebooks/general/`):**
     1. `00_raw_feature_extraction.ipynb`: Trích xuất 16 đặc trưng HRV từ tín hiệu sóng thô RAW.
     2. `01_data_normalization_and_scaling.ipynb`: Chuẩn hóa dữ liệu Z-Score & Min-Max Scaling.
   - **Huấn luyện & Đánh giá AI (`notebooks/mimic/` & `notebooks/ptbxl/`):**
     1. `01_eda_and_preprocessing.ipynb`: Phân tích thống kê y tế EDA & Tiền xử lý dữ liệu.
     2. `02_model_training_and_evaluation.ipynb`: Huấn luyện & Đánh giá các mô hình AI (XGBoost, Logistic Regression, MLP Neural Network, Soft Voting, Stacking Ensemble).

### Nguồn Dữ Liệu Kaggle (Datasets)
- **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)
- **PTB-XL Dataset:** [khyeh0719/ptb-xl-dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)

### Tài liệu tham khảo
- [1] Task Force of ESC/NASPE, "Heart rate variability: Standards of measurement, physiological interpretation and clinical use," *European Heart Journal*, vol. 17, pp. 354-381, 1996.
- [2] Schäfer, A. & Vagedes, J., "How accurate is pulse rate variability as an estimate of heart rate variability? A review on studies comparing photoplethysmographic technology with an electrocardiogram," *International Journal of Cardiology*, 2013.
- [3] Perez, M.V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation (Apple Heart Study)," *New England Journal of Medicine*, 2019.
- [4] Peralta, E. et al., "Assessing the Quality of Heart Rate Variability Estimated from Wrist and Finger PPG," *Sensors*, 2019.
- Xem danh sách đầy đủ tại file `REFERENCES.md`.

---

## English

**HealthSense ML** is the dedicated repository for physiological signal processing (ECG/PPG), Heart Rate Variability (HRV) feature extraction, and Machine Learning model training for Atrial Fibrillation (AFib) detection and cardiovascular anomaly screening in the HealthSense project.

### Key Features
- Processes raw ECG/PPG waveform data from medical sensors and clinical datasets (`data/raw/`).
- Signal Preprocessing: Bandpass filtering, Motion Artifact removal, and Baseline Wander correction.
- Feature Extraction: Computes 16 medical-grade HRV features following the 1996 Task Force standards (4Hz spline interpolation, FFT spectrum for LF/HF, Poincaré metrics).
- High-Performance AFib Detection Training (**Recall > 99.3%**, **Accuracy > 98.5%**).

### Tech Stack
- **Language:** Python 3.12+
- **Signal Processing & Scaling:** SciPy, Scikit-learn
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Neural Networks (MLP)
- **Experimentation Environment:** Jupyter Notebook

### Project Structure
- `data/raw/`: Raw waveform datasets (`mimic_perform/ppg_af_dataset.csv`, `ptbxl/`).
- `data/features/`: Extracted 16-HRV feature tables (`mimic_features.csv`, `ptbxl_features.csv`).
- `data/processed/`: Scaled datasets (`*_zscore_scaled.csv`, `*_minmax_scaled.csv`) ready for ML training.
- `notebooks/`:
  - `general/`: 
    - `00_raw_feature_extraction.ipynb`: Raw signal feature extraction (`data/raw/` ➔ `data/features/`).
    - `01_data_normalization_and_scaling.ipynb`: Z-Score & Min-Max Scaling (`data/features/` ➔ `data/processed/`).
  - `mimic/`: Pipeline for AFib Detection using MIMIC-III (Recall ~99.3%).
  - `ptbxl/`: Pipeline for Cardiovascular Pathology Screening using PTB-XL.
- `models/`: Serialized trained AI models (`models/mimic/`, `models/ptbxl/`).

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
3. End-to-End Pipeline Workflow:
   - **Raw Signal Processing & Scaling (`notebooks/general/`):**
     1. `00_raw_feature_extraction.ipynb`: Extract 16 HRV metrics from raw ECG/PPG waveforms.
     2. `01_data_normalization_and_scaling.ipynb`: Apply Z-Score and Min-Max scaling.
   - **AI Training & Evaluation (`notebooks/mimic/` & `notebooks/ptbxl/`):**
     1. `01_eda_and_preprocessing.ipynb`: Exploratory Data Analysis & Imputation.
     2. `02_model_training_and_evaluation.ipynb`: Train and evaluate ML models (XGBoost, Logistic Regression, MLP, Soft Voting, Stacking Ensemble).

### Kaggle Datasets
- **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)
- **PTB-XL Dataset:** [khyeh0719/ptb-xl-dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)
