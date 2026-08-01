# HealthSense ML

## Tiếng Việt

**HealthSense ML** là kho chứa mã nguồn dành riêng cho việc phân tích dữ liệu và huấn luyện mô hình Machine Learning cho dự án HealthSense.

### Chức năng chính
- Thu thập và lưu trữ dữ liệu PPG thô từ cảm biến MAX30102 qua ESP32.
- Tiền xử lý tín hiệu: loại bỏ nhiễu chuyển động (Motion Artifact) và Baseline Wander.
- Trích xuất 16 đặc trưng HRV (Heart Rate Variability) theo chuẩn Task Force 1996.
- Huấn luyện mô hình phân loại trạng thái sức khỏe bằng Random Forest.

### Công nghệ
- **Ngôn ngữ:** Python 3.12+
- **Xử lý tín hiệu:** SciPy (Butterworth Filter, Welch Periodogram)
- **Phân tích dữ liệu:** Pandas, NumPy, Matplotlib
- **Machine Learning:** Scikit-learn
- **Môi trường thí nghiệm:** Jupyter Notebook

### Cấu trúc dự án
- `data/raw/`: Chứa 3 tập dữ liệu thô (`mitbih/`, `mimic_perform/`, `ptbxl/`).
- `data/processed/`: Chứa các dữ liệu đã tiền xử lý theo từng nguồn (`mimic_processed/`).
- `data/features/`: Chứa các bảng đặc trưng HRV đã trích xuất (16 đặc trưng) và các tập dữ liệu đã chuẩn hóa (`*_scaled.csv`).
- `notebooks/`: Các Jupyter Notebook nghiên cứu phân tích & huấn luyện phân theo từng tập dữ liệu (`general/`, `mimic/`, `ptbxl/`, `mitbih/`).
- `models/`: Chứa các pipeline mô hình AI đã huấn luyện (`models/mimic_stacking_pipeline.pkl`, `models/ptbxl_stacking_pipeline.pkl`, v.v.).

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
3. Tiến trình nghiên cứu khoa học từ dữ liệu ➔ huấn luyện mô hình:
   - **`notebooks/general/` (Quy đổi chung):**
     1. `01_data_normalization_and_scaling.ipynb`: Chuẩn hóa dữ liệu Z-Score & Min-Max Scaling.
   - **`notebooks/mimic/` / `notebooks/ptbxl/` / `notebooks/mitbih/` (Phân tích & Train AI):**
     1. `01_eda_and_preprocessing.ipynb`: Phân tích thống kê y tế EDA & Tiền xử lý dữ liệu.
     2. `02_model_training_and_evaluation.ipynb`: Huấn luyện & Đánh giá các mô hình AI (Random Forest, LightGBM, SVM, TabPFN, Stacking Ensemble).

### Nguồn Dữ Liệu Kaggle (Datasets)
- **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)
- **MIT-BIH Database:** [mondejar/mitbih-database](https://www.kaggle.com/datasets/mondejar/mitbih-database)
- **PTB-XL Dataset:** [khyeh0719/ptb-xl-dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)

### Tài liệu tham khảo
- [1] Task Force of ESC/NASPE, "Heart rate variability: Standards of measurement, physiological interpretation and clinical use," *European Heart Journal*, vol. 17, pp. 354-381, 1996.
- [2] Schäfer, A. & Vagedes, J., "How accurate is pulse rate variability as an estimate of heart rate variability? A review on studies comparing photoplethysmographic technology with an electrocardiogram," *International Journal of Cardiology*, 2013.
- [3] Perez, M.V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation (Apple Heart Study)," *New England Journal of Medicine*, 2019.
- [4] Peralta, E. et al., "Assessing the Quality of Heart Rate Variability Estimated from Wrist and Finger PPG," *Sensors*, 2019.
- Xem danh sách đầy đủ (19 bài báo) tại file `REFERENCES.md`.

---

## English

**HealthSense ML** is the repository dedicated to data analysis and Machine Learning model training for the HealthSense project.

### Key Features
- Collects and stores raw PPG data from the MAX30102 sensor via ESP32.
- Signal preprocessing: removes Motion Artifacts and Baseline Wander.
- Extracts 16 HRV (Heart Rate Variability) features following the Task Force 1996 standard.
- Trains an Atrial Fibrillation (AFib) and health status classification model using Random Forest.

### Tech Stack
- **Language:** Python 3.12+
- **Signal Processing:** SciPy (Butterworth Filter, Welch Periodogram)
- **Data Analysis:** Pandas, NumPy, Matplotlib
- **Machine Learning:** Scikit-learn
- **Experimentation:** Jupyter Notebook

### Project Structure
- `data/raw/`: Contains 3 raw datasets (`mitbih/`, `mimic_perform/`, `ptbxl/`).
- `data/processed/`: Contains preprocessed datasets categorized by source (`mimic_processed/`).
- `data/features/`: Contains extracted 16 HRV feature tables and normalized datasets (`*_scaled.csv`).
- `notebooks/`: Jupyter Notebooks categorized by dataset (`general/`, `mimic/`, `ptbxl/`, `mitbih/`).
- `models/`: Contains trained AI model pipelines (`mimic_stacking_pipeline.pkl`, `ptbxl_stacking_pipeline.pkl`, etc.).

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
3. End-to-end research flow:
   - **`notebooks/general/` (General Utilities):**
     1. `01_data_normalization_and_scaling.ipynb`: Z-Score & Min-Max Scaling.
   - **`notebooks/mimic/` / `notebooks/ptbxl/` / `notebooks/mitbih/` (AI Pipeline):**
     1. `01_eda_and_preprocessing.ipynb`: Exploratory Data Analysis & Feature Processing.
     2. `02_model_training_and_evaluation.ipynb`: Model Training & Evaluation (Random Forest, LightGBM, SVM, TabPFN, Stacking Ensemble).

### Kaggle Datasets
- **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)
- **MIT-BIH Database:** [mondejar/mitbih-database](https://www.kaggle.com/datasets/mondejar/mitbih-database)
- **PTB-XL Dataset:** [khyeh0719/ptb-xl-dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)

### References
- [1] Task Force of ESC/NASPE, "Heart rate variability: Standards of measurement, physiological interpretation and clinical use," *European Heart Journal*, vol. 17, pp. 354-381, 1996.
- [2] Schäfer, A. & Vagedes, J., "How accurate is pulse rate variability as an estimate of heart rate variability? A review on studies comparing photoplethysmographic technology with an electrocardiogram," *International Journal of Cardiology*, 2013.
- [3] Perez, M.V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation (Apple Heart Study)," *New England Journal of Medicine*, 2019.
- [4] Peralta, E. et al., "Assessing the Quality of Heart Rate Variability Estimated from Wrist and Finger PPG," *Sensors*, 2019.
- See the full list (19 papers) in the `REFERENCES.md` file.
