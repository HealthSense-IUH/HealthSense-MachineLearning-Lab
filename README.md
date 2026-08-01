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
- `data/raw/`: Chứa các file CSV gốc thu thập từ ESP32.
- `data/processed/`: Chứa các file CSV sau khi đã làm sạch và lọc nhiễu.
- `data/features/`: Chứa bảng đặc trưng HRV đã trích xuất (16 features).
- `notebooks/01_preprocessing.ipynb`: Tiền xử lý tín hiệu PPG (Bandpass Filter, Peak Detection).
- `notebooks/02_feature_engineering.ipynb`: Trích xuất 16 đặc trưng HRV (SDNN, RMSSD, LF/HF, ...).
- `notebooks/03_model_training.ipynb`: Huấn luyện và đánh giá mô hình Random Forest.
- `models/`: Chứa file mô hình đã huấn luyện (model.pkl).

### Cài đặt và Sử dụng
1. Tạo môi trường ảo và cài đặt thư viện:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Mở Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Chạy lần lượt các notebook từ `01_preprocessing` đến `03_model_training`.

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
- `data/raw/`: Raw CSV files collected from ESP32.
- `data/processed/`: Cleaned and filtered CSV files.
- `data/features/`: Extracted HRV feature tables (16 features).
- `notebooks/01_preprocessing.ipynb`: PPG signal preprocessing (Bandpass Filter, Peak Detection).
- `notebooks/02_feature_engineering.ipynb`: 16 HRV feature extraction (SDNN, RMSSD, LF/HF, ...).
- `notebooks/03_model_training.ipynb`: Random Forest model training and evaluation.
- `models/`: Trained model files (model.pkl).

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
3. Run the notebooks sequentially from `01_preprocessing` to `03_model_training`.

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
