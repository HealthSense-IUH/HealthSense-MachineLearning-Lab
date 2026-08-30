# HealthSense ML

> 📚 **Tài liệu:** mở [`docs/index.html`](docs/index.html) — trang chủ docs liên kết sơ đồ kiến trúc thành phần ([`components.html`](docs/components.html)), sơ đồ pipeline ML ([`pipeline_v4.html`](docs/pipeline_v4.html), tương tác: guided views, pan/zoom, dark mode, export), giải thích thuật ngữ và kết quả.

## Tiếng Việt

**HealthSense ML** là kho chứa mã nguồn dành riêng cho việc phân tích dữ liệu tín hiệu sinh lý (ECG/PPG), trích xuất đặc trưng biến thiên nhịp tim (HRV) và huấn luyện các mô hình Machine Learning phát hiện Rung Nhĩ (AFib) cho dự án HealthSense dựa trên tập dữ liệu y tế chuẩn **MIMIC-III (PERform AFib Dataset)**.

### Chức năng chính
- Xử lý sóng PPG thô **theo từng bệnh nhân** từ 2 dataset y tế công khai: MIMIC PERform AF (35 bệnh nhân PPG) và MIT-BIH AFDB (23 bệnh nhân ECG, AF kịch phát).
- Tiền xử lý tín hiệu: lọc Butterworth bandpass 0.5–8 Hz, phát hiện nhịp, trích chuỗi NN.
- Trích xuất **16 đặc trưng HRV** chuẩn Task Force 1996 (time / frequency / nonlinear) bằng cửa sổ trượt 30s, mỗi hàng gắn `record_id`.
- Đánh giá **không data leakage**: LOSO theo bệnh nhân + cross-dataset MIMIC ↔ AFDB. Kết quả v4 (mức bệnh nhân): **Recall 100% (0 ca AFib bị bỏ sót), Accuracy 94.3%, ROC-AUC 0.93** — con số phản ánh bệnh nhân chưa từng thấy.

### Công nghệ
- **Ngôn ngữ:** Python 3.12+
- **Xử lý tín hiệu:** SciPy (Butterworth sosfiltfilt, find_peaks, Welch)
- **Dữ liệu:** Pandas, NumPy, kagglehub (tải MIMIC), wfdb (annotation PhysioNet AFDB)
- **Machine Learning:** Scikit-learn (Pipeline, LeaveOneGroupOut, GroupKFold), XGBoost
- **Biểu đồ:** Matplotlib, Seaborn

### Cấu trúc dự án (v4)
- `src/healthsense_ml/`: **Package Python trung tâm** — toàn bộ logic pipeline nằm ở đây:
  - `config.py`: Đường dẫn, hằng số tín hiệu, danh sách đặc trưng, tham số huấn luyện.
  - `data_loading.py`: Nạp dữ liệu MIMIC PERform **theo từng bệnh nhân** (`record_id`), tự tải từ Kaggle.
  - `signal_processing.py`: Lọc Butterworth bandpass 0.5–8 Hz, phát hiện nhịp, trích chuỗi NN.
  - `hrv_features.py`: 16 đặc trưng HRV chuẩn Task Force 1996 (time/frequency/nonlinear).
  - `feature_extraction.py`: Cửa sổ trượt 30s/10s ➔ bảng đặc trưng **có `record_id`**.
  - `training.py`: Benchmark LOSO chống data leakage (chi tiết bên dưới).
  - `evaluation.py`: Metrics 2 cấp (cửa sổ & bệnh nhân) + biểu đồ.
  - `afdb.py`: Dataset thứ hai MIT-BIH AFDB — dựng chuỗi NN từ annotation QRS (không cần tải sóng thô).
- `scripts/`:
  - `run_v4_extraction.py`: Bước 1 — raw ➔ `data/features/mimic_features_v4.csv`.
  - `run_v4_benchmark.py`: Bước 2 — LOSO benchmark ➔ `models/benchmark_v4/`.
  - `run_cross_dataset.py`: Bước 3 — cross-dataset MIMIC ↔ AFDB ➔ `models/cross_dataset/`.
  - `run_final_model.py`: Bước 4 — gộp 60 bệnh nhân (pooled LOSO + cân bằng nguồn) ➔ `models/final/` (.pkl triển khai).
  - `legacy/`: Script v3 cũ (bị leakage, chỉ tham khảo).
- `data/raw/mimic_perform/`: Dữ liệu thô theo từng bệnh nhân (19 AF + 16 non-AF, PPG 125 Hz).
- `data/features/`: Bảng đặc trưng HRV có `record_id` (`mimic_features_v4.csv`, `afdb_features_v4.csv`).
- `models/`: Kết quả hiện hành (`benchmark_v4/`, `cross_dataset/`, `final/` — model triển khai .pkl + model card).
- `docs/`: Sơ đồ kiến trúc tương tác (`pipeline_v4.html`) + giải thích thuật ngữ (`GIAI_THICH_THUAT_NGU.md`).
- `legacy/`: TOÀN BỘ thí nghiệm v1–v3 (notebooks, features cũ, kết quả benchmark v3) — chỉ để tham khảo, kết quả bị data leakage (xem bên dưới); có README riêng bên trong.

### ⚠️ Vì sao có v4? (Data Leakage trong v1–v3)
Các phiên bản trước có 2 lỗi phương pháp khiến kết quả 98–99% bị thổi phồng:
1. **Subject leakage:** đặc trưng không mang `record_id`, dữ liệu được chia random theo cửa sổ — các cửa sổ của cùng một bệnh nhân nằm ở cả train lẫn test, mô hình chỉ cần "nhận mặt" bệnh nhân là đạt điểm cao.
2. **Preprocessing leakage:** Scaler và ngưỡng lọc outlier IQR được fit trên toàn bộ dữ liệu (gồm cả test) trước khi chia.

v4 sửa tận gốc: **Leave-One-Subject-Out** theo bệnh nhân, tiền xử lý fit train-only trong từng fold, tuning nested (GroupKFold), loại nhóm đặc trưng LF không đủ tin cậy trên cửa sổ 30s, và báo cáo metric ở **mức bệnh nhân** — con số phản ánh đúng khả năng nhận diện bệnh nhân chưa từng thấy.

### Cài đặt và Sử dụng
1. Tạo môi trường ảo và cài đặt thư viện:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Chạy pipeline 4 bước (dữ liệu tự tải nếu chưa có — MIMIC từ Kaggle ~100MB, AFDB chỉ tải annotation từ PhysioNet ~vài MB):
   ```bash
   python scripts/run_v4_extraction.py
   python scripts/run_v4_benchmark.py
   python scripts/run_cross_dataset.py
   python scripts/run_final_model.py
   ```
   - Thêm cờ `--full16` cho benchmark nếu muốn dùng đủ 16 đặc trưng (mặc định loại nhóm LF).
3. Kết quả:
   - `models/benchmark_v4/` — LOSO trên MIMIC: `benchmark_results_v4.csv` (metrics 2 cấp), `loso_predictions.csv`, confusion matrix / ROC / biểu đồ xác suất theo bệnh nhân.
   - `models/cross_dataset/` — kiểm định chéo MIMIC (PPG) ↔ MIT-BIH AFDB (ECG): train trên dataset này, test trên dataset kia — bằng chứng tổng quát hóa mạnh nhất.
   - `models/final/` — **mô hình triển khai**: pooled LOSO 60 bệnh nhân (cân bằng nguồn) + `healthsense_afib_pipeline.pkl` (kèm scaler, nạp thẳng vào `HealthSense-AI-Service`) + `model_card.json`.

### Nguồn Dữ Liệu (Datasets)
- **MIMIC PERform AF** (Kaggle): [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset) — tự tải bằng `kagglehub`.
- **MIT-BIH AFDB** (PhysioNet): [physionet.org/content/afdb](https://physionet.org/content/afdb/) — chỉ tải annotation QRS + rhythm bằng `wfdb`.

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
- Processes raw PPG **per patient** from 2 public clinical datasets: MIMIC PERform AF (35 PPG patients) and MIT-BIH AFDB (23 ECG patients, paroxysmal AF).
- Signal preprocessing: Butterworth bandpass 0.5–8 Hz, beat detection, NN-interval extraction.
- Extracts **16 Task Force 1996 HRV features** (time / frequency / nonlinear) via 30s sliding windows, every row tagged with `record_id`.
- **Leakage-free** evaluation: patient-wise LOSO + MIMIC ↔ AFDB cross-dataset validation. v4 results (subject level): **Recall 100% (zero missed AFib patients), Accuracy 94.3%, ROC-AUC 0.93** on unseen patients.

### Tech Stack
- **Language:** Python 3.12+
- **Signal Processing & Scaling:** SciPy, Scikit-learn
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Neural Networks (MLP)
- **Experimentation Environment:** Jupyter Notebook

### Project Structure (v4)
- `src/healthsense_ml/`: **Core Python package** — all pipeline logic lives here:
  - `config.py`: Paths, signal constants, feature lists, training parameters.
  - `data_loading.py`: Loads MIMIC PERform **per patient** (`record_id`), auto-downloads from Kaggle.
  - `signal_processing.py`: Butterworth bandpass 0.5–8 Hz, beat detection, NN-interval extraction.
  - `hrv_features.py`: 16 Task Force 1996 HRV features (time / frequency / nonlinear).
  - `feature_extraction.py`: 30s/10s sliding window ➔ feature table **with `record_id`**.
  - `training.py`: Leakage-free LOSO benchmark (see below).
  - `evaluation.py`: Two-level metrics (window & subject) + plots.
  - `afdb.py`: Second dataset (MIT-BIH AFDB) — NN series from QRS annotations, no raw waveform download needed.
- `scripts/`: `run_v4_extraction.py` (Step 1), `run_v4_benchmark.py` (Step 2), `run_cross_dataset.py` (Step 3), `run_final_model.py` (Step 4 — pooled deployment model), `legacy/` (old v3 scripts).
- `data/raw/mimic_perform/`: Per-patient raw data (19 AF + 16 non-AF, 125 Hz PPG).
- `data/features/`: HRV feature tables with `record_id` (`mimic_features_v4.csv`, `afdb_features_v4.csv`).
- `models/`: Current results (`benchmark_v4/`, `cross_dataset/`).
- `docs/`: Interactive architecture diagram (`pipeline_v4.html`) + plain-language glossary (`GIAI_THICH_THUAT_NGU.md`).
- `legacy/`: ALL v1–v3 experiments (notebooks, old features, v3 benchmark results) — reference only, results suffer from data leakage (see note below); has its own README.

### ⚠️ Why v4? (Data Leakage in v1–v3)
Earlier versions had two methodological flaws that inflated the reported 98–99% results:
1. **Subject leakage:** features carried no `record_id` and windows were split randomly — windows from the same patient appeared in both train and test.
2. **Preprocessing leakage:** scalers and IQR outlier thresholds were fit on the full dataset (including test) before splitting.

v4 fixes both at the root: **Leave-One-Subject-Out** splitting by patient, train-only preprocessing inside each fold, nested hyperparameter tuning (GroupKFold), removal of LF features (unreliable on 30s windows), and **subject-level** reporting — numbers that reflect performance on unseen patients.

### Installation and Usage
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the 2-step pipeline (data auto-downloads from Kaggle if missing, ~100MB):
   ```bash
   python scripts/run_v4_extraction.py
   python scripts/run_v4_benchmark.py
   ```
   - Add `--full16` to the benchmark to use all 16 features (LF group excluded by default).
3. Outputs land in `models/benchmark_v4/`.

### Kaggle Datasets
- **MIMIC PERform AF Dataset:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset)
