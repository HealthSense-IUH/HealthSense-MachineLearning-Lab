# HealthSense ML

> 📚 **Tài liệu:** mở [`docs/index.html`](docs/index.html) — TOÀN BỘ docs trong 1 file: tổng quan kết quả, sơ đồ kiến trúc hệ thống, sơ đồ pipeline ML (tương tác: guided views, pan/zoom, export), và giải thích thuật ngữ dễ hiểu. Tái tạo bằng `scripts/build_docs.py` từ 2 spec `docs/*.json`.

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
  - `run_beat_validation.py`: Kiểm chứng dò nhịp PPG bằng ECG đồng bộ ➔ `models/beat_validation/` (xem mục Kết quả chi tiết).
  - `build_docs.py`: Tái tạo `docs/index.html` từ 2 spec sơ đồ.
- `data/raw/mimic_perform/`: Dữ liệu thô theo từng bệnh nhân (19 AF + 16 non-AF, PPG 125 Hz).
- `data/features/`: Bảng đặc trưng HRV có `record_id` (`mimic_features_v4.csv`, `afdb_features_v4.csv`).
- `models/`: Kết quả hiện hành (`benchmark_v4/`, `cross_dataset/`, `final/` — model triển khai .pkl + model card).
- `docs/`: `index.html` — toàn bộ tài liệu trong 1 file (kết quả, 2 sơ đồ tương tác, giải thích thuật ngữ); kèm 2 spec `.json` để tái tạo sơ đồ; `HealthSense_ML_Slides.pptx` — bộ slide giải thích toàn bộ phần ML.

### 📚 Bảo tàng phiên bản (`src/v1` … `src/v4` + `src/report`)
Bốn phiên bản pipeline được **tái dựng thành code chạy được**, đặt cạnh nhau trên cùng một bộ dữ liệu và chấm bằng cùng một thước đo — để thấy rõ phần "tiến bộ" nào là thật, phần nào do data leakage tạo ra.

- `src/v1/` … `src/v4/`: mỗi thư mục là một phiên bản, gồm `pipeline.py` (chạy được: `python src/vN/pipeline.py`) và `README.md` dạng "thẻ phiên bản" ghi rõ cấu hình, chỗ đúng, chỗ sai.
- `src/vlab/`: tiện ích dùng chung — đọc tín hiệu thô theo kênh, cửa sổ trượt tham số hóa, metrics 2 cấp, biểu đồ, và **`honest.py`** (chấm cùng một bảng theo 2 cách: ngẫu nhiên vs LOSO).
- `src/report/`: **6 notebook tiếng Việt đã chạy sẵn, nhúng đủ kết quả + biểu đồ** (5 báo cáo phiên bản + báo cáo sản phẩm). Bắt đầu từ [`00_tong_quan.ipynb`](src/report/00_tong_quan.ipynb).
- `models/vN/results.json`: kết quả từng phiên bản (kèm `original_claim` — con số bản gốc từng công bố, để đối chiếu).

Kết quả cốt lõi: điểm **công bố** tăng dần 95.9% → 97.4% → 98.7%, nhưng khi chấm bằng LOSO thì **cả bốn phiên bản đều quanh 92%**. Toàn bộ "tiến bộ" nằm trong cái thước đo hỏng. Chi tiết cách tái dựng và giới hạn của nó: [`src/report/README.md`](src/report/README.md).

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

## 📊 Dữ liệu & Kết quả chi tiết

### 1. Tổng Quan Tập Dữ Liệu MIMIC PERform AF

- Dữ liệu công khai từ MIMIC-III: **19 bệnh nhân AFib + 16 bệnh nhân Normal**, mỗi người 20 phút PPG @ 125 Hz, lưu **file riêng từng bệnh nhân** tại `data/raw/mimic_perform/{af,non-af}/`.
- Trích xuất v4: cửa sổ trượt 30s / bước 10s ➔ **4.130 cửa sổ** (2.242 AFib / 1.888 Normal), mỗi hàng mang 16 đặc trưng HRV + **`record_id`** (danh tính bệnh nhân).
- **Mục tiêu AI:** Sàng lọc Rung Nhĩ (AFib Detection) cho vòng đeo HealthSense.
- **Link Kaggle:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset) (tự tải bằng `kagglehub`, không cần token).

---

### 2. Kết Quả Benchmark v4 (LOSO — không data leakage)

Đánh giá bằng **Leave-One-Subject-Out** (35 folds, mỗi fold giữ trọn 1 bệnh nhân làm test), tiền xử lý fit train-only, tuning nested GroupKFold(3), 13 đặc trưng (loại nhóm LF).

**Mức bệnh nhân** (con số báo cáo chính — trung bình xác suất các cửa sổ của mỗi người):

| Model | Accuracy | Recall | Specificity | F1 | ROC-AUC | FN | FP |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 94.29% | **100%** | 87.50% | 95.00% | 0.8750 | **0** | 2 |
| Random Forest | 94.29% | **100%** | 87.50% | 95.00% | **0.9309** | **0** | 2 |
| XGBoost | 94.29% | **100%** | 87.50% | 95.00% | 0.9013 | **0** | 2 |

- **Không bỏ sót bệnh nhân AFib nào** (FN = 0 trên cả 3 mô hình).
- 2 ca báo nhầm đều là cùng 2 bệnh nhân Normal (`non_af_012`, `non_af_014`) — nhịp của họ bất thường thật sự ở mức tín hiệu, đáng xem lại thủ công.

**Mức cửa sổ 30s** (từng lần đo đơn lẻ):

| Model | Accuracy | Recall | Specificity | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 92.28% | 98.22% | 85.22% | 0.8707 |
| Random Forest | 92.13% | 97.46% | 85.81% | 0.9398 |
| XGBoost | 91.82% | 96.30% | 86.49% | 0.9129 |

> ⚠️ **Về kết quả 98–99% của v1–v3:** các phiên bản cũ bị subject leakage (chia random theo cửa sổ, không theo bệnh nhân) và preprocessing leakage (scaler/IQR fit trên cả test) nên con số bị thổi phồng. Kết quả v4 thấp hơn nhưng **thật** — phản ánh khả năng nhận diện bệnh nhân chưa từng thấy. Chi tiết: README mục "Vì sao có v4?".

Kết quả đầy đủ: `models/benchmark_v4/` (metrics CSV, dự đoán từng cửa sổ, confusion matrix, ROC, biểu đồ xác suất theo bệnh nhân).

---

### 3. Kết Quả Cross-Dataset (MIMIC ↔ MIT-BIH AFDB)

Bài kiểm tra tổng quát hóa khắc nghiệt nhất: train trên dataset này, test **toàn bộ** dataset kia — khác bệnh viện, khác loại cảm biến (PPG kẹp ngón vs ECG), khác quần thể. AFDB: 25 bệnh nhân, 28.903 cửa sổ (11.190 AFib / 17.713 Normal), nhiều ca AF kịch phát.

| Hướng | Model tốt nhất | Accuracy | Recall | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| Train MIMIC (PPG) → Test AFDB (ECG) | XGBoost | 94.56% | 96.99% | 93.03% | **0.9870** |
| Train AFDB (ECG) → Test MIMIC (PPG) | Random Forest | 91.86% | 97.37% | 85.33% | **0.9757** |

**Ý nghĩa:** mô hình giữ được AUC ~0.98 khi nhảy sang dataset hoàn toàn lạ theo cả 2 chiều — bằng chứng mạnh rằng nó học được **dấu hiệu sinh lý của Rung Nhĩ** (nhịp bất thường trong chuỗi NN) chứ không học thuộc đặc điểm bệnh nhân hay thiết bị. Kết quả đầy đủ: `models/cross_dataset/cross_dataset_results.csv`.

---

### 4. Mô Hình Triển Khai Cuối Cùng (Pooled — 60 bệnh nhân)

Gộp MIMIC (35) + AFDB (25) với **cân bằng nguồn bằng sample weight** (mỗi dataset đóng góp tổng trọng số bằng nhau, tránh AFDB 29k cửa sổ đè MIMIC 4k). Đánh giá bằng pooled LOSO 60 folds:

| Model | Accuracy (pooled) | Recall | Specificity | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** 🏆 | **95.89%** | 97.16% | 95.01% | **0.9883** |
| Random Forest | 95.25% | 96.55% | 94.36% | 0.9876 |
| Logistic Regression | 94.78% | 97.33% | 93.04% | 0.9700 |

- Mức bệnh nhân trên nhánh MIMIC: **Recall vẫn 100%** (0 bệnh nhân AFib bị bỏ sót).
- **File triển khai:** `models/final/healthsense_afib_pipeline.pkl` (XGBoost + StandardScaler đóng gói chung, nạp bằng `joblib.load`) + `models/final/model_card.json` (đặc tả input/output, 13 đặc trưng, giới hạn sử dụng).
- Input: cửa sổ 30s ➔ chuỗi NN ➔ 13 đặc trưng HRV (thứ tự trong model card). Output: `predict_proba[:, 1]` = P(AFib).
- ⚠️ Giới hạn: chưa kiểm định trên PPG cổ tay MAX30102 và dữ liệu ngoài bệnh viện; không phải thiết bị chẩn đoán y tế.

---

### 5. Kiểm Chứng Dò Nhịp PPG Bằng ECG Đồng Bộ (`scripts/run_beat_validation.py`)

Dùng R-peak trên cột ECG (ghi song song trong MIMIC PERform) làm đáp án chuẩn chấm điểm bộ dò nhịp PPG (khớp từng nhịp ±150 ms sau khi bù PTT):

| Nhóm | F1 dò nhịp (median) | HR MAE (median) |
|---|---|---|
| Normal (16 người) | **0.991** | 1.1 bpm |
| AFib (19 người) | 0.855 | 4.7 bpm |

- Dò nhịp trên người nhịp thường gần như hoàn hảo; nhóm AFib khó hơn (biên độ mạch thay đổi từng nhịp). Lưu ý: với AFib, PTT dao động theo nhịp nên phép khớp ±150 ms **đánh giá thấp** chất lượng thật — HR MAE nhỏ cho thấy số nhịp đếm được vẫn đúng.
- **Thẩm định 2 ca "Normal" nghi nhãn sai bằng chính ECG** (không qua PPG):
  - `non_af_012`: RMSSD(ECG) = 233 ms, pNN50 = 41% — **nhịp loạn thật sự ngay trên ECG**, còn loạn hơn median nhóm AFib (170 ms). Nhãn "Normal" của record này đáng nghi ngờ; mô hình báo AFib là có cơ sở sinh lý.
  - `non_af_014`: RMSSD(ECG) = 31 ms, pNN50 = 2% — **tim hoàn toàn bình thường trên ECG**, nhưng PPG của record này chất lượng rất kém (F1 dò nhịp 0.196, HR MAE 24 bpm). Mô hình báo AFib vì chuỗi NN rác do tín hiệu xấu → đây là false positive do **chất lượng tín hiệu**, không phải do mô hình sai logic.
- **Bài học triển khai:** cần thêm **Signal Quality Index (SQI)** — cửa sổ nào dò nhịp không đạt chất lượng thì từ chối phân loại thay vì đoán bừa. Đây là nâng cấp quan trọng nhất trước khi chạy trên MAX30102.

**So kết quả đo PPG vs ECG cùng thời điểm** (1.400 cửa sổ 30s, toàn MIMIC; file `models/beat_validation/ppg_vs_ecg_windows.csv`):

| Chỉ số | PPG vs ECG | Nhận xét |
|---|---|---|
| Nhịp tim (HR) | MAE 3.5–4 bpm, r = 0.91–0.94; 82% cửa sổ nhóm Normal lệch ≤3 bpm | HR đo bằng PPG tin cậy được |
| RMSSD | MAE ~50 ms, r = 0.69–0.74 | PPG "phóng đại" độ biến thiên so với ECG — hiện tượng PRV ≠ HRV kinh điển (Schäfer & Vagedes 2013, tài liệu [2]); mô hình không bị ảnh hưởng vì học trực tiếp trên đặc trưng PPG |

Kết quả từng bệnh nhân: `models/beat_validation/beat_validation.csv`.

---

### 6. Cấu Trúc Code

- **Package `src/healthsense_ml/`**: config, data_loading, signal_processing, hrv_features, feature_extraction, training, evaluation, afdb.
- **Scripts v4**: `scripts/run_v4_extraction.py` ➔ `scripts/run_v4_benchmark.py` ➔ `scripts/run_cross_dataset.py`.
- **Tài liệu**: `docs/index.html` — toàn bộ trong 1 file (kết quả, sơ đồ kiến trúc hệ thống, sơ đồ pipeline, giải thích thuật ngữ).
- **Bảo tàng phiên bản (v1–v4)**: `src/v1` … `src/v4` (mỗi phiên bản một `pipeline.py` chạy được) + `src/vlab` (tiện ích dùng chung) + `src/report` (5 notebook báo cáo). Notebook thí nghiệm gốc của v1–v3 đã được gỡ khỏi repo — xem lịch sử git nếu cần đối chiếu.

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
- `scripts/`: `run_v4_extraction.py` (Step 1), `run_v4_benchmark.py` (Step 2), `run_cross_dataset.py` (Step 3), `run_final_model.py` (Step 4 — pooled deployment model).
- `data/raw/mimic_perform/`: Per-patient raw data (19 AF + 16 non-AF, 125 Hz PPG).
- `data/features/`: HRV feature tables with `record_id` (`mimic_features_v4.csv`, `afdb_features_v4.csv`).
- `models/`: Current results (`benchmark_v4/`, `cross_dataset/`).
- `docs/`: `index.html` — all documentation in one file (results, 2 interactive diagrams, plain-language glossary); plus 2 `.json` specs to regenerate the diagrams.
- `src/v1` … `src/v4`, `src/vlab`, `src/report`: **version museum** — all four pipeline generations rebuilt as runnable code, scored side by side on the same data with the same metric. Start at [`src/report/00_tong_quan.ipynb`](src/report/00_tong_quan.ipynb). The original v1–v3 experiment notebooks have been removed from the repo; see git history if you need them.

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
