# Tóm Tắt Tập Dữ Liệu Và Kết Quả Đánh Giá Mô Hình HealthSense-ML

---

## 1. Tổng Quan Tập Dữ Liệu MIMIC PERform AF

- Dữ liệu công khai từ MIMIC-III: **19 bệnh nhân AFib + 16 bệnh nhân Normal**, mỗi người 20 phút PPG @ 125 Hz, lưu **file riêng từng bệnh nhân** tại `data/raw/mimic_perform/{af,non-af}/`.
- Trích xuất v4: cửa sổ trượt 30s / bước 10s ➔ **4.130 cửa sổ** (2.242 AFib / 1.888 Normal), mỗi hàng mang 16 đặc trưng HRV + **`record_id`** (danh tính bệnh nhân).
- **Mục tiêu AI:** Sàng lọc Rung Nhĩ (AFib Detection) cho vòng đeo HealthSense.
- **Link Kaggle:** [raditya0/mimic-perform-iii-af-and-non-af-dataset](https://www.kaggle.com/datasets/raditya0/mimic-perform-iii-af-and-non-af-dataset) (tự tải bằng `kagglehub`, không cần token).

---

## 2. Kết Quả Benchmark v4 (LOSO — không data leakage)

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

## 3. Kết Quả Cross-Dataset (MIMIC ↔ MIT-BIH AFDB)

Bài kiểm tra tổng quát hóa khắc nghiệt nhất: train trên dataset này, test **toàn bộ** dataset kia — khác bệnh viện, khác loại cảm biến (PPG kẹp ngón vs ECG), khác quần thể. AFDB: 25 bệnh nhân, 28.903 cửa sổ (11.190 AFib / 17.713 Normal), nhiều ca AF kịch phát.

| Hướng | Model tốt nhất | Accuracy | Recall | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| Train MIMIC (PPG) → Test AFDB (ECG) | XGBoost | 94.56% | 96.99% | 93.03% | **0.9870** |
| Train AFDB (ECG) → Test MIMIC (PPG) | Random Forest | 91.86% | 97.37% | 85.33% | **0.9757** |

**Ý nghĩa:** mô hình giữ được AUC ~0.98 khi nhảy sang dataset hoàn toàn lạ theo cả 2 chiều — bằng chứng mạnh rằng nó học được **dấu hiệu sinh lý của Rung Nhĩ** (nhịp bất thường trong chuỗi NN) chứ không học thuộc đặc điểm bệnh nhân hay thiết bị. Kết quả đầy đủ: `models/cross_dataset/cross_dataset_results.csv`.

---

## 4. Mô Hình Triển Khai Cuối Cùng (Pooled — 60 bệnh nhân)

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

## 5. Cấu Trúc Code

- **Package `src/healthsense_ml/`**: config, data_loading, signal_processing, hrv_features, feature_extraction, training, evaluation, afdb.
- **Scripts v4**: `scripts/run_v4_extraction.py` ➔ `scripts/run_v4_benchmark.py` ➔ `scripts/run_cross_dataset.py`.
- **Tài liệu**: `docs/pipeline_v4.html` (sơ đồ kiến trúc tương tác), `docs/GIAI_THICH_THUAT_NGU.md` (giải thích thuật ngữ dễ hiểu).
- **Legacy (v1–v3)**: tất cả gom về `legacy/` (notebooks, features cũ, benchmark v3) + `scripts/legacy/` — giữ để tham khảo, không dùng cho báo cáo. Xem `legacy/README.md`.
