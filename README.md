# HealthSense Machine Learning Lab

> **HealthSense Machine Learning** là kho nghiên cứu dành cho xử lý tín hiệu sinh lý, trích xuất đặc trưng và phát triển mô hình hỗ trợ **tầm soát rung nhĩ (Atrial Fibrillation, AF)** từ **quang thể tích ký (Photoplethysmography, PPG)**.  
> Trạng thái nghiên cứu hiện tại: **V6_RESEARCH_FREEZE_2026_09**.

> **Phạm vi lâm sàng:** hệ thống hỗ trợ **sàng lọc và cảnh báo dấu hiệu nghi ngờ rung nhĩ từ PPG**. Kết quả PPG không thay thế xác nhận bằng **điện tâm đồ (Electrocardiography, ECG)** và không phải chẩn đoán y khoa [8].

---

## Tiếng Việt

## 1. Trạng thái nghiên cứu hiện tại

Quá trình phát triển đã đi từ V4 → V5 → V6. Phiên bản hiện tại được đóng băng ở **V6 Research Freeze** để tránh tiếp tục tối ưu trên các benchmark đã được sử dụng nhiều lần và chuẩn bị cho external validation trên dữ liệu chưa từng tham gia training hoặc tuning.

### Mô hình nghiên cứu hiện tại

- **Model:** `HealthSense AF V6C Stacking`
- **Base models:** Extra Trees, Random Forest, XGBoost
- **Meta-model:** Logistic Regression
- **Input:** 14 đặc trưng trích xuất từ PPG
- **Primary output:** `meta_probability`
- **Calibration:** Platt scaling chỉ dùng cho phân tích, chưa được đưa vào runtime chính
- **Temporal alert candidate:** `v6_locked_global_rule_v2`
- **Current status:** dừng tuning trên các benchmark hiện tại; chờ untouched external validation

---

## 2. Bài toán nghiên cứu

HealthSense tập trung vào phát hiện dấu hiệu nghi ngờ AF từ PPG đeo tay.

```text
PPG waveform
    ↓
Signal preprocessing
    ↓
Beat detection
    ↓
NN interval series
    ↓
Heart Rate Variability features
    ↓
AF probability
    ↓
Temporal persistence rule
    ↓
Suspected-AF screening alert
```

Các thuật ngữ chính:

- **Heart Rate Variability (HRV):** biến thiên khoảng thời gian giữa các nhịp tim.
- **NN interval:** khoảng thời gian giữa các nhịp hợp lệ liên tiếp.
- **Signal Quality Index (SQI):** chỉ số chất lượng tín hiệu.
- **Premature Atrial Contraction (PAC):** ngoại tâm thu nhĩ.
- **Premature Ventricular Contraction (PVC):** ngoại tâm thu thất.

PAC và PVC là nhóm **hard negative** quan trọng vì có thể tạo kiểu nhịp bất thường gây nhầm với AF [5].

---

## 3. Hành trình V4 → V5 → V6

### V4 — sửa data leakage và xây baseline theo bệnh nhân

V4 sửa hai lỗi chính của các phiên bản đầu:

1. **Subject leakage:** các cửa sổ của cùng một bệnh nhân từng xuất hiện ở cả train và test.
2. **Preprocessing leakage:** scaler hoặc outlier filtering từng được fit trước khi chia dữ liệu.

V4 chuyển sang đánh giá theo bệnh nhân bằng **Leave-One-Subject-Out (LOSO)** và chỉ fit preprocessing trên training fold.

Dataset chính là **Medical Information Mart for Intensive Care (MIMIC) PERform AF** [1]:

- 35 đối tượng
- 19 AF
- 16 non-AF
- 4.130 cửa sổ PPG
- cửa sổ 30 giây, bước trượt 10 giây

V4 là baseline phương pháp luận, không còn là model nghiên cứu hiện tại.

---

### V5 — frozen single-domain model và external validation

V5 đóng băng một Random Forest sử dụng 14 đặc trưng, trong đó bổ sung đặc trưng tự tương quan PPG (`PPG_AC`).

Model:

```text
models/mimic/healthsense_af_v5_rf_ac_frozen.pkl
```

Feature contract:

```text
HR_mean
Mean_NN
SDNN
RMSSD
NN50
pNN50
CV
HF
Total_Power
HF_norm
SD1
SD2
SampEn
PPG_AC
```

V5 được đánh giá trên nhiều nguồn dữ liệu:

- DeepBeat [2]
- PulseWatch [3]
- Liu 2022 [4]
- targeted PAC/PVC challenge [5]

Kết quả cho thấy AF so với nhịp xoang tương đối dễ hơn, trong khi PAC/PVC là hard-negative quan trọng. Đây là một trong các động lực chính để chuyển sang huấn luyện multidomain ở V6.

Notebook:

[HealthSense V5 Visual Review](notebooks/HealthSense_V5_Visual_Review.ipynb)

---

### V6 — multidomain, hard-negative aware và stacking

V6 mở rộng training sang nhiều domain và đưa hard negatives vào quá trình phát triển.

Các model từng được benchmark:

- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Histogram Gradient Boosting

V6C giữ ba base learner:

```text
Extra Trees
Random Forest
XGBoost
    ↓
Logistic Regression meta-model
    ↓
meta_probability
```

### Reused subject-held-out benchmark

Các test split dưới đây tách theo subject nhưng đã được inspect nhiều lần trong quá trình nghiên cứu. Vì vậy chúng được gọi là **reused subject-held-out benchmark**, không phải pristine external validation.

| Dataset | Area Under Receiver Operating Characteristic Curve (AUROC) | Area Under Precision-Recall Curve (PR-AUC) | Brier score |
|---|---:|---:|---:|
| DeepBeat | 0.9904 | 0.9890 | 0.0583 |
| PulseWatch | 0.9985 | 0.9890 | 0.0095 |
| Liu | 0.9866 | 0.9780 | 0.0322 |

Notebook:

[HealthSense V6 Research Freeze Visual Review](notebooks/HealthSense_V6_Research_Freeze_Visual_Review.ipynb)

---

## 4. Locked scientific validation protocol

Sau các thử nghiệm temporal V6D–V6H, phần đánh giá cuối được reset sang protocol khóa tại:

```text
experiments/08_v6_locked_protocol/
```

Protocol gồm 18 bước, bao phủ:

- generation of locked predictions
- subject split và temporal-order audit
- verified stream construction
- development-only temporal rule selection
- test evaluation
- calibration diagnostics
- subject-level confidence intervals
- **Leave-One-Domain-Out (LODO)** domain-generalization analysis
- `PPG_AC` ablation
- PAC/PVC hard-negative challenge
- DeepBeat failure-mode analysis
- final research freeze

### Global Rule V2

Global Rule V2 gồm hai nhánh:

- **Base rule:** probability ≥ 0.80, model disagreement ≤ 0.05, ít nhất 5 cửa sổ liên tiếp
- **Rescue rule:** probability ≥ 0.90, model disagreement ≤ 0.05, ít nhất 2 cửa sổ liên tiếp

`ensemble_std` là độ lệch chuẩn giữa probability của các base model và chỉ được hiểu là **model disagreement**, không phải calibrated uncertainty.

Global Rule V2 hiện là exploratory candidate cho untouched external validation.

---

## 5. Leave-One-Domain-Out domain-generalization analysis

**Leave-One-Domain-Out (LODO)** giữ toàn bộ một dataset làm held-out domain và train trên các domain còn lại.

| Held-out domain | AUROC | PR-AUC | Brier score |
|---|---:|---:|---:|
| DeepBeat | 0.8384 | 0.8289 | 0.1898 |
| PulseWatch | 0.9877 | 0.9417 | 0.0560 |
| Liu | 0.9797 | 0.9599 | 0.0500 |
| MIMIC PERform | 0.9856 | 0.9867 | 0.0530 |

DeepBeat cho thấy domain shift rõ rệt và là failure mode quan trọng cần theo dõi.

LODO trong repo là **retrospective domain-generalization analysis**, không phải pristine external validation.

---

## 6. PPG_AC, calibration và hard negatives

### PPG_AC ablation

Matched development ablation:

| Variant | Macro AUROC | Macro PR-AUC | Macro Brier |
|---|---:|---:|---:|
| 13 features, không `PPG_AC` | 0.9154 | 0.6175 | 0.0729 |
| 14 features, có `PPG_AC` | 0.9263 | 0.6693 | 0.0607 |

Kết quả tổng thể hỗ trợ giữ `PPG_AC`, nhưng không đồng nghĩa đặc trưng này giải quyết toàn bộ domain shift.

### Calibration

Platt calibration được đánh giá bằng development-only grouped cross-validation nhưng không cải thiện nhất quán trên mọi domain.

Vì vậy:

- `meta_probability` thô vẫn là probability chính
- Platt scaling chỉ dùng cho analysis
- runtime không tự động áp dụng calibration

### PAC/PVC challenge

PAC/PVC được theo dõi riêng vì đây là nguồn false positive quan trọng ở cấp cửa sổ. Các nghiên cứu smartwatch PPG trước đây cũng chỉ ra rằng PAC/PVC có thể làm giảm độ đặc hiệu của AF screening nếu không được xử lý riêng [5].

---

## 7. Datasets

### Dataset đã dùng trong quá trình phát triển

| Dataset | Vai trò trong HealthSense | Tín hiệu / nhãn chính | Link truy cập |
|---|---|---|---|
| **MIMIC PERform AF** [1] | V4 baseline, V5/V6 reference domain | PPG + ECG; AF / non-AF | [Dataset documentation](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/) · [Zenodo](https://zenodo.org/records/6973963) |
| **DeepBeat** [2] | V5 external evaluation, V6 multidomain | wearable PPG; AF + signal quality | [Dataset: Synapse `syn21985690`](https://www.synapse.org/Synapse:syn21985690) · [Code](https://github.com/AshleyLab/deepbeat) |
| **PulseWatch** [3] | V5/V6 external and hard-negative evaluation | smartwatch PPG + reference ECG; AF, normal sinus rhythm, PAC/PVC | [Dataset: Synapse `syn23565056`](https://www.synapse.org/Synapse:syn23565056) |
| **Liu 2022** [4] | V5/V6 multiclass external evaluation | PPG; sinus rhythm, PVC, PAC, ventricular tachycardia, supraventricular tachycardia, AF | [Dataset + code](https://github.com/zdzdliu/PPGArrhythmiaDetection) |
| **Selected MIMIC-III / PAC-PVC challenge data** [5] | targeted PAC/PVC hard-negative challenge | fingertip PPG; AF, PAC/PVC, normal sinus rhythm | [UConn resource / Synapse access](https://biosignal.uconn.edu/resources/) |

### Dataset dành cho untouched external validation

| Dataset | Trạng thái trong HealthSense | Link |
|---|---|---|
| **MIMIC-III-Ext-PPG v1.1.0** [6] | Chưa dùng cho training, tuning hoặc current benchmark | [PhysioNet](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/) · [Official code/documentation](https://github.com/AI4HealthUOL/MIMIC-III-Ext-PPG_dataset) |
| **TriggersAF** [7] | Chưa dùng; dự kiến làm external validation khi có quyền truy cập phù hợp | [Project page](https://biomedicine.ktu.edu/projects/wearable-technology-for-personalized-identification-and-management-of-paroxysmal-atrial-fibrillation-triggers-triggersaf/) |

> **MIMIC-III-Ext-PPG** là credentialed-access dataset trên PhysioNet. Quyền truy cập yêu cầu credentialing, Data Use Agreement và khóa đào tạo nghiên cứu dữ liệu người theo yêu cầu của PhysioNet [6].

---

## 8. Cấu trúc repository

```text
HealthSense-MachineLearning-Lab/
├── src/
│   ├── healthsense_ml/           # Signal processing và HRV feature extraction
│   ├── v1/ ... v4/              # Historical pipeline museum
│   └── report/                   # Historical notebooks
│
├── experiments/
│   ├── 00_baseline_reproduction/
│   ├── 01_label_audit/
│   ├── 02_beat_detector/
│   ├── 03_sqi/
│   ├── 04_feature_model/
│   ├── 05_calibration_alert/
│   ├── 06_external_validation/   # V5 external validation
│   ├── 07_v6_multidomain/       # V6 model development
│   └── 08_v6_locked_protocol/    # Locked validation protocol
│
├── notebooks/
│   ├── HealthSense_V5_Visual_Review.ipynb
│   └── HealthSense_V6_Research_Freeze_Visual_Review.ipynb
│
├── models/
│   ├── mimic/
│   └── multidomain/
│
├── results/
├── docs/
└── data/
```

---

## 9. Cài đặt và chạy

### Môi trường nghiên cứu hiện tại

```bash
cd HealthSense-MachineLearning-Lab
conda activate healthsense-af-v5
jupyter lab
```

Hoặc tạo môi trường mới:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mở notebook:

```bash
jupyter lab notebooks/
```

Một số bước như LODO, ablation và bootstrap có thể tốn thời gian. Không nên tuning tiếp trên các benchmark hiện tại nếu mục tiêu là giữ tính độc lập cho bước validation tiếp theo.

---

## 10. Giới hạn nghiên cứu

- PPG screening không thay thế ECG confirmation [8].
- Các reused subject-held-out benchmarks đã được inspect nhiều lần.
- LODO là retrospective analysis.
- DeepBeat cho thấy domain shift đáng kể.
- PAC/PVC vẫn là hard-negative quan trọng.
- Temporal rule cần validation trên dataset hoàn toàn chưa dùng.
- Signal Quality Index chưa phải end-to-end gate cuối cùng của production pipeline.
- V6C hiện phù hợp research/server-side runtime hơn edge deployment.
- Chưa có clinical validation trên wearable HealthSense thực tế.

---

## 11. Hướng tiếp theo

1. Giữ nguyên V6 freeze, không tuning tiếp trên benchmark hiện tại.
2. Untouched external validation trên MIMIC-III-Ext-PPG khi được cấp quyền.
3. Untouched external validation trên TriggersAF khi có quyền truy cập phù hợp.
4. Đánh giá Signal Quality Index end-to-end.
5. Chuẩn hóa temporal stride và session semantics.
6. Báo cáo subject-level sensitivity, specificity và false alerts per hour.
7. Nếu phát triển V7, sử dụng dữ liệu hoặc hypothesis mới và yêu cầu một untouched validation set mới.

---

# English

## Current research status

HealthSense Machine Learning develops **Photoplethysmography (PPG)-based screening for suspected Atrial Fibrillation (AF)**.

Current frozen research candidate:

**V6_RESEARCH_FREEZE_2026_09**

Current model:

**HealthSense AF V6C Stacking**

```text
Extra Trees
Random Forest
XGBoost
    ↓
Logistic Regression meta-model
    ↓
raw meta_probability
```

The model uses 14 PPG-derived features, including `PPG_AC`.

The existing subject-held-out test datasets have been repeatedly inspected and are therefore reported as **reused subject-held-out benchmarks**, not pristine external validation.

**Leave-One-Domain-Out (LODO)** evaluation is treated as retrospective domain-generalization analysis.

Future untouched external validation is reserved for:

- MIMIC-III-Ext-PPG
- TriggersAF

The system is intended for suspected-AF screening and early warning. It does not replace **Electrocardiography (ECG)-based confirmation** [8].

## Main results

### V6C reused subject-held-out benchmark

| Dataset | Area Under Receiver Operating Characteristic Curve (AUROC) | Area Under Precision-Recall Curve (PR-AUC) | Brier score |
|---|---:|---:|---:|
| DeepBeat | 0.9904 | 0.9890 | 0.0583 |
| PulseWatch | 0.9985 | 0.9890 | 0.0095 |
| Liu | 0.9866 | 0.9780 | 0.0322 |

### Retrospective Leave-One-Domain-Out analysis

| Held-out domain | AUROC | PR-AUC | Brier score |
|---|---:|---:|---:|
| DeepBeat | 0.8384 | 0.8289 | 0.1898 |
| PulseWatch | 0.9877 | 0.9417 | 0.0560 |
| Liu | 0.9797 | 0.9599 | 0.0500 |
| MIMIC PERform | 0.9856 | 0.9867 | 0.0530 |

DeepBeat exposes a substantial domain shift and remains an important failure mode.

## Dataset access

| Dataset | Use in this repository | Access |
|---|---|---|
| MIMIC PERform AF [1] | baseline and reference domain | [Documentation](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/) |
| DeepBeat [2] | external and multidomain evaluation | [Synapse `syn21985690`](https://www.synapse.org/Synapse:syn21985690) |
| PulseWatch [3] | smartwatch PPG and hard-negative evaluation | [Synapse `syn23565056`](https://www.synapse.org/Synapse:syn23565056) |
| Liu 2022 [4] | multiclass arrhythmia external evaluation | [GitHub dataset](https://github.com/zdzdliu/PPGArrhythmiaDetection) |
| Selected MIMIC-III PAC/PVC data [5] | targeted hard-negative challenge | [UConn resources](https://biosignal.uconn.edu/resources/) |
| MIMIC-III-Ext-PPG v1.1.0 [6] | future untouched external validation | [PhysioNet](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/) |
| TriggersAF [7] | future untouched external validation | [Project page](https://biomedicine.ktu.edu/projects/wearable-technology-for-personalized-identification-and-management-of-paroxysmal-atrial-fibrillation-triggers-triggersaf/) |

## Reproducibility

Core research code:

```text
experiments/06_external_validation/
experiments/07_v6_multidomain/
experiments/08_v6_locked_protocol/
```

Large generated feature matrices, row-level predictions and V6 model binaries are intentionally not stored in normal Git history.

Main notebooks:

- [V5 Visual Review](notebooks/HealthSense_V5_Visual_Review.ipynb)
- [V6 Research Freeze Visual Review](notebooks/HealthSense_V6_Research_Freeze_Visual_Review.ipynb)

---

## References

1. **Charlton PH, et al.** Detecting beats in the photoplethysmogram: benchmarking open-source algorithms. *Physiological Measurement*. 2022. DOI: [10.1088/1361-6579/ac826d](https://doi.org/10.1088/1361-6579/ac826d). Dataset: [MIMIC PERform AF](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/).

2. **Torres-Soto J, Ashley EA.** Multi-task deep learning for cardiac rhythm detection in wearable devices. *npj Digital Medicine*. 2020;3:116. DOI: [10.1038/s41746-020-00320-4](https://doi.org/10.1038/s41746-020-00320-4). Dataset/code: [DeepBeat](https://github.com/AshleyLab/deepbeat), Synapse `syn21985690`.

3. **Han D, Moon J, Mercado Díaz LR, et al.** Multiclass Arrhythmia Classification Using Multimodal Smartwatch Photoplethysmography Signals Collected in Real-Life Settings. *IEEE Transactions on Biomedical Engineering*. 2026;73(4):1679-1693. DOI: [10.1109/TBME.2025.3613471](https://doi.org/10.1109/TBME.2025.3613471). Dataset: [PulseWatch, Synapse `syn23565056`](https://www.synapse.org/Synapse:syn23565056).

4. **Liu Z, Zhou B, Jiang Z, et al.** Multiclass Arrhythmia Detection and Classification From Photoplethysmography Signals Using a Deep Convolutional Neural Network. *Journal of the American Heart Association*. 2022;11(7):e023555. DOI: [10.1161/JAHA.121.023555](https://doi.org/10.1161/JAHA.121.023555). Dataset/code: [PPGArrhythmiaDetection](https://github.com/zdzdliu/PPGArrhythmiaDetection).

5. **Han D, Bashar SK, Mohagheghian F, et al.** Premature Atrial and Ventricular Contraction Detection Using Photoplethysmographic Data from a Smartwatch. *Sensors*. 2020;20(19):5683. DOI: [10.3390/s20195683](https://doi.org/10.3390/s20195683). Related UConn/Synapse resources: [Biosignal Processing and Wearable Device Lab](https://biosignal.uconn.edu/resources/).

6. **Moulaeifard M, Charlton PH, Strodthoff N.** MIMIC-III-Ext-PPG: A PPG Benchmark Dataset for Cardiorespiratory Analysis, version 1.1.0. *PhysioNet*. 2026. DOI: [10.13026/r6k1-xt76](https://doi.org/10.13026/r6k1-xt76). Dataset: [PhysioNet](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/). See also: Moulaeifard M, Kutscher M, Aston PJ, et al. *Scientific Data*. 2026;13:668. DOI: [10.1038/s41597-026-07335-8](https://doi.org/10.1038/s41597-026-07335-8).

7. **TriggersAF project.** Wearable technology for personalized identification and management of paroxysmal atrial fibrillation triggers. Kaunas University of Technology / Vilnius University. Project page: [TriggersAF](https://biomedicine.ktu.edu/projects/wearable-technology-for-personalized-identification-and-management-of-paroxysmal-atrial-fibrillation-triggers-triggersaf/). Related clinical publication: Bacevičius J, et al. *EP Europace*. 2025;27(Suppl 1). DOI: [10.1093/europace/euaf085.262](https://doi.org/10.1093/europace/euaf085.262).

8. **Van Gelder IC, Rienstra M, Bunting KV, et al.** 2024 European Society of Cardiology Guidelines for the management of atrial fibrillation developed in collaboration with the European Association for Cardio-Thoracic Surgery. *European Heart Journal*. 2024;45(36):3314-3414. DOI: [10.1093/eurheartj/ehae176](https://doi.org/10.1093/eurheartj/ehae176).

For additional background references, see [`REFERENCES.md`](REFERENCES.md).
