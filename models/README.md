# models/ — thứ nạp được vào chương trình

Thư mục này **chỉ chứa model**. Số liệu, biểu đồ, bảng kết quả nằm ở
[`results/`](../results/README.md).

## Cái nào đem đi triển khai?

**Đúng một file: `healthsense_afib_pipeline.pkl`.**

| | |
|---|---|
| Nội dung | sklearn Pipeline: `StandardScaler` → `XGBClassifier` (150 cây) |
| Đầu vào | 13 đặc trưng HRV, **đúng thứ tự** ghi trong `model_card.json` |
| Đầu ra | `predict_proba[:, 1]` = P(AFib), ngưỡng mặc định 0.5 |
| Huấn luyện trên | 60 bệnh nhân (MIMIC 35 + AFDB 25), có cân bằng nguồn |
| Đang chạy ở | `HealthSense-AI-Service/app/models/` (bản sao giống hệt từng byte) |

Thẻ đầy đủ: [`model_card.json`](model_card.json).

## Hai quy tắc khi xuất model — học từ sai lầm có thật

**1. Luôn gói scaler vào trong Pipeline.**

`HealthSense-AI-Service/app/models/` từng chứa `best_model_8165.pkl` — một
`MLPClassifier` **trần, không kèm scaler**, sinh ra từ pipeline v3 vốn chuẩn
hóa dữ liệu toàn cục từ trước. Ai nạp nó rồi đưa đặc trưng thô vào sẽ nhận kết
quả rác **mà không có lỗi nào báo ra**.

File đó đã được gỡ khỏi service, và `load_model()` bên đó nay từ chối mọi model
không đóng gói tiền xử lý bên trong Pipeline. Gói scaler vào Pipeline khiến
chuyện này không thể xảy ra ngay từ đầu.

**2. Luôn kèm thẻ ghi điểm thật.**

Các đời pipeline đầu (v1–v3, nay đã gỡ khỏi repo) từng bị thổi phồng 6–7 điểm do data leakage. File model đi
tới đâu thì con số trung thực đi theo tới đó — đó là việc của model card `.json` đặt cạnh mỗi `.pkl`.

Notebook `training/` theo đúng quy tắc này: luôn ghi model card JSON cạnh file pkl.

## Lưu ý về git

`healthsense_afib_pipeline.pkl` (248 KB) được commit nhờ ngoại lệ `!models/*.pkl`; các model lớn hơn trong `mimic/`, `multidomain/` không commit.

File này **bắt buộc** commit vì AI-Service dùng bản
sao từng byte của nó. Tái tạo bằng `scripts/run_final_model.py` hoặc
`notebooks/training/01_train_v4_pooled_model.ipynb`; hai cách này ghi ra
`models/final/` để không đè lên bản đang triển khai, muốn thay thì chép tay.

## Các đời sau V4

| Thư mục | Model | Commit? | Cách có |
|---|---|---|---|
| `mimic/` | V5: `healthsense_af_v5_rf_ac_frozen.pkl` + `.json`, Random Forest 14 đặc trưng, đóng băng trên MIMIC | Chỉ JSON | `experiments/06/train_frozen_model.py` hoặc `notebooks/training/02` |
| `multidomain/` | V6C: `healthsense_af_v6c_stacking.pkl` (khoảng 1,4 GB) + các `.json` của V6A, V6B, V6C | Chỉ JSON | `experiments/07/train_v6c_stacking.py` hoặc `notebooks/training/03`; bản chuẩn xin từ máy đã train, đối chiếu SHA-256 trong `data/MANIFEST.csv` |
| `final/` | Đầu ra của lần chạy lại V4 | Không | Notebook `training/01` |

Notebook 02 và 03 mặc định ghi tên `*_retrained.pkl`, chỉ ghi đè tên đóng băng khi bật `OVERWRITE_FROZEN`.
