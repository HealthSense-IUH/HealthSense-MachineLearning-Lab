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

## Bốn file `vN.pkl` là gì?

`v1.pkl` … `v4.pkl` là **hiện vật học tập**: nếu mỗi đời pipeline ngày đó đem
đi triển khai thật thì file model sẽ trông như thế nào. Mỗi file kèm một
`vN.json` ghi rõ nguồn gốc và **điểm thật** (đo bằng LOSO), không phải điểm
bản gốc từng công bố.

**Không dùng chúng cho sản phẩm.** Ba lý do, theo mức độ nghiêm trọng:

1. **`v3.pkl` học trên kênh ECG**, không phải PPG. Vòng đeo tay dùng cảm biến
   quang — đưa đặc trưng PPG vào model học từ ECG là sai loại tín hiệu.
2. Cả bốn chỉ học trên **35 bệnh nhân MIMIC**, trong khi model triển khai học
   trên 60 bệnh nhân của hai bộ dữ liệu và đã qua kiểm định chéo.
3. `v4.pkl` dùng Random Forest để so sánh công bằng với v1–v3, **khác** với
   XGBoost tinh chỉnh lồng của model triển khai.

Sinh lại bất cứ lúc nào:

```bash
python src/v1/pipeline.py
```

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

Điểm bản gốc của v1–v3 bị thổi phồng 6–7 điểm do data leakage. File model đi
tới đâu thì con số trung thực đi theo tới đó — đó là việc của `vN.json`.

Cài đặt: [`src/vlab/export.py`](../src/vlab/export.py).

## Lưu ý về git

Cả 5 file `.pkl` đều **được commit** (`.gitignore` có ngoại lệ `!models/*.pkl`),
vì mỗi file chỉ khoảng 200 KB và như vậy mở repo lên là thấy đủ hiện vật.

Riêng `healthsense_afib_pipeline.pkl` thì **bắt buộc** phải commit: nó không
tái tạo được nữa, script sinh ra nó (`scripts/run_final_model.py`) đã bị gỡ
khỏi repo. Khôi phục script bằng `git checkout d3123cf -- scripts`.

Bốn file `vN.pkl` thì tái tạo được bất cứ lúc nào bằng
`python src/vN/pipeline.py`.
