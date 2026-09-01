# healthsense_ml — cái máy

> Nhét sóng ánh sáng từ cảm biến vào, nhả ra "P(AFib) = 0.87".

Đây là **phần y học** của dự án. Toàn bộ hiểu biết về "rung nhĩ trông như thế
nào trong tín hiệu mạch" nằm ở đây, không nằm chỗ nào khác.

Đừng nhầm với tên repo (`HealthSense-MachineLearning-Lab`) — thư mục này chỉ
là **một phần** của repo, phần lõi tính toán.

## ⚠️ Đọc trước khi sửa

`HealthSense-AI-Service` giữ một **bản sao có kiểm soát** của thư mục này:

- Bản sao: `HealthSense-AI-Service/app/services/hrv_v4.py`
- Bài kiểm tra đối chiếu: `HealthSense-AI-Service/tests/parity_check.py`

Bài test đó bắt buộc service phải tính ra **16/16 đặc trưng giống hệt** code
ở đây, với cùng một tín hiệu đầu vào.

**Vì sao khắt khe vậy:** model `.pkl` đang chạy trong sản phẩm được huấn luyện
trên đặc trưng do code này tính. Nếu service tính lệch dù chỉ một chút, model
sẽ nhận đầu vào khác với lúc học — và nó vẫn trả về một con số trông hợp lý,
nên **sai âm thầm**, không có lỗi nào báo ra.

Vì vậy: sửa `signal_processing.py` hoặc `hrv_features.py` ở đây thì **phải
cập nhật bản sao bên AI-Service và chạy lại parity test**.

## Dòng dữ liệu

| Bước | File | Việc |
|---|---|---|
| 1 | `data_loading.py` | Đọc file thô 35 bệnh nhân MIMIC (tự tải từ Kaggle nếu thiếu) |
| 2 | `signal_processing.py` | Lọc nhiễu → dò từng nhịp tim → đo khoảng cách giữa các nhịp (**dãy NN**) |
| 3 | `hrv_features.py` | Tóm dãy NN thành **16 con số** chuẩn Task Force 1996 |
| 4 | `feature_extraction.py` | Lặp bước 2–3 trên từng cửa sổ 30 giây → bảng đặc trưng có `record_id` |
| 5 | `training.py` | Huấn luyện LOSO theo bệnh nhân, tinh chỉnh lồng bằng GroupKFold |
| 6 | `evaluation.py` | Đo kết quả 2 mức (cửa sổ / bệnh nhân) + biểu đồ |

Hai file phụ:

- `afdb.py` — bộ dữ liệu thứ hai MIT-BIH AFDB, dựng dãy NN thẳng từ annotation
  QRS trên PhysioNet (không cần tải sóng thô).
- `beat_validation.py` — chấm điểm bộ dò nhịp PPG bằng R-peak trên ECG ghi
  song song. Chính công cụ này đã phát hiện 2 ca `non_af_012` (nhãn sai) và
  `non_af_014` (tín hiệu PPG hỏng).

`config.py` giữ mọi hằng số: tần số lấy mẫu, dải lọc, độ dài cửa sổ, danh sách
đặc trưng.

## Vì sao 16 đặc trưng mà chỉ huấn luyện trên 13

`config.UNRELIABLE_30S_FEATURES` loại `LF`, `LF_norm`, `LF_HF_Ratio`.

Chuẩn Task Force 1996 yêu cầu bản ghi **tối thiểu 2 phút** mới ước lượng được
dải tần thấp (LF). Cửa sổ ở đây chỉ 30 giây — con số tính ra vẫn là số thực
trông hợp lý, nhưng về mặt sinh lý là vô nghĩa.

Pipeline vẫn **tính** đủ 16 cột để tương thích, chỉ không **huấn luyện** trên
3 cột đó.

## Quan hệ với phần còn lại của `src/`

`vlab/` mượn đúng một thứ từ đây:

```python
from healthsense_ml.hrv_features import compute_hrv_features
```

Nghĩa là **công thức 16 đặc trưng chỉ tồn tại một bản duy nhất**, ở đây. Bảo
tàng phiên bản (`src/v1` … `src/v4`) dùng lại y hệt, không viết lại — nhờ vậy
chênh lệch kết quả giữa 4 phiên bản phản ánh đúng *phương pháp*, không lẫn với
khác biệt công thức.

Bản đồ toàn bộ `src/`: [`../README.md`](../README.md)
