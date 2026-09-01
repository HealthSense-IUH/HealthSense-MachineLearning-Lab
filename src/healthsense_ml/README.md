# healthsense_ml — định nghĩa gốc của phép đo HRV

> Chỉ còn 3 module, 214 dòng. Nhưng là 214 dòng **không được phép sai**.

## Nó còn lại gì

| File | Việc |
|---|---|
| `config.py` | Hằng số tín hiệu: 125 Hz, dải lọc 0.5–8 Hz, cửa sổ 30 giây; danh sách 16 đặc trưng và 13 cột dùng huấn luyện |
| `signal_processing.py` | Lọc bandpass → dò nhịp tim → đo chuỗi NN |
| `hrv_features.py` | Chuỗi NN → **16 đặc trưng** chuẩn Task Force 1996 |

Trước đây thư mục này có 10 module (nạp dữ liệu, cửa sổ trượt, huấn luyện,
đánh giá, AFDB, kiểm chứng nhịp bằng ECG). Sáu module đó đã được gỡ cùng lúc
với `scripts/` — chúng chỉ tồn tại để phục vụ các script huấn luyện, không ai
gọi tới nữa. Lấy lại từ lịch sử git nếu cần:

```bash
git checkout ddd2876 -- src/healthsense_ml
```

## ⚠️ Vì sao 3 file này không xóa được

`HealthSense-AI-Service/tests/parity_check.py` **import trực tiếp cả ba** lúc
chạy test:

```python
ML_LAB = os.path.join(..., "HealthSense-MachineLearning-Lab")
sys.path.insert(0, os.path.join(ML_LAB, "src"))

from healthsense_ml import config as lab_config
from healthsense_ml.hrv_features import compute_hrv_features as lab_features
from healthsense_ml.signal_processing import extract_nn_series as lab_nn
```

Test đó so từng con số giữa lab và bản sao trong service
(`app/services/hrv_v4.py`), bắt buộc khớp **16/16 đặc trưng**.

**Vì sao khắt khe vậy:** model `.pkl` đang chạy được huấn luyện trên đặc trưng
do code này tính. Nếu service tính lệch dù chỉ một chút, model nhận đầu vào
khác với lúc học — và nó vẫn trả về một con số trông hợp lý. Hỏng **âm thầm**,
không có lỗi nào báo ra.

Xóa thư mục này là mất luôn cái neo đó.

## Ai còn dùng trong lab

Chỉ hai chỗ:

- `vlab/extract.py` → `compute_hrv_features` (công thức 16 đặc trưng dùng chung
  cho cả 4 phiên bản, để chênh lệch giữa chúng phản ánh đúng *phương pháp*
  chứ không lẫn với khác biệt công thức)
- `report/00_final_report.ipynb` → `config` (3 hằng số đường dẫn)

## Vì sao tính 16 mà chỉ huấn luyện trên 13

`config.UNRELIABLE_30S_FEATURES` loại `LF`, `LF_norm`, `LF_HF_Ratio`.

Chuẩn Task Force 1996 yêu cầu bản ghi **tối thiểu 2 phút** mới ước lượng được
dải tần thấp. Cửa sổ ở đây chỉ 30 giây — con số tính ra vẫn là số thực trông
hợp lý, nhưng về mặt sinh lý là vô nghĩa. Vẫn **tính** đủ 16 cột để tương
thích, chỉ không **huấn luyện** trên 3 cột đó.

Bản đồ toàn bộ `src/`: [`../README.md`](../README.md)
