# v1 — Đường cơ sở MIMIC PPG

> "HRV có phân biệt được AFib không?" — Có. Nhưng cách đo điểm thì sai.

| | |
|---|---|
| **Kênh tín hiệu** | PPG |
| **Cửa sổ** | 30 giây, bước 30 giây (không chồng lấn) |
| **Đặc trưng** | 13 cột tuyến tính (7 thời gian + 6 tần số) |
| **Làm sạch** | không có |
| **Chuẩn hóa** | StandardScaler trong Pipeline ✅ đúng |
| **Chia dữ liệu** | ngẫu nhiên 80/20 theo cửa sổ ❌ |
| **Mô hình** | Random Forest (100 cây) |
| **Điểm công bố** | 95.2% (RF) → 95.9% (XGB/LGBM tinh chỉnh) |

## Ý tưởng

Tâm nhĩ rung thì khoảng cách giữa các nhịp tim trở nên hỗn loạn. Đo sự hỗn
loạn đó bằng 13 chỉ số HRV rồi để máy học phân biệt. Ý tưởng này **đúng** —
và nó sống sót qua cả 4 phiên bản.

## Chỗ sai

Chỉ có một, nhưng đủ để làm hỏng mọi con số: **chia ngẫu nhiên theo cửa sổ**.

Mỗi bệnh nhân đóng góp ~40 cửa sổ. Khi trộn đều 1.400 cửa sổ rồi bốc 20%
làm test, gần như chắc chắn mỗi người ở test cũng có mặt trong train. Chạy
thử sẽ thấy con số phũ phàng: **35/35 bệnh nhân test đều xuất hiện trong
train (100%)**.

Mô hình vì thế không cần học "AFib trông như thế nào". Nó chỉ cần học
"nhịp của bác A trông như thế nào" — và bác A có mặt ở cả hai bên. Giống
như cho học sinh làm đúng đề đã ôn: điểm cao nhưng không chứng minh được
điều gì về người mới.

## Chạy thử

```bash
python src/v1/pipeline.py
```

Script chấm cùng một bảng dữ liệu theo hai cách — cách gốc và LOSO theo
bệnh nhân — rồi in ra phần chênh lệch. Kết quả lưu ở `models/v1/results.json`,
báo cáo đầy đủ ở [`src/report/v1_bao_cao.ipynb`](../report/v1_bao_cao.ipynb).

## Ghi chú lịch sử

Trước v1 trên MIMIC còn hai nhánh thử nghiệm nay đã dừng:

- **Thiết bị tự chế** (Custom_PPG_Walking): tự đo bằng MAX30102, 33 cửa sổ,
  một người, một nhãn — không đủ để huấn luyện. Notebook huấn luyện và đánh
  giá chỉ còn là khung rỗng.
- **MIT-BIH AFDB trên ECG** (MIT_BIH_AF): 29.087 cửa sổ,
  25 bệnh nhân, 10 đặc trưng. Chứng minh được ý tưởng HRV→AFib hoạt động.
  Đáng chú ý: bảng đặc trưng **có sẵn cột `record`** (danh tính bệnh nhân)
  nhưng dòng huấn luyện lại chủ động bỏ đi rồi chia ngẫu nhiên. Công cụ để
  làm đúng đã nằm sẵn trong tay từ đầu.
