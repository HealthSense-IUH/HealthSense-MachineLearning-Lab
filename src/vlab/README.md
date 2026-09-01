# vlab — tiện ích dùng chung cho bảo tàng phiên bản

Bốn thư mục `src/v1` … `src/v4` khác nhau ở **phương pháp**. Những thứ không
thuộc về phương pháp — đọc file thô, đo metrics, vẽ biểu đồ, lưu kết quả —
được gom vào đây, để mỗi thư mục `vN` chỉ còn đúng phần "chất riêng" của nó.

| Module | Nội dung |
|---|---|
| `raw.py` | Liệt kê 35 bệnh nhân MIMIC, đọc tín hiệu theo kênh (PPG hoặc ECG) |
| `extract.py` | Cửa sổ trượt tham số hóa → bảng đặc trưng 16 cột, **luôn kèm `record_id`** |
| `metrics.py` | Metrics mức cửa sổ và mức bệnh nhân |
| `honest.py` | **Trung tâm**: chấm cùng một bảng theo 2 cách — ngẫu nhiên vs LOSO |
| `store.py` | Lưu/nạp `models/vN/results.json` |
| `viz.py` | Biểu đồ dùng lại trong notebook báo cáo |

## Hai hàm quan trọng nhất

```python
from vlab import honest

# Cách các phiên bản cũ tự chấm: trộn cửa sổ, bốc ngẫu nhiên 20% làm test
honest.leaky_random_split(df, features, make_model)

# Cách trung thực: giữ trọn một bệnh nhân ra ngoài mỗi vòng
honest.loso(df, features, make_model, train_filter=...)
```

`train_filter` là chỗ để đặt các bước làm sạch (luật theo nhãn của v2, lọc
IQR của v3, lọc IQR của v4). Bản gốc áp luật lên **toàn bộ** dữ liệu — tức là
vứt luôn cả những ca khó trong test. Ở đây luật chỉ được nhìn train của từng
fold; test giữ nguyên 100%.

## Vì sao bảng đặc trưng luôn có `record_id`?

Các phiên bản v1–v3 lịch sử **không có** cột này — và đó chính là lý do chúng
không thể tự kiểm định lại được. Rõ nhất là các bảng đặc trưng đa quy mô của
v3: một khi đã mất danh tính bệnh nhân, chúng vĩnh viễn không chấm lại được,
dù dữ liệu vẫn còn nguyên.

Bảo tàng luôn giữ `record_id` để có thể chấm lại bằng LOSO. Phần huấn luyện
của mỗi phiên bản cũ vẫn **bỏ cột này đi** đúng như bản gốc — xem
`honest.leaky_random_split`.

## Cache

`extract.extract_table()` đặt tên file cache theo tham số:
`data/features/museum_{kênh}_w{cửa sổ}_s{bước}.csv`

Nhờ vậy v1 và v2 (cùng cấu hình PPG/30s/30s) tự động dùng chung một file, còn
4 quy mô của v3 mỗi cái một file riêng.
