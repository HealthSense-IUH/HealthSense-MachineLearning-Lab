"""
v2 — Giai đoạn "làm sạch dữ liệu" và thêm đặc trưng phi tuyến.

Câu chuyện: v1 sai 11/269 cửa sổ. Nhìn vào các ca sai, nhóm kết luận "đây là
nhãn nhiễu" và viết luật xóa chúng đi. Điểm nhảy từ 95.9% lên 97.4%.
Sau đó thêm SD1, SD2, SampEn (nhóm phi tuyến) -> 16 đặc trưng.

Điểm công bố ngày đó: 97.4% (LightGBM trên dữ liệu đã "làm sạch").

HAI LỖI CHỒNG NHAU:
1. Vẫn chia ngẫu nhiên theo cửa sổ (kế thừa từ v1).
2. Luật làm sạch dựa trên NHÃN và áp lên TOÀN BỘ dữ liệu, kể cả test.
   Nghĩa là những ca khó nhất — AFib trông hiền, người thường trông loạn —
   bị xóa khỏi đề thi. Điểm tăng không phải vì mô hình giỏi lên mà vì
   đề thi dễ đi.

Một lỗi thứ ba tinh vi hơn (notebook 08 cũ): ngưỡng SQI được chọn bằng cách
thử nhiều giá trị rồi xem điểm TEST cái nào cao nhất — xem `sqi_threshold_sweep()`.
"""

from .pipeline import run, sqi_threshold_sweep  # noqa: F401

__version__ = '2.0'
VERSION_ID = 'v2'
