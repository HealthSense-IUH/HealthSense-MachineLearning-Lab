"""
v1 — Đường cơ sở trên MIMIC PPG (2 phiên bản con: v1a ECG, v1b PPG).

Ý tưởng: HRV (nhịp tim biến thiên) đủ để phân biệt AFib hay không.
Cách làm: cắt tín hiệu thành cửa sổ 30 giây KHÔNG chồng lấn, tính 13 đặc
trưng HRV tuyến tính, chia ngẫu nhiên 80/20, huấn luyện Random Forest.

Điểm công bố ngày đó: ~95% accuracy (RF), ~95.9% sau khi tinh chỉnh XGB/LGBM.

LỖI CHÍ MẠNG: chia ngẫu nhiên theo CỬA SỔ. Mỗi bệnh nhân có ~40 cửa sổ nên
gần như chắc chắn người ở test cũng nằm trong train — mô hình chỉ cần nhớ
mặt bệnh nhân. Xem `pipeline.run()` để thấy con số sụp đổ thế nào khi chấm
lại bằng LOSO.
"""

from .pipeline import run  # noqa: F401

__version__ = '1.0'
VERSION_ID = 'v1'
