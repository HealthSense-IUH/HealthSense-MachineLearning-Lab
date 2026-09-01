"""
v3 — Benchmark đa quy mô: nơi con số đẹp nhất và sai nhiều nhất.

Ý tưởng: nếu 1.360 cửa sổ cho 97%, thì 16.358 cửa sổ chắc còn tốt hơn. Nhóm
tạo 4 quy mô dữ liệu bằng cách cho cửa sổ 30 giây trượt với bước ngắn dần
(30s, 10s, 5s, 2.5s), rồi ném vào 7 mô hình kể cả Voting và Stacking.

Điểm công bố ngày đó: 98.65% (Stacking), 98.73% / AUC 0.9998 (RF).
Đây chính là những con số bị README gọi là "đẹp nhưng ảo".

BỐN LỖI CHỒNG NHAU:
1. Cửa sổ chồng lấn tới 91%: hai hàng cạnh nhau gần như cùng một đoạn tín
   hiệu. Chia ngẫu nhiên là gần như chắc chắn có bản sao của mẫu test nằm
   trong train.
2. StandardScaler/MinMaxScaler fit trên TOÀN BỘ dữ liệu rồi mới chia.
3. Lọc outlier IQR tính ngưỡng trên toàn bộ bảng.
4. Vẫn chia ngẫu nhiên, không theo bệnh nhân.

MỘT CHI TIẾT ÍT AI ĐỂ Ý: v3 trích đặc trưng từ kênh ECG chứ không phải PPG
(`sub['ecg'].values` trong notebook cũ). Nghĩa là con số v3 không so sánh
trực tiếp được với v1/v2, và một sản phẩm đeo tay dùng PPG lại đang dựa trên
kết quả đo bằng điện tâm đồ.
"""

from .pipeline import run  # noqa: F401

__version__ = '3.0'
VERSION_ID = 'v3'
