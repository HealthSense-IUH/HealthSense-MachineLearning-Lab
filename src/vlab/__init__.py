"""
vlab — bộ tiện ích dùng chung cho "bảo tàng phiên bản" v1 → v4.

Mục đích: 4 phiên bản pipeline (src/v1 … src/v4) khác nhau ở PHƯƠNG PHÁP
(cách chia dữ liệu, cách tiền xử lý, bộ đặc trưng). Những thứ KHÔNG thuộc
về phương pháp — đọc file thô, đo metrics, vẽ biểu đồ, lưu kết quả — được
gom vào đây để mỗi thư mục vN chỉ còn đúng phần "chất riêng" của nó.

Module:
- raw:     đọc tín hiệu thô MIMIC PERform theo kênh (PPG hoặc ECG).
- metrics: đo hiệu năng ở mức cửa sổ và mức bệnh nhân.
- honest:  đánh giá LOSO trung thực — dùng để "chấm lại" các phiên bản cũ
           trên chính bảng đặc trưng của chúng.
- store:   lưu/nạp kết quả từng phiên bản (JSON) cho notebook báo cáo.
- viz:     biểu đồ dùng lại trong các notebook báo cáo.
"""

__version__ = '1.0.0'
