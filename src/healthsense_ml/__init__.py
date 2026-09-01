"""
HealthSense ML — định nghĩa gốc của phép đo HRV.

Thư mục này từng là toàn bộ pipeline sản phẩm. Sau khi các script huấn luyện
bị gỡ khỏi repo, nó thu lại còn đúng phần **không được phép mất**: định nghĩa
chuẩn của việc biến sóng PPG thành 16 đặc trưng HRV.

Ba module còn lại:
- config:            hằng số tín hiệu (125 Hz, dải lọc, cửa sổ 30s) và danh
                     sách đặc trưng — 16 cột đầy đủ, 13 cột dùng để huấn luyện.
- signal_processing: lọc bandpass -> dò nhịp -> chuỗi NN.
- hrv_features:      chuỗi NN -> 16 đặc trưng chuẩn Task Force 1996.

⚠️ ĐÂY LÀ BẢN THAM CHIẾU CỦA MỘT SERVICE ĐANG CHẠY.
`HealthSense-AI-Service/tests/parity_check.py` import trực tiếp cả ba module
này lúc chạy test, để chứng minh service tính ra kết quả giống hệt lab. Model
`.pkl` trong sản phẩm được huấn luyện trên đặc trưng do code ở đây định nghĩa
— lệch một chút là model nhận sai đầu vào mà không có lỗi nào báo ra.

Sửa ở đây thì phải cập nhật bản sao `app/services/hrv_v4.py` bên AI-Service
và chạy lại parity test.
"""

__version__ = '4.1.0'
