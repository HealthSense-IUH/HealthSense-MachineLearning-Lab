"""
v4 — Pipeline hiện hành: chấp nhận điểm thấp hơn để có con số THẬT.

Ba sửa chữa cốt lõi so với v1-v3:

1. CHIA THEO BỆNH NHÂN (LOSO). Mỗi vòng giữ trọn một người ra ngoài. Không
   một cửa sổ nào của người đó xuất hiện trong train. Đây là mô phỏng đúng
   tình huống sản phẩm gặp người lạ.

2. TIỀN XỬ LÝ CHỈ NHÌN TRAIN. Scaler nằm trong Pipeline nên fit lại theo
   từng fold; lọc outlier tính ngưỡng trên train và chỉ xóa hàng train —
   tập test giữ nguyên 100%.

3. TINH CHỈNH LỒNG BÊN TRONG (nested CV). GridSearchCV chạy với GroupKFold
   bên trong mỗi fold LOSO, nên tập test của fold không hề tham gia vào
   việc chọn hyperparameter.

Ngoài ra v4 bỏ nhóm đặc trưng LF (LF, LF_norm, LF_HF_Ratio): Task Force 1996
yêu cầu bản ghi tối thiểu 2 phút mới ước lượng được dải LF, mà cửa sổ ở đây
chỉ 30 giây. Giữ lại là tự lừa mình bằng những con số vô nghĩa.

Kết quả: 94.29% mức bệnh nhân, recall 100% (LOSO trên 35 bệnh nhân MIMIC).
Thấp hơn v3 khoảng 4 điểm — nhưng đây là con số dùng được.
"""

from .pipeline import run  # noqa: F401

__version__ = '4.1'
VERSION_ID = 'v4'
