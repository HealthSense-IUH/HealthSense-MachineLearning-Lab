"""
HealthSense ML — Package phân tích PPG và phát hiện Rung Nhĩ (AFib).

Cấu trúc module:
- config:             Đường dẫn, hằng số tín hiệu, danh sách đặc trưng.
- data_loading:       Nạp dữ liệu MIMIC PERform theo TỪNG BỆNH NHÂN (record_id).
- signal_processing:  Lọc bandpass, phát hiện nhịp, trích chuỗi NN/IBI.
- hrv_features:       Tính 16 đặc trưng HRV chuẩn Task Force 1996.
- feature_extraction: Cửa sổ trượt -> bảng đặc trưng (có record_id).
- training:           Benchmark chống leakage: LOSO theo bệnh nhân + nested CV.
- evaluation:         Metrics mức cửa sổ & mức bệnh nhân, biểu đồ.
"""

__version__ = '4.0.0'
