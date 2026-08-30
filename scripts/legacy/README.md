# Scripts Legacy (v3)

Các script của pipeline v3 — giữ lại để tham khảo lịch sử thí nghiệm.

⚠️ **Không dùng cho kết quả báo cáo.** Pipeline v3 có 2 lỗi data leakage:
1. Không chia dữ liệu theo bệnh nhân (subject leakage) — đặc trưng không có `record_id`.
2. Scaler và ngưỡng IQR fit trên toàn bộ dữ liệu trước khi chia train/test.

Dùng pipeline v4 thay thế: `scripts/run_v4_extraction.py` ➔ `scripts/run_v4_benchmark.py`.
Chi tiết trong README gốc của project, mục "Vì sao có v4?".
