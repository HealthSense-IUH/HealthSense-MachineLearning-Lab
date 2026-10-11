# tests/ — kiểm thử nhanh, không cần dữ liệu ngoài

Chạy từ gốc repo: `venv\Scripts\python -m pytest tests -q`

| Tệp | Kiểm tra gì |
|---|---|
| `test_hrv_pipeline_smoke.py` | Pipeline lọc, dò nhịp, 16 đặc trưng chạy được trên tín hiệu PPG tổng hợp, ra số hữu hạn và đúng số đặc trưng |

Bài kiểm thử tương đương giữa lab và AI-Service (`parity_check.py`) nằm bên repo HealthSense-AI-Service vì cần cả hai mã nguồn.
