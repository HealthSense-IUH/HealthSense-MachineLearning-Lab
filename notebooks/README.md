# notebooks/ — ba nhóm, ba mục đích

| Thư mục | Mục đích | Có huấn luyện model? |
|---|---|---|
| `training/` | Sinh ra các model V4, V5, V6C từ dữ liệu đặc trưng, có giải thích từng bước. Bản đọc được của script huấn luyện | **Có** |
| `review/` | Xem lại kết quả V5 và V6 từ hiện vật đã khoá: ROC, PR, calibration, LODO, PAC/PVC | Không |
| `defense/` | Bằng chứng tái lập được để bảo vệ trước hội đồng, đọc đúng file CSV đã commit và ghi lại Git HEAD lúc chạy | Không |

Mọi notebook tự tìm gốc repo bằng cách đi ngược lên các thư mục cha cho tới khi thấy `src/healthsense_ml`, nên mở từ thư mục nào cũng chạy. Môi trường: `venv/` của repo (`venv\Scripts\jupyter lab`).

Thứ tự đọc cho người mới: `training/01` → `training/03` → `review/HealthSense_V6_Research_Freeze_Visual_Review` → `defense/02`.
