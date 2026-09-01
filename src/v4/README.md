# v4 — Pipeline hiện hành

> Chấp nhận điểm thấp hơn 4 điểm để đổi lấy con số dùng được.

| | |
|---|---|
| **Kênh tín hiệu** | PPG (đúng loại cảm biến của vòng đeo) |
| **Cửa sổ** | 30 giây, bước 10 giây → chồng lấn 67% ✅ an toàn (xem bên dưới) |
| **Đặc trưng** | 13 cột = 16 trừ nhóm LF |
| **Làm sạch** | IQR ×3.0, ngưỡng tính **trên train**, chỉ xóa hàng train ✅ |
| **Chuẩn hóa** | StandardScaler trong Pipeline, fit lại **theo từng fold** ✅ |
| **Chia dữ liệu** | **LOSO theo bệnh nhân** ✅ |
| **Mô hình** | LR / RF / XGBoost, tinh chỉnh lồng bằng GroupKFold |
| **Kết quả thật** | 94.29% mức bệnh nhân, recall 100% (35 bệnh nhân MIMIC) |

## Ba sửa chữa cốt lõi

**1. Chia theo bệnh nhân (LOSO).** Mỗi vòng giữ trọn một người ra ngoài,
huấn luyện trên 34 người còn lại. Không một cửa sổ nào của người đó xuất
hiện trong train. Đây là mô phỏng đúng tình huống sản phẩm gặp người lạ —
mà đó chính là tình huống duy nhất có thật.

**2. Tiền xử lý chỉ nhìn train.** Scaler nằm trong `Pipeline` nên fit lại
theo từng fold. Lọc outlier tính ngưỡng trên train và **chỉ xóa hàng train**;
tập test giữ nguyên 100%. Ngoài đời không ai được phép vứt dữ liệu của bệnh
nhân mới chỉ vì nó khó.

**3. Tinh chỉnh lồng bên trong (nested CV).** `GridSearchCV` chạy với
`GroupKFold(3)` bên trong mỗi fold LOSO, nên tập test của fold không tham
gia vào việc chọn hyperparameter.

## Vì sao vẫn dùng cửa sổ chồng lấn mà không sao?

Đây là điểm hay gây nhầm. Chồng lấn **tự nó không phải leakage** — nó chỉ
nguy hiểm khi **đi kèm chia ngẫu nhiên**, vì lúc đó bản sao gần đúng của một
cửa sổ test nằm sẵn trong train.

Với LOSO, toàn bộ cửa sổ của một người nằm trọn về một phía của ranh giới.
Chúng chồng lên nhau bao nhiêu cũng không thể vượt qua ranh giới đó. Khi ấy
chồng lấn chỉ còn tác dụng tốt: có thêm mẫu để học từ cùng một lượng tín hiệu.

## Vì sao bỏ nhóm LF?

`LF`, `LF_norm`, `LF_HF_Ratio` bị loại khỏi bộ đặc trưng huấn luyện. Chuẩn
Task Force 1996 yêu cầu bản ghi **tối thiểu 2 phút** mới ước lượng được dải
tần thấp (LF). Cửa sổ ở đây chỉ 30 giây — con số tính ra vẫn là một số thực
trông có vẻ hợp lý, nhưng về mặt sinh lý là vô nghĩa. Giữ lại chỉ là tự lừa
mình. Pipeline vẫn **tính** đủ 16 cột để tương thích, chỉ không **huấn luyện**
trên 3 cột đó.

## Điều quan trọng nhất mà script này chứng minh

Chạy `python src/v4/pipeline.py` sẽ thấy: nếu đem **chính dữ liệu v4** ra
chấm theo cách cũ (chia ngẫu nhiên), điểm cũng vọt lên ~98%.

Nghĩa là dữ liệu v4 không hề "khó hơn", mô hình v4 cũng không "kém hơn".
**Toàn bộ chênh lệch nằm ở cách chấm điểm.** Các phiên bản trước không có
mô hình tệ — chúng có một cái thước đo hỏng.

## Kiểm định ba tầng của sản phẩm

Con số 94.29% ở trên mới là tầng 1. Mô hình đang chạy còn qua hai tầng nữa:

| Tầng | Cách kiểm | Kết quả |
|---|---|---|
| 1. LOSO trên MIMIC | giữ ra từng bệnh nhân | 94.29% mức bệnh nhân, recall 100% |
| 2. Chéo bộ dữ liệu | học MIMIC → thi AFDB và ngược lại | AUC 0.987 / 0.976 |
| 3. Gộp 60 bệnh nhân | LOSO trên MIMIC + AFDB | 95.89%, AUC 0.988 (XGBoost) |

Tầng 2 là tầng khắt khe nhất: mô hình phải làm việc trên bộ dữ liệu khác,
thiết bị khác, thậm chí loại tín hiệu khác. Qua được tầng này thì mới tin
được là nó học "dấu hiệu của AFib" chứ không phải "đặc điểm của bộ dữ liệu".

## Chạy thử

```bash
python src/v4/pipeline.py
```

Kết quả lưu ở `models/v4/results.json`. Báo cáo đầy đủ ở
[`src/report/04_v4_report.ipynb`](../report/04_v4_report.ipynb).

Engine của pipeline sản phẩm vẫn nằm ở `src/healthsense_ml/`, nhưng các script
dựng nên mô hình triển khai (AFDB, cross-dataset, mô hình gộp, xuất `.pkl`) đã
được gỡ khỏi repo — kết quả của chúng còn trong `models/`, khôi phục script
bằng `git checkout d3123cf -- scripts`.
