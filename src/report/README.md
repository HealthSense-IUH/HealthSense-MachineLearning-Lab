# Báo cáo từng phiên bản

Năm notebook kể lại toàn bộ hành trình ML của HealthSense — viết cho người
đang học, bằng tiếng Việt, mọi con số đều được **tính lại trực tiếp** trong
notebook chứ không chép tay.

| Notebook | Nội dung |
|---|---|
| [`00_tong_quan.ipynb`](00_tong_quan.ipynb) | **Đọc trước.** Toàn cảnh trong 10 phút: vì sao v3 đạt 98.7% lại tệ hơn v4 đạt 94.3% |
| [`v1_bao_cao.ipynb`](v1_bao_cao.ipynb) | Đường cơ sở — và bằng chứng 100% bệnh nhân test nằm trong train |
| [`v2_bao_cao.ipynb`](v2_bao_cao.ipynb) | Khi "làm sạch dữ liệu" biến thành xóa câu khó khỏi đề thi |
| [`v3_bao_cao.ipynb`](v3_bao_cao.ipynb) | Bốn lỗi chồng nhau, và quy luật "càng chồng lấn điểm giả càng đẹp" |
| [`v4_bao_cao.ipynb`](v4_bao_cao.ipynb) | Cách chữa, và vì sao chấp nhận tụt 4 điểm |
| [`bao_cao_san_pham.ipynb`](bao_cao_san_pham.ipynb) | Báo cáo sản phẩm: 3 tầng kiểm định, cross-dataset, kiểm chứng dò nhịp bằng ECG, 2 ca đặc biệt |

## Cách mở

Notebook đã được chạy sẵn và **nhúng đầy đủ kết quả + biểu đồ**, nên mở lên
là đọc được ngay, không cần chạy lại.

Muốn chạy lại:

```bash
python -m jupyter lab src/report
```

Mỗi notebook tự nạp kết quả từ `models/vN/results.json`. Nếu file chưa có, ô
lệnh đầu tiên sẽ tự chạy pipeline tương ứng (mất vài phút cho lần đầu, sau đó
bảng đặc trưng được cache lại trong `data/features/museum_*.csv`).

Chạy lại toàn bộ 4 phiên bản từ đầu:

```bash
python src/v1/pipeline.py
python src/v2/pipeline.py
python src/v3/pipeline.py
python src/v4/pipeline.py
```

## Ý tưởng chung của bộ báo cáo

Bốn phiên bản được chạy lại trên **cùng một bộ dữ liệu** (35 bệnh nhân MIMIC
PERform) và chấm bằng **cùng một thước đo**, để trả lời một câu hỏi:

> Phần "tiến bộ" từ 95.9% (v1) lên 98.7% (v3) là thật hay ảo?

Câu trả lời nằm ở chỗ mỗi phiên bản được chấm hai lần:

1. **Cách bản gốc tự chấm** — chia ngẫu nhiên theo cửa sổ 30 giây.
2. **Cách trung thực (LOSO)** — giữ trọn một bệnh nhân ra ngoài mỗi vòng.

Chênh lệch giữa hai con số chính là phần điểm ảo do data leakage.

## Lưu ý về tính trung thực của việc tái dựng

Đây là **tái dựng có kiểm soát**, không phải chạy lại nguyên xi notebook cũ.

Phần toán học chung — bộ lọc bandpass, thuật toán dò nhịp, công thức 16 đặc
trưng HRV — được **giữ cố định** cho cả 4 phiên bản. Chỉ những thứ thật sự
định nghĩa nên mỗi phiên bản mới thay đổi: kênh tín hiệu, bước trượt cửa sổ,
bộ đặc trưng, luật làm sạch, cách chuẩn hóa, cách chia dữ liệu, mô hình.

Lý do: nếu tái dựng cả những tiểu tiết xử lý tín hiệu khác nhau, ta sẽ không
biết chênh lệch kết quả đến từ **phương pháp** hay chỉ từ **tham số dò đỉnh**.
Cố định phần chung biến bảo tàng này thành một thí nghiệm có đối chứng.

Vì vậy con số tái dựng không trùng khít con số lịch sử (ví dụ v1 tái dựng cho
98.2% thay vì 95.9% mà notebook cũ ghi). Điều được tái hiện thành công là
**hiện tượng**: cách chấm cũ luôn cho điểm cao hơn cách chấm thật khoảng 6–7
điểm phần trăm, và khoảng cách đó nới rộng khi cửa sổ chồng lấn nhiều hơn.
Con số gốc của từng phiên bản được ghi trong trường `original_claim` của mỗi
`results.json` để đối chiếu.

Notebook thí nghiệm gốc của v1–v3 đã được gỡ khỏi repo cho gọn. Nếu cần đối
chiếu nguyên bản, chúng vẫn nằm trong lịch sử git (commit `e61e601` trở về trước).

## Liên quan

- Mã nguồn từng phiên bản: [`src/v1`](../v1) · [`src/v2`](../v2) · [`src/v3`](../v3) · [`src/v4`](../v4)
- Tiện ích dùng chung: [`src/vlab`](../vlab)
- Pipeline sản phẩm đầy đủ: [`src/healthsense_ml`](../healthsense_ml) + `scripts/run_*.py`
- Bộ slide giải thích toàn bộ phần ML: [`docs/HealthSense_ML_Slides.pptx`](../../docs/HealthSense_ML_Slides.pptx)
