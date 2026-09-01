# v2 — Làm sạch dữ liệu + đặc trưng phi tuyến

> Sửa nhầm chỗ: thay vì sửa cách chấm điểm, ta đi xóa các câu hỏi khó.

| | |
|---|---|
| **Kênh tín hiệu** | PPG (giống v1) |
| **Cửa sổ** | 30 giây, bước 30 giây (giống v1) |
| **Đặc trưng** | **16 cột** = 13 của v1 + SD1, SD2, SampEn |
| **Làm sạch** | luật theo nhãn, áp lên **toàn bộ** dữ liệu ❌ |
| **Chuẩn hóa** | StandardScaler trong Pipeline ✅ |
| **Chia dữ liệu** | ngẫu nhiên 80/20 ❌ (kế thừa v1) |
| **Mô hình** | LightGBM tinh chỉnh |
| **Điểm công bố** | 97.4% |

## Câu chuyện

v1 sai 11 trên 269 cửa sổ. Nhóm mở từng ca sai ra xem, thấy chúng "trông
kỳ lạ", và kết luận: đây là **nhãn nhiễu** — dữ liệu gắn nhãn sai chứ mô
hình không sai. Thế là viết luật xóa chúng đi:

```python
# Gắn nhãn Bình thường mà HRV loạn bất thường -> coi là nhãn sai
drop  status == 0  và  (SDNN > 300 hoặc RMSSD > 400 hoặc pNN50 > 90)
# Gắn nhãn AFib mà nhịp lại quá đều -> coi là nhãn sai
drop  status == 1  và  SDNN < 50  và  pNN50 < 20
```

Điểm nhảy từ 95.9% lên 97.4%. Cảm giác như vừa sửa được một lỗi thật.

## Chỗ sai

**Luật này dùng NHÃN để quyết định giữ hay bỏ, và được áp lên cả tập test.**

Hãy để ý luật lọc gì: những ca AFib trông hiền lành và những ca bình thường
trông hỗn loạn — tức là chính xác **những ca khó nhất**. Xóa chúng khỏi
tập test không làm mô hình giỏi lên; nó làm đề thi dễ đi.

Chạy `python src/v2/pipeline.py` sẽ thấy luật xóa 51 cửa sổ, trong đó **50
cửa sổ là ca Normal khó** — gần như toàn bộ những trường hợp người bình
thường có nhịp bất thường, đúng loại ca mà sản phẩm ngoài đời sẽ gặp.

Còn một lỗi thứ ba tinh vi hơn, tái dựng trong `sqi_threshold_sweep()`:
ngưỡng SQI được chọn bằng cách thử nhiều giá trị rồi xem **điểm test** cái
nào cao nhất. Đây là nhìn trộm đáp án ở tầng hyperparameter — và tệ hơn,
mỗi ngưỡng lại thay đổi luôn thành phần tập test, nên các con số thậm chí
không so sánh được với nhau.

## Phần v2 làm ĐÚNG

Ba đặc trưng phi tuyến thêm vào — **SD1, SD2** (Poincaré) và **SampEn**
(Sample Entropy) — là đóng góp thật và được giữ nguyên đến v4. SampEn đo
độ "khó đoán" của chuỗi nhịp, rất hợp với bản chất của AFib.

## Chạy thử

```bash
python src/v2/pipeline.py
```

Kết quả lưu ở `results/v2.json`, báo cáo đầy đủ ở
[`src/report/02_v2_report.ipynb`](../report/02_v2_report.ipynb).
