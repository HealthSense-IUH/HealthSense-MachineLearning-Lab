# src/ — có gì trong này

Mở thư mục này ra thấy 7 mục. Đây là bản đồ đọc trong 2 phút.

## Đọc theo thứ tự nào

Nếu bạn mới vào dự án, **mở [`report/00_final_report.ipynb`](report/00_final_report.ipynb) trước**.
Notebook đó kể toàn bộ câu chuyện, mọi thứ còn lại chỉ là code đằng sau nó.

## Bảy thư mục làm gì

| Thư mục | Nó là gì | Ví von |
|---|---|---|
| `healthsense_ml/` | **Cái máy.** Biến sóng ánh sáng từ cảm biến thành chẩn đoán. | Cỗ máy chính |
| `vlab/` | **Cái thước.** Chấm điểm và so sánh 4 phiên bản cho công bằng. | Hội đồng chấm thi |
| `v1/` `v2/` `v3/` `v4/` | **Bốn tờ khai.** Mỗi tờ mô tả một đời pipeline làm khác nhau chỗ nào. | 4 công thức nấu ăn |
| `report/` | **Năm bài đọc.** Notebook đã chạy sẵn, mở ra là đọc được. | Bài giảng |

Ba nhóm này **không lồng vào nhau** — chúng nằm ngang hàng. Quan hệ giữa chúng
là "ai gọi ai":

```
v1  v2  v3  v4          ← mỗi thư mục ~150 dòng, chỉ khai báo cấu hình
 └───┴───┴───┘
       ↓ gọi
     vlab               ← cắt dữ liệu, chấm điểm, vẽ biểu đồ
       ↓ mượn công thức HRV
  healthsense_ml        ← lọc sóng, dò nhịp, tính 16 đặc trưng
```

Mũi tên chỉ đi **một chiều xuống**. `vlab` không biết v1–v4 tồn tại;
`healthsense_ml` càng không biết gì về `vlab`.

---

## `healthsense_ml/` — cái máy

Đây là **phần y học** của dự án: nhét sóng PPG vào, nhả ra "P(AFib) = 0.87".

Đi theo dòng dữ liệu:

1. `data_loading.py` — đọc file thô của 35 bệnh nhân
2. `signal_processing.py` — lọc nhiễu → tìm từng nhịp tim → đo khoảng cách
   giữa các nhịp (dãy NN)
3. `hrv_features.py` — tóm dãy NN thành **16 con số** chuẩn Task Force 1996
4. `feature_extraction.py` — làm việc trên từng cửa sổ 30 giây → ra bảng số
5. `training.py` + `evaluation.py` — huấn luyện và đo kết quả
6. `afdb.py`, `beat_validation.py` — bộ dữ liệu thứ hai và kiểm chứng bằng ECG

> ⚠️ **Đừng đổi tên hay di chuyển thư mục này.** `HealthSense-AI-Service` giữ
> một bản sao của nó (`app/services/hrv_v4.py`) và có bài test đối chiếu
> (`tests/parity_check.py`) bắt buộc service phải tính ra **16/16 đặc trưng
> giống hệt** ở đây. Model `.pkl` được huấn luyện trên đặc trưng do code này
> tính — service tính lệch là model nhận sai đầu vào mà không ai biết.
> Chi tiết: [`healthsense_ml/README.md`](healthsense_ml/README.md).

## `vlab/` — cái thước

Viết tắt của **"version lab"** (phòng thí nghiệm so phiên bản).

Thư mục này **không biết AFib là gì** và không chẩn đoán ai cả. Việc của nó:
đưa cho tôi một bảng số + một mô hình, tôi chấm điểm và vẽ biểu đồ.

File quan trọng nhất là `honest.py`, chỉ 3 hàm nhưng là linh hồn của cả bộ:

| Hàm | Chấm theo cách nào |
|---|---|
| `leaky_random_split()` | Cách **cũ** — trộn hết cửa sổ, bốc ngẫu nhiên 20% làm đề thi |
| `loso()` | Cách **trung thực** — giữ trọn một bệnh nhân ra ngoài mỗi vòng |
| `compare()` | Chấm **cả hai** trên cùng một bảng, trả về phần chênh lệch |

Chính `compare()` sinh ra con số "+6.14 điểm ảo" bạn thấy trong báo cáo v1.

Các file còn lại phục vụ việc đó: `extract.py` (cắt cửa sổ theo tham số),
`metrics.py` (đo 2 mức: cửa sổ và bệnh nhân), `store.py` (lưu kết quả),
`viz.py` (biểu đồ). Chi tiết: [`vlab/README.md`](vlab/README.md).

## `v1/` `v2/` `v3/` `v4/` — bốn tờ khai

Mỗi thư mục chỉ có **3 file, hơn 100 dòng**, vì phần việc nặng đã nằm ở
`vlab` và `healthsense_ml`. Nội dung chủ yếu là khai báo:

```python
CHANNEL  = 'PPG'         # dùng kênh nào
WINDOW_S = 30.0          # cửa sổ dài bao nhiêu giây
STEP_S   = 30.0          # trượt bước bao nhiêu
FEATURES = LINEAR_13     # dùng bộ đặc trưng nào
def make_model(): ...    # mô hình gì
```

Rồi nhờ `vlab` làm việc:

```python
df = extract_table(CHANNEL, WINDOW_S, STEP_S)
result = honest.compare(df, FEATURES, make_model)
```

**Vì sao tách mỏng như vậy:** nếu mỗi thư mục v tự chứa thuật toán riêng, khi
thấy v3 điểm khác v1 sẽ không biết là do *phương pháp khác* hay do *code khác*.
Cố định phần dùng chung thì phần khác nhau giữa 4 thư mục chính là — và chỉ
là — phương pháp. Đọc `v1/pipeline.py` cạnh `v3/pipeline.py` là thấy ngay.

Mỗi thư mục có README riêng dạng "thẻ phiên bản" ghi rõ cấu hình, chỗ làm
đúng, chỗ làm sai: [v1](v1/README.md) · [v2](v2/README.md) ·
[v3](v3/README.md) · [v4](v4/README.md)

Chạy lại một phiên bản:

```bash
python src/v1/pipeline.py
```

## `report/` — năm bài đọc

Notebook tiếng Việt **đã chạy sẵn**, nhúng đủ kết quả và biểu đồ, mở ra đọc
được ngay không cần chạy lại.

| Notebook | Nội dung |
|---|---|
| `00_final_report.ipynb` | **Bắt đầu ở đây.** Phần A: kết quả sản phẩm (3 tầng kiểm định). Phần B: hành trình 4 phiên bản |
| `01_v1_report.ipynb` | Đường cơ sở — bằng chứng 100% bệnh nhân test nằm trong train |
| `02_v2_report.ipynb` | Khi "làm sạch dữ liệu" biến thành xóa câu khó khỏi đề thi |
| `03_v3_report.ipynb` | Bốn lỗi chồng nhau, quy luật "càng chồng lấn điểm giả càng đẹp" |
| `04_v4_report.ipynb` | Cách chữa, và vì sao chấp nhận tụt 4 điểm |

```bash
python -m jupyter lab src/report
```
