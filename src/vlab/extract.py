"""Trích bảng đặc trưng bằng cửa sổ trượt — tham số hóa theo từng phiên bản.

NGUYÊN TẮC THIẾT KẾ CỦA BẢO TÀNG PHIÊN BẢN
------------------------------------------
Bốn phiên bản v1-v4 khác nhau ở rất nhiều thứ. Nếu tái dựng nguyên xi từng
chi tiết (kể cả tham số dò đỉnh hơi khác nhau), ta sẽ không biết chênh lệch
kết quả đến từ PHƯƠNG PHÁP hay chỉ từ tiểu tiết xử lý tín hiệu.

Vì vậy bảo tàng cố định phần toán học chung — bộ lọc bandpass, thuật toán dò
nhịp, công thức 16 đặc trưng HRV (dùng lại healthsense_ml) — và chỉ thay đổi
đúng những thứ ĐỊNH NGHĨA nên mỗi phiên bản:

  * kênh tín hiệu   : PPG (v1, v2, v4) hay ECG (v3)
  * độ dài cửa sổ   : 30 giây (mọi phiên bản)
  * bước trượt      : 30s = không chồng lấn (v1, v2) … 2.5s = chồng 91% (v3)
  * bộ đặc trưng    : 13 cột (v1) / 16 cột (v2, v3) / 13 cột bỏ nhóm LF (v4)
  * luật làm sạch   : không có (v1) / luật theo nhãn (v2) / IQR toàn cục (v3)
  * cách chuẩn hóa  : trong pipeline (v1, v2) / fit toàn bộ (v3) / theo fold (v4)
  * cách chia dữ liệu: ngẫu nhiên theo cửa sổ (v1-v3) / LOSO theo người (v4)

Nhờ vậy phần chênh lệch quan sát được là do phương pháp — đúng thứ ta muốn đo.

MỘT ĐIỂM QUAN TRỌNG: bảng ở đây LUÔN có cột record_id, kể cả khi tái dựng
các phiên bản cũ vốn không có nó. Lý do: phải có danh tính bệnh nhân thì mới
"chấm lại" được bằng LOSO. Phần huấn luyện của mỗi phiên bản cũ vẫn bỏ cột
này đi đúng như bản gốc — xem `honest.leaky_random_split`.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from .raw import FS, features_path, list_records, load_channel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from healthsense_ml.hrv_features import compute_hrv_features  # noqa: E402

# 16 đặc trưng HRV chuẩn Task Force 1996
ALL_FEATURES = [
    'HR_mean', 'Mean_NN', 'SDNN', 'RMSSD', 'NN50', 'pNN50', 'CV',
    'LF', 'HF', 'Total_Power', 'LF_HF_Ratio', 'LF_norm', 'HF_norm',
    'SD1', 'SD2', 'SampEn',
]

# Bộ 13 cột của v1/v2 giai đoạn đầu (chưa có nhóm phi tuyến)
LINEAR_13 = [f for f in ALL_FEATURES if f not in ('SD1', 'SD2', 'SampEn')]

# Bộ 13 cột của v4: bỏ nhóm LF vì cửa sổ 30 giây quá ngắn để ước lượng LF
# (Task Force 1996 yêu cầu bản ghi >= 2 phút cho dải LF).
UNRELIABLE_30S = ['LF', 'LF_norm', 'LF_HF_Ratio']
CORE_13 = [f for f in ALL_FEATURES if f not in UNRELIABLE_30S]

# Giới hạn sinh lý của khoảng NN (ms)
NN_MIN_MS, NN_MAX_MS = 250, 2000


def _bandpass(sig, low, high, order=4, fs=FS):
    sos = sp_signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return sp_signal.sosfiltfilt(sos, sig)


def _detect_beats(filtered, channel, fs=FS):
    """Dò nhịp trên tín hiệu đã lọc, trả về thời điểm nhịp (giây).

    Prominence tính trên tín hiệu z-score nên không phụ thuộc biên độ tuyệt
    đối. Ngưỡng cho ECG cao hơn PPG: sóng R rất nhọn, cần chặn để không bắt
    nhầm sóng T ngay sau đó.
    """
    std = np.std(filtered)
    if std == 0:
        return np.array([])
    z = (filtered - np.mean(filtered)) / std

    if channel == 'ECG':
        distance_s, prominence = 0.30, 1.0
    else:
        distance_s, prominence = 0.27, 0.5

    peaks, _ = sp_signal.find_peaks(
        z, distance=int(distance_s * fs), prominence=prominence)
    return peaks / fs


def nn_series(sig, channel, fs=FS):
    """Tín hiệu thô -> (nn_ms, nn_times_s).

    Chỉ loại khoảng NN phi sinh lý (<250ms hoặc >2000ms) do lỗi dò đỉnh.
    KHÔNG lọc theo độ lệch so với trung vị: với AFib, chính sự bất thường
    của khoảng NN là dấu hiệu bệnh cần giữ lại.
    """
    low, high = (0.5, 40.0) if channel == 'ECG' else (0.5, 8.0)
    filtered = _bandpass(sig, low, high, fs=fs)
    beats = _detect_beats(filtered, channel, fs)
    if len(beats) < 2:
        return np.array([]), np.array([])

    nn = np.diff(beats) * 1000.0
    nn_t = beats[1:]
    mask = (nn >= NN_MIN_MS) & (nn <= NN_MAX_MS)
    return nn[mask], nn_t[mask]


def extract_record(record_id, label, csv_path, channel='PPG',
                   window_s=30.0, step_s=30.0, min_beats=10):
    """Trích đặc trưng cửa sổ trượt cho 1 bệnh nhân."""
    t, sig = load_channel(csv_path, channel)
    nn, nn_t = nn_series(sig, channel)
    if len(nn) == 0:
        return []

    rows, t_end, start = [], t[-1], 0.0
    while start + window_s <= t_end:
        mask = (nn_t >= start) & (nn_t < start + window_s)
        nn_w, nn_tw = nn[mask], nn_t[mask]
        if len(nn_w) >= min_beats:
            feats = compute_hrv_features(nn_w, nn_tw)
            feats['record_id'] = record_id
            feats['t_start'] = start
            feats['status'] = label
            rows.append(feats)
        start += step_s
    return rows


def cache_name(channel, window_s, step_s):
    tag_step = str(step_s).replace('.', 'p')
    return f'museum_{channel.lower()}_w{int(window_s)}_s{tag_step}.csv'


def extract_table(channel='PPG', window_s=30.0, step_s=30.0, min_beats=10,
                  use_cache=True, verbose=True):
    """Trích (hoặc nạp từ cache) bảng đặc trưng cho một cấu hình cửa sổ.

    Cache đặt tên theo tham số nên v1 và v2 — cùng cấu hình PPG/30s/30s —
    tự động dùng chung một file, còn 4 quy mô của v3 mỗi cái một file.
    """
    path = features_path(cache_name(channel, window_s, step_s))
    if use_cache and os.path.exists(path):
        if verbose:
            print(f'Dùng cache: {os.path.basename(path)}')
        return pd.read_csv(path)

    records = list_records()
    if verbose:
        print(f'Trích đặc trưng: kênh {channel}, cửa sổ {window_s}s, '
              f'bước {step_s}s, {len(records)} bệnh nhân')

    all_rows = []
    for i, (record_id, label, csv_path) in enumerate(records, 1):
        rows = extract_record(record_id, label, csv_path, channel,
                              window_s, step_s, min_beats)
        all_rows.extend(rows)
        if verbose:
            tag = 'AFib  ' if label == 1 else 'Normal'
            print(f'  [{i:>2}/{len(records)}] {record_id:<28} {tag} '
                  f'{len(rows):>4} cửa sổ')

    df = pd.DataFrame(all_rows)[['record_id', 't_start', 'status'] + ALL_FEATURES]
    df.to_csv(path, index=False)
    if verbose:
        n_af = int((df['status'] == 1).sum())
        print(f'\nTổng {len(df)} cửa sổ ({n_af} AFib / {len(df) - n_af} Normal) '
              f'từ {df["record_id"].nunique()} bệnh nhân -> {os.path.basename(path)}')
    return df
