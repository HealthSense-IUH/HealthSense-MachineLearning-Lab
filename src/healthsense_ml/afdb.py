"""MIT-BIH AF Database (AFDB) — nguồn dữ liệu thứ hai cho cross-dataset.

Điểm khác MIMIC PERform:
- ECG 250 Hz thay vì PPG: nhưng ta KHÔNG cần sóng thô — AFDB có sẵn
  annotation QRS (vị trí từng nhịp tim) và annotation rhythm (đoạn nào
  là AFib / Normal). Chỉ tải annotation (~vài trăm KB/bệnh nhân) là đủ
  để dựng chuỗi NN, thay vì ~800 MB tín hiệu.
- AF kịch phát: cùng một bệnh nhân có cả đoạn AFib lẫn Normal — buộc
  mô hình học "dấu hiệu AFib" thay vì "nhận mặt bệnh nhân".

Đặc trưng HRV được tính bằng CHÍNH hrv_features.compute_hrv_features của
pipeline v4 — bắt buộc để so sánh chéo MIMIC <-> AFDB hợp lệ.
"""

import os

import numpy as np
import pandas as pd

from . import config
from .signal_processing import beats_to_nn
from .hrv_features import compute_hrv_features

AFDB_DB = 'afdb'
AFDB_FS = 250.0

# Cửa sổ 30s giống MIMIC; bước 30s (không chồng lấn) vì mỗi record dài ~10h,
# bước 10s sẽ tạo ~80k cửa sổ tự tương quan cao mà không thêm thông tin.
AFDB_WINDOW_S = config.WINDOW_S
AFDB_STEP_S = 30

# Ngưỡng nhãn: >= 80% thời lượng cửa sổ thuộc 1 rhythm mới nhận nhãn đó;
# đoạn chuyển tiếp hỗn hợp bị bỏ.
LABEL_PURITY = 0.8

FEATURES_AFDB_FILE = os.path.join(config.FEATURES_DIR, 'afdb_features_v4.csv')


def _rhythm_label(aux_note):
    """Map aux_note của AFDB -> nhãn. 1=AFib/AFL, 0=Normal, None=loại."""
    aux = aux_note.strip().upper()
    if 'AFIB' in aux or 'AFL' in aux:
        return 1
    if aux.startswith('(N'):
        return 0
    return None  # (J - junctional rhythm và các loại khác: loại khỏi bài toán


def load_record_annotations(rec_name):
    """Tải annotation QRS + rhythm của 1 record từ PhysioNet.

    Trả về (beat_times_s, segments) với segments = [(t_start, t_end, label)].
    """
    import wfdb

    qrs = wfdb.rdann(rec_name, 'qrs', pn_dir=AFDB_DB)
    atr = wfdb.rdann(rec_name, 'atr', pn_dir=AFDB_DB)

    beat_times = np.asarray(qrs.sample, dtype=np.float64) / AFDB_FS

    # Rhythm segments từ atr.aux_note
    segments = []
    starts, labels = [], []
    for sample, aux in zip(atr.sample, atr.aux_note):
        if not aux:
            continue
        label = _rhythm_label(aux)
        starts.append(sample / AFDB_FS)
        labels.append(label)

    record_end = beat_times[-1] if len(beat_times) else 0.0
    for i, (t0, lab) in enumerate(zip(starts, labels)):
        t1 = starts[i + 1] if i + 1 < len(starts) else record_end
        if lab is not None and t1 > t0:
            segments.append((t0, t1, lab))
    return beat_times, segments


def _window_label(segments, start, stop):
    """Nhãn của cửa sổ [start, stop): theo tỉ lệ thời lượng phủ."""
    dur = stop - start
    af_time = 0.0
    normal_time = 0.0
    for t0, t1, lab in segments:
        overlap = max(0.0, min(stop, t1) - max(start, t0))
        if overlap <= 0:
            continue
        if lab == 1:
            af_time += overlap
        else:
            normal_time += overlap

    if af_time >= LABEL_PURITY * dur:
        return 1
    if normal_time >= LABEL_PURITY * dur:
        return 0
    return None


def extract_record(rec_name, verbose=True):
    """Trích đặc trưng cửa sổ trượt cho 1 record AFDB."""
    beat_times, segments = load_record_annotations(rec_name)
    if len(beat_times) < 2 or not segments:
        return []

    nn_ms, nn_times = beats_to_nn(beat_times)

    rows = []
    t_end = beat_times[-1]
    start = 0.0
    while start + AFDB_WINDOW_S <= t_end:
        stop = start + AFDB_WINDOW_S
        label = _window_label(segments, start, stop)
        if label is not None:
            mask = (nn_times >= start) & (nn_times < stop)
            nn_win = nn_ms[mask]
            nn_t_win = nn_times[mask]
            if len(nn_win) >= config.MIN_BEATS_PER_WINDOW:
                feats = compute_hrv_features(nn_win, nn_t_win)
                feats['record_id'] = f'afdb_{rec_name}'
                feats['t_start'] = start
                feats['status'] = label
                rows.append(feats)
        start += AFDB_STEP_S

    if verbose:
        n_af = sum(1 for r in rows if r['status'] == 1)
        print(f'    {rec_name}: {len(rows)} cửa sổ ({n_af} AFib / {len(rows) - n_af} Normal)')
    return rows


def extract_all(verbose=True):
    """Trích toàn bộ AFDB -> DataFrame và lưu FEATURES_AFDB_FILE."""
    import wfdb

    record_list = wfdb.get_record_list(AFDB_DB)
    if verbose:
        print(f'  Tìm thấy {len(record_list)} record AFDB trên PhysioNet')

    all_rows = []
    for i, rec in enumerate(record_list, 1):
        if verbose:
            print(f'  [{i:>2}/{len(record_list)}]', end='')
        try:
            all_rows.extend(extract_record(rec, verbose=verbose))
        except Exception as e:
            if verbose:
                print(f'    {rec}: LỖI {e} — bỏ qua')

    df = pd.DataFrame(all_rows)
    cols = ['record_id', 't_start', 'status'] + config.ALL_FEATURES
    df = df[cols]

    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    df.to_csv(FEATURES_AFDB_FILE, index=False)
    if verbose:
        n_af = (df['status'] == 1).sum()
        print(f'\nTổng AFDB: {len(df)} cửa sổ ({n_af} AFib / {len(df) - n_af} Normal) '
              f'từ {df["record_id"].nunique()} bệnh nhân')
        print(f'Đã lưu: {FEATURES_AFDB_FILE}')
    return df


def load_features():
    """Nạp bảng đặc trưng AFDB (trích xuất nếu chưa có)."""
    if not os.path.exists(FEATURES_AFDB_FILE):
        return extract_all()
    return pd.read_csv(FEATURES_AFDB_FILE)
