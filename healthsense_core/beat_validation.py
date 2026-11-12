"""Kiểm chứng chất lượng phát hiện nhịp PPG bằng ECG đồng bộ (MIMIC PERform).

Mỗi file MIMIC có cột ECG ghi song song với PPG — dùng R-peak trên ECG làm
"đáp án chuẩn" để chấm điểm bộ dò nhịp PPG của pipeline:

1. Dò R-peak trên ECG (bandpass 5–20 Hz, z-score, find_peaks).
2. Dò nhịp PPG bằng CHÍNH pipeline v4 (signal_processing.detect_beats).
3. Ước lượng PTT (pulse transit time — độ trễ sinh lý PPG so với ECG,
   thường 150–500 ms) bằng median lệch thời gian, rồi khớp từng nhịp
   trong cửa sổ dung sai ±150 ms.
4. Metrics: Sensitivity (nhịp ECG được PPG bắt), PPV (nhịp PPG là thật),
   F1, và MAE nhịp tim theo cửa sổ 30s (HR từ PPG vs HR từ ECG).

ECG quá nhiễu/phẳng (dò được quá ít nhịp hoặc HR phi sinh lý) sẽ bị đánh
dấu `ecg_unreliable` và loại khỏi thống kê tổng.
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from . import config
from .data_loading import list_records
from .signal_processing import bandpass_filter, detect_beats

MATCH_TOL_S = 0.150   # dung sai khớp nhịp sau khi trừ PTT
MAX_PTT_S = 1.0       # tìm PTT trong phạm vi 0..1s sau R-peak


def detect_r_peaks(ecg, fs=config.FS):
    """Dò R-peak trên ECG: bandpass 5-20 Hz -> z-score -> find_peaks."""
    ecg = np.asarray(ecg, dtype=np.float64)
    if np.isnan(ecg).any():
        idx = np.arange(len(ecg))
        good = ~np.isnan(ecg)
        if good.sum() < 10:
            return np.array([])
        ecg = np.interp(idx, idx[good], ecg[good])

    sos = sp_signal.butter(3, [5.0, 20.0], btype='bandpass', fs=fs, output='sos')
    filt = sp_signal.sosfiltfilt(sos, ecg)
    std = np.std(filt)
    if std == 0:
        return np.array([])
    z = filt / std
    # R có thể âm tùy cực điện cực -> dò trên |z|
    peaks, _ = sp_signal.find_peaks(np.abs(z), distance=int(0.25 * fs), prominence=1.5)
    return peaks / fs


def match_beats(ecg_t, ppg_t):
    """Khớp nhịp PPG với R-peak ECG sau khi bù PTT.

    Trả về dict: ptt_ms, sensitivity, ppv, f1, n_ecg, n_ppg, n_match.
    """
    if len(ecg_t) < 10 or len(ppg_t) < 10:
        return None

    # Ước lượng PTT: với mỗi nhịp PPG, tìm R-peak gần nhất TRƯỚC nó trong 1s
    idx = np.searchsorted(ecg_t, ppg_t) - 1
    valid = idx >= 0
    deltas = ppg_t[valid] - ecg_t[idx[valid]]
    deltas = deltas[(deltas > 0) & (deltas < MAX_PTT_S)]
    if len(deltas) < 10:
        return None
    ptt = float(np.median(deltas))

    # Dịch PPG về trục thời gian ECG rồi khớp greedy 1-1 trong ±tol
    ppg_shifted = np.sort(ppg_t - ptt)
    used = np.zeros(len(ppg_shifted), dtype=bool)
    n_match = 0
    j = 0
    for t in ecg_t:
        # tiến con trỏ tới ứng viên gần nhất
        while j < len(ppg_shifted) and ppg_shifted[j] < t - MATCH_TOL_S:
            j += 1
        best = -1
        for k in (j, j + 1):
            if k < len(ppg_shifted) and not used[k] and abs(ppg_shifted[k] - t) <= MATCH_TOL_S:
                if best < 0 or abs(ppg_shifted[k] - t) < abs(ppg_shifted[best] - t):
                    best = k
        if best >= 0:
            used[best] = True
            n_match += 1

    sens = n_match / len(ecg_t)
    ppv = n_match / len(ppg_shifted)
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    return {
        'ptt_ms': ptt * 1000.0,
        'sensitivity': sens,
        'ppv': ppv,
        'f1': f1,
        'n_ecg_beats': len(ecg_t),
        'n_ppg_beats': len(ppg_shifted),
        'n_matched': n_match,
    }


def window_hr_mae(ecg_t, ppg_t, t_end, window_s=30.0):
    """MAE nhịp tim theo cửa sổ 30s: HR đếm từ ECG vs từ PPG."""
    errs = []
    start = 0.0
    while start + window_s <= t_end:
        stop = start + window_s
        n_e = np.sum((ecg_t >= start) & (ecg_t < stop))
        n_p = np.sum((ppg_t >= start) & (ppg_t < stop))
        if n_e >= 10 and n_p >= 10:
            errs.append(abs(n_e - n_p) * (60.0 / window_s))
        start += window_s
    return float(np.mean(errs)) if errs else np.nan


def ecg_rhythm_stats(ecg_t):
    """RMSSD/pNN50 tính từ chính ECG — dùng thẩm định nhãn."""
    rr = np.diff(ecg_t) * 1000.0
    rr = rr[(rr >= config.NN_MIN_MS) & (rr <= config.NN_MAX_MS)]
    if len(rr) < 10:
        return np.nan, np.nan
    diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    pnn50 = float(np.mean(np.abs(diff) > 50) * 100)
    return rmssd, pnn50


def validate_record(record_id, label, csv_path):
    """Chấm điểm dò nhịp PPG của 1 bệnh nhân."""
    df = pd.read_csv(csv_path)
    if 'ECG' not in df.columns:
        return None
    t = df['Time'].to_numpy(dtype=np.float64)
    ppg = df['PPG'].to_numpy(dtype=np.float64)
    if np.isnan(ppg).any():
        idx = np.arange(len(ppg))
        good = ~np.isnan(ppg)
        ppg = np.interp(idx, idx[good], ppg[good])

    ecg_t = detect_r_peaks(df['ECG'].to_numpy())
    ppg_t = detect_beats(bandpass_filter(ppg))

    row = {'record_id': record_id, 'label': 'AFib' if label else 'Normal'}

    # Cờ chất lượng ECG: quá ít nhịp hoặc HR phi sinh lý -> không chấm được
    dur_min = (t[-1] - t[0]) / 60.0
    hr_ecg = len(ecg_t) / dur_min if dur_min > 0 else 0
    ecg_ok = (len(ecg_t) >= 200) and (30 <= hr_ecg <= 220)
    row['ecg_unreliable'] = not ecg_ok
    row['hr_ecg_mean'] = hr_ecg

    m = match_beats(ecg_t, ppg_t) if ecg_ok else None
    if m:
        row.update(m)
        row['hr_mae_bpm'] = window_hr_mae(ecg_t, ppg_t, t[-1])
    rmssd, pnn50 = ecg_rhythm_stats(ecg_t) if ecg_ok else (np.nan, np.nan)
    row['ecg_RMSSD'] = rmssd
    row['ecg_pNN50'] = pnn50
    return row


def validate_all(verbose=True):
    rows = []
    records = list_records()
    for i, (rid, label, path) in enumerate(records, 1):
        row = validate_record(rid, label, path)
        if row is None:
            continue
        rows.append(row)
        if verbose:
            if row.get('f1') is not None and not row['ecg_unreliable']:
                print(f"  [{i:>2}/{len(records)}] {rid}: F1={row['f1']:.3f} "
                      f"Sens={row['sensitivity']:.3f} PPV={row['ppv']:.3f} "
                      f"PTT={row['ptt_ms']:.0f}ms HR-MAE={row['hr_mae_bpm']:.2f}bpm")
            else:
                print(f"  [{i:>2}/{len(records)}] {rid}: ECG không đủ tin cậy để chấm")
    return pd.DataFrame(rows)
