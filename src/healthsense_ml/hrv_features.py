"""16 đặc trưng HRV chuẩn Task Force 1996 tính từ chuỗi NN của 1 cửa sổ.

Nhóm thời gian: HR_mean, Mean_NN, SDNN, RMSSD, NN50, pNN50, CV
Nhóm tần số  : LF, HF, Total_Power, LF_HF_Ratio, LF_norm, HF_norm
               (LƯU Ý: nhóm LF không đáng tin trên cửa sổ 30s — Task Force
               yêu cầu bản ghi >= 2 phút cho LF. v4 vẫn tính đủ 16 cột để
               tương thích, nhưng huấn luyện mặc định loại nhóm LF —
               xem config.UNRELIABLE_30S_FEATURES.)
Nhóm phi tuyến: SD1, SD2 (Poincaré), SampEn (Sample Entropy)
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

# Dải tần chuẩn (Hz)
VLF_LOW, LF_LOW, LF_HIGH, HF_HIGH = 0.0033, 0.04, 0.15, 0.4
RESAMPLE_FS = 4.0  # Hz — tần số nội suy chuỗi NN cho phân tích phổ


def _time_domain(nn):
    mean_nn = np.mean(nn)
    sdnn = np.std(nn, ddof=1)
    diff = np.diff(nn)
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else 0.0
    nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
    pnn50 = nn50 / len(diff) * 100 if len(diff) else 0.0
    return {
        'HR_mean': 60000.0 / mean_nn,
        'Mean_NN': mean_nn,
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'NN50': nn50,
        'pNN50': pnn50,
        'CV': sdnn / mean_nn,
    }


def _frequency_domain(nn, nn_times):
    """Welch periodogram trên chuỗi NN nội suy đều 4 Hz."""
    out = {'LF': 0.0, 'HF': 0.0, 'Total_Power': 0.0,
           'LF_HF_Ratio': 0.0, 'LF_norm': 0.0, 'HF_norm': 0.0}
    if len(nn) < 4:
        return out

    # Nội suy chuỗi NN về lưới thời gian đều
    t_uniform = np.arange(nn_times[0], nn_times[-1], 1.0 / RESAMPLE_FS)
    if len(t_uniform) < 8:
        return out
    interp = interp1d(nn_times, nn, kind='cubic', fill_value='extrapolate')
    nn_uniform = interp(t_uniform)
    nn_uniform = nn_uniform - np.mean(nn_uniform)

    nperseg = min(len(nn_uniform), 128)
    freqs, psd = sp_signal.welch(nn_uniform, fs=RESAMPLE_FS, nperseg=nperseg)

    def band_power(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    lf = band_power(LF_LOW, LF_HIGH)
    hf = band_power(LF_HIGH, HF_HIGH)
    total = band_power(VLF_LOW, HF_HIGH)

    out['LF'] = lf
    out['HF'] = hf
    out['Total_Power'] = total
    out['LF_HF_Ratio'] = lf / hf if hf > 0 else 0.0
    if lf + hf > 0:
        out['LF_norm'] = lf / (lf + hf) * 100
        out['HF_norm'] = hf / (lf + hf) * 100
    return out


def _poincare(nn):
    diff = np.diff(nn)
    if len(diff) == 0:
        return {'SD1': 0.0, 'SD2': 0.0}
    sd_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    sdnn = np.std(nn, ddof=1)
    sd1 = np.sqrt(0.5) * sd_diff
    sd2_sq = 2 * sdnn ** 2 - 0.5 * sd_diff ** 2
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
    return {'SD1': sd1, 'SD2': sd2}


def _sample_entropy(nn, m=2, r_factor=0.2):
    """Sample Entropy (m=2, r=0.2*SD) — độ 'hỗn loạn' của nhịp, rất mạnh
    cho AFib. Cài đặt O(n²) — đủ nhanh với ~40 nhịp/cửa sổ 30s."""
    n = len(nn)
    if n < m + 2:
        return 0.0
    sd = np.std(nn, ddof=1)
    if sd == 0:
        return 0.0
    r = r_factor * sd

    def count_matches(mm):
        templates = np.array([nn[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = count_matches(m)
    a = count_matches(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def compute_hrv_features(nn_ms, nn_times_s):
    """Tính đủ 16 đặc trưng HRV cho 1 cửa sổ. Trả về dict."""
    features = {}
    features.update(_time_domain(nn_ms))
    features.update(_frequency_domain(nn_ms, nn_times_s))
    features.update(_poincare(nn_ms))
    features['SampEn'] = _sample_entropy(nn_ms)
    return features
