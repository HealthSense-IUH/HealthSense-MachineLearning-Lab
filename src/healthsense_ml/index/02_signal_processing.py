"""Xử lý tín hiệu PPG: lọc bandpass, phát hiện nhịp, trích chuỗi NN.

Đầu vào là sóng PPG thô 125 Hz; đầu ra là chuỗi thời điểm nhịp (beat times)
và chuỗi khoảng NN (ms) — nguyên liệu cho toàn bộ đặc trưng HRV.
"""

import numpy as np
from scipy import signal as sp_signal

from . import config


def bandpass_filter(ppg, fs=config.FS):
    """Lọc Butterworth bandpass 0.5–8 Hz (zero-phase, không lệch pha nhịp)."""
    sos = sp_signal.butter(
        config.BANDPASS_ORDER,
        [config.BANDPASS_LOW, config.BANDPASS_HIGH],
        btype='bandpass', fs=fs, output='sos')
    return sp_signal.sosfiltfilt(sos, ppg)


def detect_beats(ppg_filtered, fs=config.FS):
    """Phát hiện đỉnh tâm thu trên PPG đã lọc. Trả về thời điểm nhịp (giây).

    Prominence tính trên tín hiệu z-score để không phụ thuộc biên độ
    tuyệt đối (biên độ PPG thay đổi theo bệnh nhân và theo áp lực tiếp xúc).
    """
    std = np.std(ppg_filtered)
    if std == 0:
        return np.array([])
    z = (ppg_filtered - np.mean(ppg_filtered)) / std

    peaks, _ = sp_signal.find_peaks(
        z,
        distance=int(config.MIN_BEAT_DISTANCE_S * fs),
        prominence=config.PEAK_PROMINENCE_Z)
    return peaks / fs


def beats_to_nn(beat_times_s):
    """Chuyển thời điểm nhịp -> chuỗi NN (ms), chỉ lọc giới hạn sinh lý.

    Chủ đích KHÔNG lọc theo độ lệch so với median: với AFib, sự bất thường
    của khoảng NN chính là đặc trưng bệnh lý cần giữ lại. Chỉ loại giá trị
    phi sinh lý (< 250 ms hoặc > 2000 ms) do lỗi phát hiện đỉnh.

    Trả về (nn_ms, nn_times_s): khoảng NN và thời điểm kết thúc mỗi khoảng.
    """
    if len(beat_times_s) < 2:
        return np.array([]), np.array([])
    nn = np.diff(beat_times_s) * 1000.0
    nn_times = beat_times_s[1:]
    mask = (nn >= config.NN_MIN_MS) & (nn <= config.NN_MAX_MS)
    return nn[mask], nn_times[mask]


def extract_nn_series(ppg_raw, fs=config.FS):
    """Pipeline gọn: PPG thô -> (nn_ms, nn_times_s)."""
    filtered = bandpass_filter(ppg_raw, fs)
    beats = detect_beats(filtered, fs)
    return beats_to_nn(beats)
