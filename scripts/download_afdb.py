"""
Script tải MIT-BIH AF Database (AFDB) từ PhysioNet và trích xuất 16 đặc trưng HRV.
- 25 bệnh nhân riêng biệt (mỗi người có cả đoạn Normal lẫn AFib xen kẽ)
- Tần số lấy mẫu: 250 Hz
- Kết quả: data/features/afdb_features.csv (kèm cột patient_id)
"""

import os, sys, numpy as np, pandas as pd, wfdb, warnings
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings('ignore')

# ===== Cấu hình =====
AFDB_DB_NAME = 'afdb'
AFDB_SAMPLING_RATE = 250  # Hz
WINDOW_SEC = 30
WINDOW_SIZE = AFDB_SAMPLING_RATE * WINDOW_SEC  # 7500 samples per window
STEP_SEC = 10  # Bước trượt 10s (giống MIMIC mặc định)
STEP_SIZE = AFDB_SAMPLING_RATE * STEP_SEC  # 2500 samples

# Thư mục output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, '..', 'data', 'features')
RAW_AFDB_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw', 'afdb')
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RAW_AFDB_DIR, exist_ok=True)

# ===== Hàm trích xuất HRV (giống hệt MIMIC pipeline) =====
def extract_hrv_from_window(signal, sampling_rate=250):
    min_dist = int(sampling_rate * 0.4)
    height_thresh = np.mean(signal) + 0.3 * np.std(signal)
    peaks, _ = find_peaks(signal, distance=min_dist, height=height_thresh)
    
    rr_intervals = np.diff(peaks) / float(sampling_rate)
    if len(rr_intervals) < 3:
        return None
    
    rr_ms = rr_intervals * 1000.0
    diff_rr = np.diff(rr_ms)
    
    mean_nn = np.mean(rr_ms)
    hr_mean = 60000.0 / mean_nn if mean_nn > 0 else 0.0
    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0.0
    nn50 = int(np.sum(np.abs(diff_rr) > 50))
    pnn50 = float((nn50 / len(diff_rr)) * 100.0) if len(diff_rr) > 0 else 0.0
    cv = sdnn / mean_nn if mean_nn > 0 else 0.0
    
    # Phổ LF/HF (nội suy 4Hz)
    time_rr = np.cumsum(rr_intervals)
    time_4hz = np.arange(time_rr[0], time_rr[-1], 0.25)
    if len(time_4hz) > 8:
        rr_4hz = np.interp(time_4hz, time_rr, rr_ms)
        mean_4hz = np.mean(rr_4hz)
        N = len(rr_4hz)
        yf = (np.abs(rfft(rr_4hz - mean_4hz))**2) / N
        xf = rfftfreq(N, 0.25)
        
        lf_band = (xf >= 0.04) & (xf < 0.15)
        hf_band = (xf >= 0.15) & (xf < 0.40)
        
        lf = float(np.sum(yf[lf_band])) if np.any(lf_band) else 0.0
        hf = float(np.sum(yf[hf_band])) if np.any(hf_band) else 0.0
        total_power = float(np.sum(yf))
        lf_hf_ratio = float(lf / hf) if hf > 0 else 0.0
        lf_norm = float((lf / (lf + hf + 1e-6)) * 100.0)
        hf_norm = float((hf / (lf + hf + 1e-6)) * 100.0)
    else:
        lf = hf = total_power = lf_hf_ratio = lf_norm = hf_norm = 0.0
    
    # Poincaré
    sd1 = np.sqrt(0.5 * np.var(diff_rr)) if len(diff_rr) > 0 else 0.0
    sd2 = np.sqrt(max(0, 2 * np.var(rr_ms) - 0.5 * np.var(diff_rr))) if len(diff_rr) > 0 else 0.0
    samp_en = float(np.std(diff_rr) / (sdnn + 1e-6))
    
    return {
        'HR_mean': hr_mean, 'Mean_NN': mean_nn, 'SDNN': sdnn, 'RMSSD': rmssd,
        'NN50': nn50, 'pNN50': pnn50, 'CV': cv,
        'LF': lf, 'HF': hf, 'Total_Power': total_power,
        'LF_HF_Ratio': lf_hf_ratio, 'LF_norm': lf_norm, 'HF_norm': hf_norm,
        'SD1': sd1, 'SD2': sd2, 'SampEn': samp_en
    }


# ===== Bước 1: Lấy danh sách record từ PhysioNet AFDB =====
print("=" * 70)
print("🏥 MIT-BIH Atrial Fibrillation Database (AFDB) - 25 Bệnh Nhân")
print("=" * 70)

print("\n📥 Bước 1: Đang lấy danh sách bệnh nhân từ PhysioNet...")
record_list = wfdb.get_record_list(AFDB_DB_NAME)
print(f"   ✅ Tìm thấy {len(record_list)} bản ghi: {record_list}")

# ===== Bước 2: Tải và trích xuất từng bệnh nhân =====
print(f"\n🔬 Bước 2: Tải tín hiệu & annotation cho từng bệnh nhân...")

all_records = []
patient_stats = []

for idx, rec_name in enumerate(record_list):
    print(f"\n--- [{idx+1}/{len(record_list)}] Bệnh nhân: {rec_name} ---")
    
    try:
        # Tải tín hiệu ECG
        record = wfdb.rdrecord(rec_name, pn_dir=AFDB_DB_NAME)
        signal = record.p_signal[:, 0]  # Lấy kênh ECG đầu tiên
        fs = record.fs  # Sampling rate (250 Hz)
        total_sec = len(signal) / fs
        
        print(f"   📡 Tần số: {fs} Hz | Thời lượng: {total_sec/3600:.1f} giờ | Tổng điểm: {len(signal):,}")
        
        # Tải annotation nhịp (rhythm annotation)
        try:
            ann = wfdb.rdann(rec_name, 'atr', pn_dir=AFDB_DB_NAME)
        except Exception:
            # Thử dùng 'qrs' nếu 'atr' không có
            try:
                ann = wfdb.rdann(rec_name, 'qrs', pn_dir=AFDB_DB_NAME)
            except Exception:
                print(f"   ⚠️ Không tải được annotation cho {rec_name}, bỏ qua.")
                continue
        
        # Xây dựng mảng nhãn cho từng sample
        # AFDB annotation: (AFIB = Rung nhĩ, N = Normal/Sinus, ... )
        sample_labels = np.full(len(signal), -1, dtype=int)  # -1 = unknown
        
        # Duyệt qua các annotation để gán nhãn cho từng đoạn
        ann_samples = ann.sample
        ann_symbols = ann.symbol if hasattr(ann, 'symbol') else []
        ann_aux = ann.aux_note if hasattr(ann, 'aux_note') else []
        
        # Phân tích rhythm annotations
        rhythm_starts = []
        for i in range(len(ann_samples)):
            aux = ann_aux[i].strip() if i < len(ann_aux) else ''
            sym = ann_symbols[i] if i < len(ann_symbols) else ''
            
            if aux:
                # Xác định nhãn từ aux_note
                aux_upper = aux.upper()
                if 'AFIB' in aux_upper or 'AF' == aux_upper:
                    rhythm_starts.append((ann_samples[i], 1))  # AFib
                elif 'N' == aux_upper or 'NORMAL' in aux_upper or '(N' in aux:
                    rhythm_starts.append((ann_samples[i], 0))  # Normal
                elif 'AFL' in aux_upper:  # Atrial Flutter - cũng là bất thường nhĩ
                    rhythm_starts.append((ann_samples[i], 1))  # Gộp với AFib
                elif aux.startswith('('):
                    # Các rhythm annotation khác
                    if '(AFIB' in aux or '(AF' in aux:
                        rhythm_starts.append((ann_samples[i], 1))
                    elif '(N' in aux:
                        rhythm_starts.append((ann_samples[i], 0))
        
        if len(rhythm_starts) == 0:
            print(f"   ⚠️ Không tìm thấy rhythm annotation cho {rec_name}, bỏ qua.")
            continue
        
        # Gán nhãn cho từng đoạn dựa trên rhythm annotation
        for i in range(len(rhythm_starts)):
            start_sample = rhythm_starts[i][0]
            label = rhythm_starts[i][1]
            end_sample = rhythm_starts[i+1][0] if i+1 < len(rhythm_starts) else len(signal)
            sample_labels[start_sample:end_sample] = label
        
        # Đếm phân bố nhãn
        n_normal = np.sum(sample_labels == 0)
        n_afib = np.sum(sample_labels == 1)
        n_unknown = np.sum(sample_labels == -1)
        print(f"   📊 Normal: {n_normal/fs:.0f}s | AFib: {n_afib/fs:.0f}s | Unknown: {n_unknown/fs:.0f}s")
        
        if n_normal + n_afib == 0:
            print(f"   ⚠️ Không có dữ liệu có nhãn, bỏ qua.")
            continue
        
        # Trích xuất HRV cho từng cửa sổ 30s (bước trượt 10s)
        n_windows = 0
        n_normal_windows = 0
        n_afib_windows = 0
        
        for start in range(0, len(signal) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            window_labels = sample_labels[start:end]
            
            # Bỏ qua cửa sổ có nhãn unknown
            known_mask = window_labels >= 0
            if np.sum(known_mask) < 0.8 * WINDOW_SIZE:
                continue
            
            # Xác định nhãn chính (>80% cùng loại mới dùng)
            known_labels = window_labels[known_mask]
            ratio_afib = np.mean(known_labels == 1)
            
            if ratio_afib > 0.8:
                status = 1  # AFib
            elif ratio_afib < 0.2:
                status = 0  # Normal
            else:
                continue  # Bỏ qua đoạn hỗn hợp (transition zone)
            
            # Trích xuất HRV
            window_signal = signal[start:end]
            feat = extract_hrv_from_window(window_signal, sampling_rate=fs)
            
            if feat is not None:
                feat['status'] = status
                feat['patient_id'] = rec_name
                all_records.append(feat)
                n_windows += 1
                if status == 0:
                    n_normal_windows += 1
                else:
                    n_afib_windows += 1
        
        print(f"   🎯 Trích xuất: {n_windows} mẫu (Normal: {n_normal_windows}, AFib: {n_afib_windows})")
        patient_stats.append({
            'patient_id': rec_name,
            'total_seconds': total_sec,
            'normal_windows': n_normal_windows,
            'afib_windows': n_afib_windows,
            'total_windows': n_windows
        })
        
    except Exception as e:
        print(f"   ❌ Lỗi xử lý {rec_name}: {e}")
        continue

# ===== Bước 3: Lưu kết quả =====
print("\n" + "=" * 70)
print("💾 Bước 3: Lưu kết quả...")

if len(all_records) > 0:
    df_afdb = pd.DataFrame(all_records)
    
    # Lưu file chính (kèm patient_id)
    out_path = os.path.join(FEATURES_DIR, 'afdb_features.csv')
    df_afdb.to_csv(out_path, index=False)
    
    print(f"\n✅ Đã lưu: {out_path}")
    print(f"   Tổng mẫu: {len(df_afdb):,}")
    print(f"   Normal (status=0): {len(df_afdb[df_afdb.status==0]):,}")
    print(f"   AFib (status=1): {len(df_afdb[df_afdb.status==1]):,}")
    print(f"   Số bệnh nhân: {df_afdb['patient_id'].nunique()}")
    
    # Thống kê theo bệnh nhân
    print("\n📊 Thống kê từng bệnh nhân:")
    df_stats = pd.DataFrame(patient_stats)
    print(df_stats.to_string(index=False))
else:
    print("❌ Không trích xuất được mẫu nào!")

print("\n🏁 HOÀN TẤT!")
