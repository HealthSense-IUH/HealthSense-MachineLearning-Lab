"""Cấu hình trung tâm cho toàn bộ pipeline HealthSense ML."""

import os

# ============================================================
# Đường dẫn
# ============================================================
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw', 'mimic_perform')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
# Hai thư mục tách bạch:
#   models/  = thứ NẠP ĐƯỢC vào chương trình (.pkl + thẻ model đi kèm)
#   results/ = thứ ĐEM ĐI BÁO CÁO (số liệu, biểu đồ)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
BENCHMARK_V4_DIR = os.path.join(RESULTS_DIR, 'benchmark_v4')

# File đặc trưng v4 (CÓ cột record_id — bắt buộc cho chia theo bệnh nhân)
FEATURES_V4_FILE = os.path.join(FEATURES_DIR, 'mimic_features_v4.csv')

# ============================================================
# Tín hiệu PPG (MIMIC PERform: 125 Hz, chuẩn hóa 0-1)
# ============================================================
FS = 125                    # Tần số lấy mẫu (Hz)
BANDPASS_LOW = 0.5          # Hz — cắt baseline wander
BANDPASS_HIGH = 8.0         # Hz — cắt nhiễu cao tần, giữ hài bậc cao của PPG
BANDPASS_ORDER = 4

# Phát hiện nhịp
MIN_BEAT_DISTANCE_S = 0.27  # ~220 BPM tối đa
PEAK_PROMINENCE_Z = 0.5     # prominence tối thiểu trên tín hiệu đã z-score

# Giới hạn sinh lý của khoảng NN (ms). KHÔNG lọc quá tay —
# với AFib, sự bất thường của NN chính là tín hiệu cần giữ lại.
NN_MIN_MS = 250
NN_MAX_MS = 2000

# ============================================================
# Cửa sổ trượt
# ============================================================
WINDOW_S = 30               # độ dài cửa sổ (giây)
STEP_S = 10                 # bước trượt (giây)
MIN_BEATS_PER_WINDOW = 10   # dưới ngưỡng này -> bỏ cửa sổ (tín hiệu quá xấu)

# ============================================================
# Đặc trưng
# ============================================================
ALL_FEATURES = [
    'HR_mean', 'Mean_NN', 'SDNN', 'RMSSD', 'NN50', 'pNN50', 'CV',
    'LF', 'HF', 'Total_Power', 'LF_HF_Ratio', 'LF_norm', 'HF_norm',
    'SD1', 'SD2', 'SampEn',
]

# Nhóm LF không đáng tin trên cửa sổ 30s (Task Force 1996 yêu cầu >= 2 phút
# cho LF). Mặc định v4 loại các cột này khỏi huấn luyện.
UNRELIABLE_30S_FEATURES = ['LF', 'LF_norm', 'LF_HF_Ratio']
CORE_FEATURES = [f for f in ALL_FEATURES if f not in UNRELIABLE_30S_FEATURES]

# ============================================================
# Huấn luyện
# ============================================================
RANDOM_STATE = 42
IQR_MULTIPLIER = 3.0        # Lọc outlier CHỈ TRÊN TRAIN. Dùng 3.0 (extreme)
                            # thay vì 1.5 để không vứt nhầm cửa sổ AFib
                            # (AFib vốn dĩ có HRV "outlier" so với Normal).
INNER_CV_FOLDS = 3          # GroupKFold cho tuning bên trong mỗi fold LOSO
