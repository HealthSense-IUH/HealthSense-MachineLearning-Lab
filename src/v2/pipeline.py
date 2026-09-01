"""Pipeline v2 — làm sạch dữ liệu + đặc trưng phi tuyến.

CẤU HÌNH ĐỊNH NGHĨA PHIÊN BẢN NÀY
---------------------------------
  kênh tín hiệu    : PPG  (giống v1)
  cửa sổ           : 30 giây, bước 30 giây (giống v1)
  đặc trưng        : 16 cột = 13 của v1 + SD1, SD2, SampEn   <-- MỚI
  làm sạch         : luật theo nhãn, áp lên TOÀN BỘ dữ liệu  <-- CHỖ SAI MỚI
  chuẩn hóa        : StandardScaler trong Pipeline
  chia dữ liệu     : ngẫu nhiên 80/20                        <-- CHỖ SAI CŨ
  mô hình          : LightGBM có tinh chỉnh

Chạy:  python src/v2/pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from vlab import honest, store
from vlab.extract import ALL_FEATURES, extract_table

VERSION_ID = 'v2'
CHANNEL = 'PPG'
WINDOW_S = 30.0
STEP_S = 30.0
FEATURES = ALL_FEATURES  # đủ 16 cột

ORIGINAL_CLAIM = {
    'cleaned_lightgbm_accuracy': 0.9736,
    'nonlinear_16feat_accuracy': 0.9627,
    'final_fair_eval_accuracy': 0.9710,
    'n_windows_after_cleaning': 1325,
    'source': ('notebook gốc MIMIC_Training 04_mimic_clean_data_training, '
               '07_mimic_nonlinear_features, 11_mimic_fair_model_evaluation'),
}


def label_conditioned_mask(df):
    """Luật "làm sạch nhãn nhiễu" của v2 — trả về mask GIỮ LẠI.

    Nguyên văn ý tưởng cũ:
      - Cửa sổ gắn nhãn Bình thường mà HRV loạn bất thường  -> coi là nhãn sai.
      - Cửa sổ gắn nhãn AFib mà HRV lại quá đều             -> coi là nhãn sai.

    Vấn đề: luật này DÙNG NHÃN để quyết định giữ hay bỏ. Áp lên tập test là
    tự tay xóa những câu khó nhất khỏi đề thi. Chính đây là nguồn của bước
    nhảy 95.9% -> 97.4%.
    """
    noisy_normal = (df['status'] == 0) & (
        (df['SDNN'] > 300) | (df['RMSSD'] > 400) | (df['pNN50'] > 90))
    noisy_afib = (df['status'] == 1) & (df['SDNN'] < 50) & (df['pNN50'] < 20)
    return ~(noisy_normal | noisy_afib)


def make_model():
    """LightGBM cấu hình v2; lùi về XGBoost nếu máy chưa cài LightGBM."""
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=42, eval_metric='logloss', verbosity=0)


def sqi_threshold_sweep(df, thresholds=(1.0, 2.0, 3.0, 5.0, 10.0)):
    """Tái dựng lỗi "chọn ngưỡng bằng điểm test" (notebook 08 cũ).

    Cách làm cũ: với mỗi ngưỡng, lọc dữ liệu rồi huấn luyện lại, đọc accuracy
    trên TEST, chọn ngưỡng cho điểm cao nhất. Sai ở hai tầng:
      1. Ngưỡng là một hyperparameter -> chọn bằng test là nhìn trộm đáp án.
      2. Mỗi ngưỡng lại thay đổi luôn thành phần tập test -> các con số
         thậm chí không so sánh được với nhau.

    Trả về DataFrame để notebook vẽ ra "đường cong cám dỗ".
    """
    rows = []
    for t in thresholds:
        # Đại diện cho luật SQI biên độ: loại cửa sổ có độ tản NN quá lớn
        cv_amp = df['SDNN'] / df['Mean_NN']
        kept = df[cv_amp <= t]
        if kept['status'].nunique() < 2 or len(kept) < 50:
            continue
        res = honest.leaky_random_split(kept, FEATURES, make_model)
        rows.append({
            'Ngưỡng': t,
            'Giữ lại (%)': round(100.0 * len(kept) / len(df), 1),
            'Accuracy test': res['accuracy'],
            'Recall': res['recall'],
        })
    return pd.DataFrame(rows)


def run(verbose=True):
    if verbose:
        print('=' * 70)
        print('v2 — Làm sạch dữ liệu + đặc trưng phi tuyến (16 cột)')
        print('=' * 70)

    df = extract_table(CHANNEL, WINDOW_S, STEP_S, verbose=verbose)

    # --- Cách của bản gốc: áp luật lên TOÀN BỘ dữ liệu trước khi chia ---
    keep = label_conditioned_mask(df)
    df_cleaned = df[keep].reset_index(drop=True)
    n_dropped = int((~keep).sum())
    dropped = df[~keep]

    if verbose:
        print(f'\nLuật làm sạch xóa {n_dropped} cửa sổ '
              f'({n_dropped / len(df) * 100:.1f}%): '
              f'{int((dropped["status"] == 1).sum())} AFib / '
              f'{int((dropped["status"] == 0).sum())} Normal')

    # --- Chấm hai cách: bản gốc (đã lọc toàn cục) vs LOSO (lọc chỉ trên train) ---
    result = honest.compare(
        df, FEATURES, make_model,
        df_leaky=df_cleaned,
        train_filter=lambda d: label_conditioned_mask(d).to_numpy())

    payload = {
        'version': VERSION_ID,
        'title': 'Làm sạch dữ liệu + đặc trưng phi tuyến',
        'config': {
            'channel': CHANNEL,
            'window_s': WINDOW_S,
            'step_s': STEP_S,
            'overlap_pct': 0.0,
            'n_features': len(FEATURES),
            'features': FEATURES,
            'cleaning': 'luật theo nhãn (SDNN/RMSSD/pNN50), áp lên toàn bộ dữ liệu',
            'scaling': 'StandardScaler trong Pipeline',
            'split': 'ngẫu nhiên theo cửa sổ 80/20',
            'model': 'LightGBM (n=200, lr=0.1)',
        },
        'data': {
            'n_windows': int(len(df)),
            'n_windows_after_cleaning': int(len(df_cleaned)),
            'n_dropped': n_dropped,
            'n_dropped_afib': int((dropped['status'] == 1).sum()),
            'n_dropped_normal': int((dropped['status'] == 0).sum()),
            'n_subjects': int(df['record_id'].nunique()),
        },
        'original_claim': ORIGINAL_CLAIM,
        'reproduced_leaky': result['leaky'],
        'regraded_loso_window': result['loso_window'],
        'regraded_loso_subject': {
            k: v for k, v in result['loso_subject'].items()
            if k not in ('subjects', 'subject_proba', 'subject_true')
        },
        'subject_detail': {
            'subjects': result['loso_subject']['subjects'],
            'proba': result['loso_subject']['subject_proba'],
            'true': result['loso_subject']['subject_true'],
        },
        'inflation': result['inflation'],
        'leakage': [
            'Chia ngẫu nhiên theo cửa sổ (kế thừa v1).',
            'Luật làm sạch dùng NHÃN và áp lên cả tập test -> xóa mất các ca khó.',
            'Ngưỡng SQI được chọn bằng điểm test (xem sqi_threshold_sweep).',
        ],
    }
    store.save(VERSION_ID, payload)

    if verbose:
        print('\n' + result['table'].to_string(index=False))
        print(f"\nPhần điểm ảo do leakage: "
              f"{payload['inflation'] * 100:+.1f} điểm phần trăm")
        subj = payload['regraded_loso_subject']
        print(f"Mức bệnh nhân (LOSO): accuracy {subj['accuracy']:.4f} "
              f"trên {subj['n_subjects']} người")
        print(f"\nĐã lưu: {store.results_path(VERSION_ID)}")
    return payload


if __name__ == '__main__':
    run()
