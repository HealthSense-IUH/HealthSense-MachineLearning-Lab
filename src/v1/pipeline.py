"""Pipeline v1 — tái dựng đường cơ sở MIMIC PPG.

CẤU HÌNH ĐỊNH NGHĨA PHIÊN BẢN NÀY
---------------------------------
  kênh tín hiệu    : PPG
  cửa sổ           : 30 giây, bước 30 giây  -> KHÔNG chồng lấn
  đặc trưng        : 13 cột tuyến tính (7 thời gian + 6 tần số),
                     chưa có nhóm phi tuyến SD1/SD2/SampEn
  làm sạch         : không có
  chuẩn hóa        : StandardScaler nằm TRONG Pipeline (đúng — không rò rỉ)
  chia dữ liệu     : train_test_split ngẫu nhiên 80/20  <-- CHỖ SAI
  mô hình          : Random Forest (100 cây)

Chạy:  python src/v1/pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier

from vlab import honest, store
from vlab.extract import LINEAR_13, extract_table

VERSION_ID = 'v1'
CHANNEL = 'PPG'
WINDOW_S = 30.0
STEP_S = 30.0          # = WINDOW_S -> các cửa sổ không chồng lên nhau
FEATURES = LINEAR_13   # 13 cột

# Con số phiên bản gốc từng công bố (notebook thí nghiệm cũ, nay chỉ còn
# trong lịch sử git — xem README của repo)
ORIGINAL_CLAIM = {
    'random_forest_accuracy': 0.9517,
    'best_tuned_accuracy': 0.9591,
    'best_tuned_model': 'XGBoost / LightGBM (tuned)',
    'n_windows': 1342,
    'source': 'notebook gốc MIMIC_Training 02_mimic_model_training, 03_mimic_model_comparison',
}


def make_model():
    """Random Forest đúng cấu hình v1."""
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def run(verbose=True):
    """Chạy v1 hai lần trên cùng dữ liệu: cách gốc, rồi cách trung thực."""
    if verbose:
        print('=' * 70)
        print('v1 — Đường cơ sở MIMIC PPG (13 đặc trưng, chia ngẫu nhiên)')
        print('=' * 70)

    df = extract_table(CHANNEL, WINDOW_S, STEP_S, verbose=verbose)

    result = honest.compare(df, FEATURES, make_model)

    payload = {
        'version': VERSION_ID,
        'title': 'Đường cơ sở MIMIC PPG',
        'config': {
            'channel': CHANNEL,
            'window_s': WINDOW_S,
            'step_s': STEP_S,
            'overlap_pct': 0.0,
            'n_features': len(FEATURES),
            'features': FEATURES,
            'cleaning': 'không',
            'scaling': 'StandardScaler trong Pipeline (fit theo train)',
            'split': 'ngẫu nhiên theo cửa sổ 80/20',
            'model': 'Random Forest (100 cây)',
        },
        'data': {
            'n_windows': int(len(df)),
            'n_subjects': int(df['record_id'].nunique()),
            'n_afib_windows': int((df['status'] == 1).sum()),
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
            'Chia ngẫu nhiên theo cửa sổ: bệnh nhân ở test cũng có trong train '
            '(subject leakage).',
        ],
    }
    store.save(VERSION_ID, payload)

    if verbose:
        _print_summary(payload, result)
    return payload


def _print_summary(payload, result):
    leaky = payload['reproduced_leaky']
    loso = payload['regraded_loso_window']
    subj = payload['regraded_loso_subject']
    print(f"\nDữ liệu: {payload['data']['n_windows']} cửa sổ / "
          f"{payload['data']['n_subjects']} bệnh nhân")
    print(f"\nChồng lấn bệnh nhân giữa train và test: "
          f"{leaky.get('n_subjects_also_in_train')}/{leaky.get('n_test_subjects')} "
          f"({leaky.get('subject_overlap_pct')}%)")
    print('\n' + result['table'].to_string(index=False))
    print(f"\nPhần điểm ảo do leakage: {payload['inflation'] * 100:+.1f} điểm phần trăm")
    print(f"Mức bệnh nhân (LOSO): accuracy {subj['accuracy']:.4f} "
          f"trên {subj['n_subjects']} người, recall {subj['recall']:.4f}")
    print(f"\nĐã lưu: {store.results_path(VERSION_ID)}")
    return loso


if __name__ == '__main__':
    run()
