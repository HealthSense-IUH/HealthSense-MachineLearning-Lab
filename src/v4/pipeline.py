"""Pipeline v4 — phiên bản đang chạy trong sản phẩm.

CẤU HÌNH ĐỊNH NGHĨA PHIÊN BẢN NÀY
---------------------------------
  kênh tín hiệu    : PPG (đúng loại cảm biến của vòng đeo)
  cửa sổ           : 30 giây, bước 10 giây -> chồng lấn 67%
  đặc trưng        : 13 cột = 16 trừ nhóm LF (LF, LF_norm, LF_HF_Ratio)
  làm sạch         : IQR x3.0, ngưỡng tính TRÊN TRAIN, chỉ xóa hàng train
  chuẩn hóa        : StandardScaler trong Pipeline, fit lại theo từng fold
  chia dữ liệu     : LOSO theo record_id (bệnh nhân)
  mô hình          : LR / RF / XGBoost, tinh chỉnh lồng bằng GroupKFold

VÌ SAO v4 VẪN DÙNG CỬA SỔ CHỒNG LẤN MÀ KHÔNG SAO?
Chồng lấn tự nó không phải là leakage. Nó chỉ nguy hiểm khi ĐI KÈM chia
ngẫu nhiên: lúc đó bản sao gần đúng của một cửa sổ test nằm sẵn trong train.
Với LOSO, toàn bộ cửa sổ của một người nằm trọn về một phía, nên dù chúng
chồng lên nhau bao nhiêu cũng không thể rò rỉ qua ranh giới train/test.
Chồng lấn khi đó chỉ còn tác dụng tốt: có thêm mẫu để học.

Chạy:  python src/v4/pipeline.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier

from vlab import export, honest, store
from vlab.extract import CORE_13, extract_table
from vlab.raw import MODELS_DIR

VERSION_ID = 'v4'
CHANNEL = 'PPG'
WINDOW_S = 30.0
STEP_S = 10.0        # chồng lấn 67% — an toàn vì chia theo bệnh nhân
FEATURES = CORE_13   # 13 cột, đã bỏ nhóm LF

IQR_MULTIPLIER = 3.0  # rộng tay hơn v3 (1.5): cửa sổ AFib vốn dĩ "cực đoan",
                      # lọc chặt là tự tay vứt đúng thứ cần học.


def make_model():
    """Random Forest — dùng chung với v1-v3 để so sánh công bằng."""
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def train_only_iqr_mask(df_train, features=FEATURES, multiplier=IQR_MULTIPLIER):
    """Ngưỡng IQR tính từ train của fold; test không bị đụng tới."""
    import pandas as pd
    mask = pd.Series(True, index=df_train.index)
    for col in features:
        q1, q3 = df_train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df_train[col].between(q1 - multiplier * iqr, q3 + multiplier * iqr)
    return mask.to_numpy()


def load_production_results():
    """Nạp kết quả benchmark thật của sản phẩm (models/model_card.json)."""
    path = os.path.join(MODELS_DIR, 'model_card.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def run(verbose=True):
    if verbose:
        print('=' * 70)
        print('v4 — Pipeline hiện hành: LOSO theo bệnh nhân, 13 đặc trưng')
        print('=' * 70)

    df = extract_table(CHANNEL, WINDOW_S, STEP_S, verbose=verbose)

    # Chấm hai cách trên CHÍNH dữ liệu của v4.
    # Mục đích: chứng minh dữ liệu v4 không hề "khó hơn" — nếu chấm kiểu cũ
    # thì v4 cũng cho ~98%. Toàn bộ chênh lệch đến từ CÁCH CHẤM, không phải
    # từ chất lượng dữ liệu hay mô hình.
    result = honest.compare(df, FEATURES, make_model,
                            train_filter=train_only_iqr_mask)

    production = load_production_results()

    payload = {
        'version': VERSION_ID,
        'title': 'Pipeline hiện hành (LOSO, chống leakage)',
        'config': {
            'channel': CHANNEL,
            'window_s': WINDOW_S,
            'step_s': STEP_S,
            'overlap_pct': 66.7,
            'n_features': len(FEATURES),
            'features': FEATURES,
            'cleaning': f'IQR x{IQR_MULTIPLIER}, ngưỡng chỉ tính trên train của fold',
            'scaling': 'StandardScaler trong Pipeline, fit lại theo từng fold',
            'split': 'LOSO theo record_id (bệnh nhân)',
            'model': 'Random Forest (so sánh) / XGBoost tinh chỉnh lồng (sản phẩm)',
        },
        'data': {
            'n_windows': int(len(df)),
            'n_subjects': int(df['record_id'].nunique()),
            'n_afib_windows': int((df['status'] == 1).sum()),
        },
        'if_graded_the_old_way': result['leaky'],
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
        'production': production,
        'leakage': [],  # không còn
        'fixes': [
            'LOSO theo bệnh nhân: người ở test chưa từng xuất hiện trong train.',
            'Scaler fit lại theo từng fold, chỉ nhìn dữ liệu train.',
            'Lọc IQR tính trên train và chỉ xóa hàng train; test giữ nguyên 100%.',
            'Tinh chỉnh hyperparameter lồng trong fold bằng GroupKFold.',
            'Bỏ nhóm LF vì cửa sổ 30 giây quá ngắn để ước lượng dải tần này.',
        ],
    }
    store.save(VERSION_ID, payload)
    export.export(VERSION_ID, df, FEATURES, make_model, payload,
                  train_filter=train_only_iqr_mask,
                  channel=CHANNEL, verbose=verbose)

    if verbose:
        print('\n' + result['table'].to_string(index=False))
        print(f"\nNếu chấm kiểu cũ, chính dữ liệu v4 cũng cho "
              f"{result['leaky']['accuracy'] * 100:.2f}% — chênh lệch nằm ở "
              f"CÁCH CHẤM, không phải ở dữ liệu.")
        subj = payload['regraded_loso_subject']
        print(f"Mức bệnh nhân (LOSO): accuracy {subj['accuracy']:.4f}, "
              f"recall {subj['recall']:.4f} trên {subj['n_subjects']} người")
        if production:
            ev = production['evaluation']
            print(f"\nMô hình sản phẩm ({production['model']}, "
                  f"{production['training_windows']} cửa sổ, 60 bệnh nhân):")
            print(f"  pooled LOSO mức cửa sổ: acc "
                  f"{ev['pooled_loso_window']['accuracy']:.4f}, "
                  f"AUC {ev['pooled_loso_window']['roc_auc']:.4f}")
            print(f"  cross-dataset AUC: MIMIC->AFDB "
                  f"{ev['cross_dataset']['mimic_to_afdb_auc']:.4f}, "
                  f"AFDB->MIMIC {ev['cross_dataset']['afdb_to_mimic_auc']:.4f}")
        print(f"\nĐã lưu: {store.results_path(VERSION_ID)}")
    return payload


if __name__ == '__main__':
    run()
