"""Bước 3 — Cross-Dataset Validation: MIMIC PERform (PPG) <-> MIT-BIH AFDB (ECG).

Bài kiểm tra tổng quát hóa khắc nghiệt nhất: train trên dataset này,
test trên dataset kia — khác bệnh viện, khác loại cảm biến (PPG vs ECG),
khác quần thể bệnh nhân. Không có bất kỳ sự giao thoa dữ liệu nào.

- Tiền xử lý (IQR + Scaler) fit CHỈ trên dataset train.
- Tuning bằng GridSearchCV + GroupKFold(3) theo bệnh nhân của dataset train.
- Test: toàn bộ dataset còn lại, nguyên vẹn.

Chạy:  python scripts/run_cross_dataset.py
       (tự tải & trích xuất AFDB nếu chưa có — chỉ tải annotation, nhanh)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from healthsense_ml import config, afdb
from healthsense_ml.feature_extraction import load_features as load_mimic
from healthsense_ml.training import build_model_registry, iqr_train_mask

OUTPUT_DIR = os.path.join(config.MODELS_DIR, 'cross_dataset')


def evaluate_direction(train_df, test_df, train_name, test_name, feature_cols):
    """Train trên train_df, test trên test_df. Trả về list metrics."""
    X_tr = train_df[feature_cols]
    y_tr = train_df['status'].to_numpy()
    g_tr = train_df['record_id'].to_numpy()
    X_te = test_df[feature_cols]
    y_te = test_df['status'].to_numpy()

    # Lọc outlier chỉ trên train
    keep = iqr_train_mask(X_tr)
    X_tr_f = X_tr[keep]
    y_tr_f = y_tr[keep.to_numpy()]
    g_tr_f = g_tr[keep.to_numpy()]

    rows = []
    preds_all = {}
    for model_name, spec in build_model_registry().items():
        print(f'  🏋️ {model_name}: train {train_name} ({len(X_tr_f)} cửa sổ) '
              f'-> test {test_name} ({len(X_te)} cửa sổ)...')
        search = GridSearchCV(
            spec['pipeline'], spec['param_grid'],
            cv=GroupKFold(config.INNER_CV_FOLDS),
            scoring='roc_auc', n_jobs=-1, refit=True)
        search.fit(X_tr_f, y_tr_f, groups=g_tr_f)

        probs = search.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        preds_all[model_name] = probs

        cm = confusion_matrix(y_te, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            'Direction': f'{train_name} -> {test_name}',
            'Model': model_name,
            'Accuracy': accuracy_score(y_te, preds),
            'Recall (Sensitivity)': recall_score(y_te, preds, zero_division=0),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'Precision': precision_score(y_te, preds, zero_division=0),
            'F1-Score': f1_score(y_te, preds, zero_division=0),
            'ROC-AUC': roc_auc_score(y_te, probs),
            'FN': int(fn), 'FP': int(fp),
            'Best Params': str(search.best_params_),
        })
    return rows


def main():
    print('=' * 70)
    print('🔬 CROSS-DATASET VALIDATION — MIMIC PERform (PPG) <-> MIT-BIH AFDB (ECG)')
    print('=' * 70)

    feature_cols = config.CORE_FEATURES

    print('\n📂 Nạp MIMIC PERform...')
    mimic = load_mimic()
    print(f'   {len(mimic)} cửa sổ, {mimic.record_id.nunique()} bệnh nhân')

    print('\n📂 Nạp MIT-BIH AFDB (tải annotation từ PhysioNet nếu chưa có)...')
    afdb_df = afdb.load_features()
    print(f'   {len(afdb_df)} cửa sổ, {afdb_df.record_id.nunique()} bệnh nhân')

    all_rows = []
    print('\n' + '=' * 70)
    print('HƯỚNG 1: Train MIMIC (PPG) -> Test AFDB (ECG)')
    print('=' * 70)
    all_rows += evaluate_direction(mimic, afdb_df, 'MIMIC', 'AFDB', feature_cols)

    print('\n' + '=' * 70)
    print('HƯỚNG 2: Train AFDB (ECG) -> Test MIMIC (PPG)')
    print('=' * 70)
    all_rows += evaluate_direction(afdb_df, mimic, 'AFDB', 'MIMIC', feature_cols)

    results = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, 'cross_dataset_results.csv')
    results.to_csv(out_csv, index=False)

    print('\n' + '=' * 70)
    print('📋 KẾT QUẢ CROSS-DATASET (test trên dataset mô hình CHƯA TỪNG THẤY)')
    print('=' * 70)
    cols = ['Direction', 'Model', 'Accuracy', 'Recall (Sensitivity)',
            'Specificity', 'F1-Score', 'ROC-AUC', 'FN', 'FP']
    print(results[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\n💾 Đã lưu: {out_csv}')
    print('\n✅ HOÀN TẤT CROSS-DATASET!')


if __name__ == '__main__':
    main()
