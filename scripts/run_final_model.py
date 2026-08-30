"""Bước 4 — Mô hình cuối cùng: gộp MIMIC + AFDB (60 bệnh nhân).

Quy trình:
1. Gộp 2 bảng đặc trưng, gắn cột `source` (mimic / afdb).
2. CÂN BẰNG NGUỒN bằng sample weight: mỗi dataset đóng góp tổng trọng số
   bằng nhau (AFDB có 29k cửa sổ vs MIMIC 4k — trộn thô sẽ bị AFDB đè 7:1).
3. Đánh giá bằng POOLED LOSO 60 folds (LeaveOneGroupOut theo record_id,
   IQR train-only từng fold, hyperparameter cố định từ các vòng tuning trước).
4. Chọn model tốt nhất theo ROC-AUC -> train trên TOÀN BỘ 60 bệnh nhân
   -> xuất models/final/healthsense_afib_pipeline.pkl + model_card.json.

Chạy:  python scripts/run_final_model.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import xgboost as xgb

from healthsense_ml import config, afdb
from healthsense_ml.feature_extraction import load_features as load_mimic
from healthsense_ml.training import iqr_train_mask

OUTPUT_DIR = os.path.join(config.MODELS_DIR, 'final')

# Hyperparameter cố định — chọn từ các cấu hình thắng phổ biến nhất
# trong benchmark LOSO v4 và cross-dataset (tránh nested tuning 60 folds
# để thời gian chạy hợp lý; các grid trước cho thấy kết quả ít nhạy cảm).
def build_models():
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=2000,
                                       random_state=config.RANDOM_STATE)),
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=150, max_depth=10,
                                           random_state=config.RANDOM_STATE,
                                           n_jobs=-1)),
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', xgb.XGBClassifier(max_depth=4, learning_rate=0.05,
                                      n_estimators=150, eval_metric='logloss',
                                      random_state=config.RANDOM_STATE,
                                      verbosity=0)),
        ]),
    }


def source_weights(sources):
    """Trọng số cân bằng nguồn: mỗi dataset đóng góp tổng weight bằng nhau,
    chuẩn hóa để trung bình = 1."""
    sources = np.asarray(sources)
    n = len(sources)
    weights = np.empty(n, dtype=float)
    uniq = np.unique(sources)
    for s in uniq:
        mask = sources == s
        weights[mask] = n / (len(uniq) * mask.sum())
    return weights


def load_pooled():
    mimic = load_mimic()
    mimic['source'] = 'mimic'
    af = afdb.load_features()
    af['source'] = 'afdb'
    pooled = pd.concat([mimic, af], ignore_index=True)
    return pooled


def pooled_loso(pooled, feature_cols):
    X = pooled[feature_cols]
    y = pooled['status'].to_numpy()
    groups = pooled['record_id'].to_numpy()
    sources = pooled['source'].to_numpy()

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    predictions = []

    for model_name, pipe_proto in build_models().items():
        print(f'\n🏋️ {model_name} — Pooled LOSO {n_folds} folds (cân bằng nguồn)...')
        for fold_i, (tr, te) in enumerate(logo.split(X, y, groups)):
            X_tr, y_tr, s_tr = X.iloc[tr], y[tr], sources[tr]
            X_te, y_te = X.iloc[te], y[te]

            keep = iqr_train_mask(X_tr).to_numpy()
            X_tr_f, y_tr_f, s_tr_f = X_tr[keep], y_tr[keep], s_tr[keep]
            w = source_weights(s_tr_f)

            pipe = build_models()[model_name]
            pipe.fit(X_tr_f, y_tr_f, clf__sample_weight=w)

            probs = pipe.predict_proba(X_te)[:, 1]
            test_record = groups[te][0]
            test_source = sources[te][0]
            for prob, true in zip(probs, y_te):
                predictions.append({
                    'record_id': test_record, 'source': test_source,
                    'status': int(true), 'model': model_name,
                    'prob': float(prob), 'pred': int(prob >= 0.5),
                })
            if (fold_i + 1) % 15 == 0:
                print(f'    fold {fold_i + 1}/{n_folds} xong...')

    return pd.DataFrame(predictions)


def metrics_row(p, model, level, subset=''):
    y, yp, pr = p['status'], p['pred'], p['prob']
    cm = confusion_matrix(y, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'Model': model, 'Level': level, 'Subset': subset or 'all',
        'Accuracy': accuracy_score(y, yp),
        'Recall (Sensitivity)': recall_score(y, yp, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'Precision': precision_score(y, yp, zero_division=0),
        'F1-Score': f1_score(y, yp, zero_division=0),
        'ROC-AUC': roc_auc_score(y, pr),
        'FN': int(fn), 'FP': int(fp),
    }


def summarize(predictions):
    rows = []
    for model in predictions['model'].unique():
        p = predictions[predictions['model'] == model]
        rows.append(metrics_row(p, model, 'window', 'pooled'))
        for src in ['mimic', 'afdb']:
            rows.append(metrics_row(p[p['source'] == src], model, 'window', src))
        # Subject-level chỉ cho MIMIC (bệnh nhân AFDB có cả 2 nhãn xen kẽ
        # nên không quy về 1 nhãn/người được)
        pm = p[p['source'] == 'mimic']
        by = pm.groupby('record_id').agg(status=('status', 'first'),
                                         prob=('prob', 'mean')).reset_index()
        by['pred'] = (by['prob'] >= 0.5).astype(int)
        rows.append(metrics_row(by, model, 'subject', 'mimic'))
    return pd.DataFrame(rows)


def main():
    print('=' * 70)
    print('🔬 FINAL MODEL — GỘP MIMIC + AFDB (60 bệnh nhân, cân bằng nguồn)')
    print('=' * 70)

    feature_cols = config.CORE_FEATURES
    pooled = load_pooled()
    n_subj = pooled['record_id'].nunique()
    print(f'\n📊 {len(pooled)} cửa sổ | {n_subj} bệnh nhân '
          f'(MIMIC: {pooled[pooled.source=="mimic"].record_id.nunique()}, '
          f'AFDB: {pooled[pooled.source=="afdb"].record_id.nunique()})')

    # 1. Pooled LOSO
    predictions = pooled_loso(pooled, feature_cols)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions.to_csv(os.path.join(OUTPUT_DIR, 'pooled_loso_predictions.csv'), index=False)

    summary = summarize(predictions)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'pooled_loso_results.csv'), index=False)

    print('\n' + '=' * 70)
    print('📋 KẾT QUẢ POOLED LOSO (60 bệnh nhân chưa từng thấy)')
    print('=' * 70)
    cols = ['Model', 'Level', 'Subset', 'Accuracy', 'Recall (Sensitivity)',
            'Specificity', 'F1-Score', 'ROC-AUC', 'FN', 'FP']
    print(summary[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    # 2. Chọn model tốt nhất theo AUC pooled window-level
    pooled_rows = summary[(summary.Level == 'window') & (summary.Subset == 'pooled')]
    best_name = pooled_rows.sort_values('ROC-AUC', ascending=False).iloc[0]['Model']
    print(f'\n🏆 Model tốt nhất (pooled ROC-AUC): {best_name}')

    # 3. Train final trên TOÀN BỘ dữ liệu
    X = pooled[feature_cols]
    y = pooled['status'].to_numpy()
    keep = iqr_train_mask(X).to_numpy()
    X_f, y_f = X[keep], y[keep]
    w = source_weights(pooled['source'].to_numpy()[keep])

    final_pipe = build_models()[best_name]
    final_pipe.fit(X_f, y_f, clf__sample_weight=w)

    model_path = os.path.join(OUTPUT_DIR, 'healthsense_afib_pipeline.pkl')
    joblib.dump(final_pipe, model_path)

    best_metrics = pooled_rows[pooled_rows.Model == best_name].iloc[0]
    card = {
        'name': 'HealthSense AFib Detector',
        'version': '4.1.0',
        'model': best_name,
        'trained_on': '60 patients: MIMIC PERform AF (35, PPG 125Hz) + MIT-BIH AFDB (25, ECG 250Hz)',
        'training_windows': int(len(X_f)),
        'features': feature_cols,
        'input': 'Cửa sổ 30s: chuỗi NN (ms) -> 13 đặc trưng HRV theo thứ tự `features`',
        'output': 'predict_proba[:, 1] = P(AFib); ngưỡng mặc định 0.5',
        'source_balancing': 'sample weight — mỗi dataset đóng góp tổng trọng số bằng nhau',
        'evaluation': {
            'pooled_loso_window': {
                'accuracy': round(float(best_metrics['Accuracy']), 4),
                'recall': round(float(best_metrics['Recall (Sensitivity)']), 4),
                'specificity': round(float(best_metrics['Specificity']), 4),
                'roc_auc': round(float(best_metrics['ROC-AUC']), 4),
            },
            'mimic_loso_subject_v4': {'accuracy': 0.9429, 'recall': 1.0, 'roc_auc': 0.9309},
            'cross_dataset': {'mimic_to_afdb_auc': 0.9870, 'afdb_to_mimic_auc': 0.9757},
        },
        'limitations': 'Chưa kiểm định trên PPG cổ tay (MAX30102) và dữ liệu ngoài bệnh viện. '
                       'Không phải thiết bị chẩn đoán y tế.',
    }
    with open(os.path.join(OUTPUT_DIR, 'model_card.json'), 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    print(f'\n💾 Đã xuất: {model_path}')
    print(f'💾 Model card: {os.path.join(OUTPUT_DIR, "model_card.json")}')
    print('\n✅ HOÀN TẤT FINAL MODEL!')


if __name__ == '__main__':
    main()
