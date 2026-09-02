"""Benchmark v4 — Leave-One-Subject-Out (LOSO), không data leakage.

Ba nguyên tắc sửa lỗi so với v3:

1. CHIA THEO BỆNH NHÂN (LeaveOneGroupOut trên record_id):
   toàn bộ cửa sổ của bệnh nhân test không bao giờ xuất hiện trong train.
   Đây là chuẩn vàng khi số bệnh nhân nhỏ (35 người).

2. TIỀN XỬ LÝ CHỈ FIT TRÊN TRAIN:
   - StandardScaler nằm TRONG sklearn Pipeline -> fit lại theo từng fold.
   - Lọc outlier IQR: tính ngưỡng trên train, chỉ loại hàng TRAIN;
     test giữ nguyên 100% (đời thực không được vứt mẫu khó của bệnh nhân mới).

3. TUNING LỒNG TRONG TỪNG FOLD (nested CV):
   GridSearchCV với GroupKFold(3) chạy bên trong mỗi fold LOSO,
   scoring = ROC-AUC. Test set của fold không tham gia chọn hyperparameter.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV

from . import config


def build_model_registry(random_state=config.RANDOM_STATE):
    """3 mô hình + lưới hyperparameter. Scaler nằm trong Pipeline."""
    import xgboost as xgb

    return {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=2000, solver='lbfgs',
                                           random_state=random_state)),
            ]),
            'param_grid': {'clf__C': [0.01, 0.1, 1.0, 10.0]},
        },
        'Random Forest': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=random_state,
                                               n_jobs=-1)),
            ]),
            'param_grid': {
                'clf__n_estimators': [100, 150],
                'clf__max_depth': [5, 7, 10],
            },
        },
        'XGBoost': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', xgb.XGBClassifier(random_state=random_state,
                                          eval_metric='logloss',
                                          verbosity=0)),
            ]),
            'param_grid': {
                'clf__max_depth': [3, 4, 5],
                'clf__learning_rate': [0.05, 0.1],
                'clf__n_estimators': [150],
            },
        },
    }


def iqr_train_mask(X_train, multiplier=config.IQR_MULTIPLIER):
    """Mask giữ lại các hàng train trong [Q1 - k*IQR, Q3 + k*IQR].

    Ngưỡng tính từ CHÍNH train fold -> không rò rỉ thông tin test.
    """
    mask = pd.Series(True, index=X_train.index)
    for col in X_train.columns:
        q1, q3 = X_train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= X_train[col].between(q1 - multiplier * iqr, q3 + multiplier * iqr)
    return mask


def run_loso_benchmark(df, feature_cols=None, verbose=True):
    """Chạy LOSO benchmark cho cả 3 mô hình.

    Trả về DataFrame kết quả dự đoán mức cửa sổ:
    [record_id, status, model, prob, pred]
    """
    if feature_cols is None:
        feature_cols = config.CORE_FEATURES

    X = df[feature_cols]
    y = df['status'].to_numpy()
    groups = df['record_id'].to_numpy()

    registry = build_model_registry()
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)

    predictions = []
    for model_name, spec in registry.items():
        if verbose:
            print(f'\n🏋️ {model_name} — LOSO {n_folds} folds '
                  f'(nested GroupKFold({config.INNER_CV_FOLDS}) tuning)...')

        for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_te, y_te = X.iloc[test_idx], y[test_idx]
            g_tr = groups[train_idx]
            test_record = groups[test_idx][0]

            # Lọc outlier CHỈ trên train
            keep = iqr_train_mask(X_tr)
            X_tr_f, y_tr_f, g_tr_f = X_tr[keep], y_tr[keep.to_numpy()], g_tr[keep.to_numpy()]

            # Nested tuning trong train fold (chia tiếp theo bệnh nhân)
            search = GridSearchCV(
                spec['pipeline'], spec['param_grid'],
                cv=GroupKFold(config.INNER_CV_FOLDS),
                scoring='roc_auc', n_jobs=-1, refit=True)
            search.fit(X_tr_f, y_tr_f, groups=g_tr_f)

            probs = search.predict_proba(X_te)[:, 1]
            for prob, true in zip(probs, y_te):
                predictions.append({
                    'record_id': test_record,
                    'status': int(true),
                    'model': model_name,
                    'prob': float(prob),
                    'pred': int(prob >= 0.5),
                })

            if verbose and (fold_i + 1) % 10 == 0:
                print(f'    fold {fold_i + 1}/{n_folds} xong...')

    return pd.DataFrame(predictions)
