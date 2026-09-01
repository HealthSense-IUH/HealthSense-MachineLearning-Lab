"""Pipeline v3 — benchmark đa quy mô trên kênh ECG.

CẤU HÌNH ĐỊNH NGHĨA PHIÊN BẢN NÀY
---------------------------------
  kênh tín hiệu    : ECG                                     <-- KHÁC v1/v2/v4
  cửa sổ           : 30 giây; bước 30 / 10 / 5 / 2.5 giây
                     -> chồng lấn 0% / 66% / 83% / 91%       <-- CHỖ SAI MỚI
  đặc trưng        : 16 cột
  làm sạch         : IQR x1.5 tính trên TOÀN BỘ bảng          <-- CHỖ SAI MỚI
  chuẩn hóa        : StandardScaler fit TOÀN BỘ rồi mới chia  <-- CHỖ SAI MỚI
  chia dữ liệu     : ngẫu nhiên 80/20                         <-- CHỖ SAI CŨ
  mô hình          : LR, RF, XGBoost + Stacking

GHI CHÚ VỀ CÁCH SO SÁNH: cột "bản gốc" và cột "LOSO" đều dùng CÙNG một mô
hình (Random Forest) để phần chênh lệch phản ánh đúng cách chia dữ liệu, chứ
không lẫn với việc đổi mô hình. Bảng xếp hạng đầy đủ (LR/RF/XGB/Stacking)
được chạy riêng theo đúng cách cũ để tái hiện con số lịch sử.

Chạy:  python src/v3/pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vlab import export, honest, store
from vlab.extract import ALL_FEATURES, extract_table

VERSION_ID = 'v3'
CHANNEL = 'ECG'
WINDOW_S = 30.0
FEATURES = ALL_FEATURES

# 4 quy mô: bước trượt càng ngắn, các cửa sổ càng chồng lên nhau
SCALES = [
    {'step_s': 30.0, 'overlap_pct': 0.0,  'label': 'không chồng lấn'},
    {'step_s': 10.0, 'overlap_pct': 66.7, 'label': 'chồng 67%'},
    {'step_s': 5.0,  'overlap_pct': 83.3, 'label': 'chồng 83%'},
    {'step_s': 2.5,  'overlap_pct': 91.7, 'label': 'chồng 92%'},
]

IQR_MULTIPLIER = 1.5  # v3 dùng 1.5 (chuẩn "outlier nhẹ") — khá mạnh tay

ORIGINAL_CLAIM = {
    'stacking_4083_accuracy': 0.9865,
    'rf_1360_accuracy': 0.9873,
    'rf_1360_auc': 0.9965,
    'mlp_8165_accuracy': 0.9871,
    'source': ('notebook gốc v3_pipeline/mimic 02_model_training_and_evaluation, '
               '04_mimic_v3_benchmark'),
}


def make_model():
    """Random Forest — mô hình dùng cho phép so sánh hai cách chấm."""
    return RandomForestClassifier(n_estimators=100, max_depth=10,
                                  random_state=42, n_jobs=-1)


def make_stacking():
    """Stacking đúng tinh thần v3: nhiều mô hình nền + LR làm trọng tài."""
    import xgboost as xgb
    return StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=2000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=5,
                                      learning_rate=0.1, random_state=42,
                                      eval_metric='logloss', verbosity=0)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
        cv=3, n_jobs=-1)


def global_iqr_mask(df, features=FEATURES, multiplier=IQR_MULTIPLIER):
    """Lọc outlier IQR tính trên TOÀN BỘ bảng — đúng như bản gốc.

    Sai ở chỗ: ngưỡng Q1/Q3 được tính từ cả dữ liệu test. Tệ hơn nữa, nó xóa
    luôn những hàng "cực đoan" trong test — mà cửa sổ AFib nặng thì vốn dĩ
    cực đoan. Đề thi lại dễ đi một lần nữa.
    """
    mask = pd.Series(True, index=df.index)
    for col in features:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df[col].between(q1 - multiplier * iqr, q3 + multiplier * iqr)
    return mask


def train_only_iqr_mask(df_train, features=FEATURES, multiplier=IQR_MULTIPLIER):
    """Bản trung thực của luật trên: ngưỡng chỉ tính từ train của fold."""
    return global_iqr_mask(df_train, features, multiplier).to_numpy()


def globally_scaled(df, features=FEATURES):
    """Fit StandardScaler trên TOÀN BỘ dữ liệu rồi transform — đúng bản gốc.

    Mỗi cột được trừ trung bình và chia độ lệch chuẩn TÍNH TRÊN CẢ TẬP TEST.
    Thông tin về phân bố của test vì thế rò rỉ vào quá trình huấn luyện.
    """
    out = df.copy()
    out[features] = StandardScaler().fit_transform(df[features].to_numpy())
    return out


def run_one_scale(scale, verbose=True):
    """Chạy đầy đủ một quy mô: tái dựng bản gốc + chấm lại bằng LOSO."""
    step_s = scale['step_s']
    df = extract_table(CHANNEL, WINDOW_S, step_s, verbose=verbose)

    # --- Đường đi của bản gốc: IQR toàn cục -> scale toàn cục -> chia ngẫu nhiên
    keep = global_iqr_mask(df)
    df_leaky = globally_scaled(df[keep].reset_index(drop=True))
    n_dropped = int((~keep).sum())

    # scale=False vì bảng đã được chuẩn hóa sẵn (đúng kiểu sai của bản gốc)
    leaky = honest.leaky_random_split(df_leaky, FEATURES, make_model, scale=False)

    # --- Đường đi trung thực: LOSO, IQR chỉ trên train, scaler theo fold
    win, subj, _ = honest.loso(df, FEATURES, make_model,
                               train_filter=train_only_iqr_mask)

    row = {
        'step_s': step_s,
        'overlap_pct': scale['overlap_pct'],
        'label': scale['label'],
        'n_windows': int(len(df)),
        'n_dropped_iqr': n_dropped,
        'leaky_accuracy': leaky['accuracy'],
        'leaky_auc': leaky['roc_auc'],
        'loso_accuracy': win['accuracy'],
        'loso_auc': win['roc_auc'],
        'loso_subject_accuracy': subj['accuracy'],
        'inflation': round(leaky['accuracy'] - win['accuracy'], 4),
    }
    if verbose:
        print(f"  -> {scale['label']:<18} {row['n_windows']:>6} cửa sổ | "
              f"gốc {row['leaky_accuracy'] * 100:5.2f}% | "
              f"LOSO {row['loso_accuracy'] * 100:5.2f}% | "
              f"ảo {row['inflation'] * 100:+5.2f} điểm")
    return row, df, df_leaky, subj


def run(verbose=True):
    if verbose:
        print('=' * 70)
        print('v3 — Benchmark đa quy mô trên kênh ECG (4 mức chồng lấn)')
        print('=' * 70)

    rows, detail_subj, biggest_df, biggest_leaky = [], None, None, None
    for scale in SCALES:
        if verbose:
            print(f"\n[quy mô] bước {scale['step_s']}s — {scale['label']}")
        row, df, df_leaky, subj = run_one_scale(scale, verbose=verbose)
        rows.append(row)
        if scale['step_s'] == 10.0:      # quy mô "mặc định" của v3 (4.083 cửa sổ)
            detail_subj, biggest_df, biggest_leaky = subj, df, df_leaky

    scale_table = pd.DataFrame(rows)

    # Bảng xếp hạng mô hình theo đúng cách cũ (chỉ để tái hiện con số lịch sử)
    if verbose:
        print('\nBảng xếp hạng mô hình theo cách chấm CŨ (quy mô mặc định):')
    leaderboard = []
    for name, factory in [('Logistic Regression',
                           lambda: LogisticRegression(max_iter=2000, random_state=42)),
                          ('Random Forest', make_model),
                          ('Stacking Ensemble', make_stacking)]:
        res = honest.leaky_random_split(biggest_leaky, FEATURES, factory, scale=False)
        leaderboard.append({'Mô hình': name,
                            'Accuracy': res['accuracy'],
                            'Recall': res['recall'],
                            'ROC-AUC': res['roc_auc']})
        if verbose:
            print(f"  {name:<22} acc {res['accuracy']:.4f}  AUC {res['roc_auc']:.4f}")

    default = next(r for r in rows if r['step_s'] == 10.0)
    payload = {
        'version': VERSION_ID,
        'title': 'Benchmark đa quy mô trên kênh ECG',
        'config': {
            'channel': CHANNEL,
            'window_s': WINDOW_S,
            'scales': [{'step_s': s['step_s'], 'overlap_pct': s['overlap_pct']}
                       for s in SCALES],
            'n_features': len(FEATURES),
            'features': FEATURES,
            'cleaning': f'IQR x{IQR_MULTIPLIER} tính trên toàn bộ bảng',
            'scaling': 'StandardScaler fit toàn bộ dữ liệu trước khi chia',
            'split': 'ngẫu nhiên theo cửa sổ 80/20',
            'model': 'Random Forest (so sánh) + LR/RF/Stacking (bảng xếp hạng)',
        },
        'data': {
            'n_windows': default['n_windows'],
            'n_subjects': int(biggest_df['record_id'].nunique()),
        },
        'original_claim': ORIGINAL_CLAIM,
        'scale_table': scale_table.to_dict(orient='records'),
        'leaderboard': leaderboard,
        'reproduced_leaky': {'accuracy': default['leaky_accuracy'],
                             'roc_auc': default['leaky_auc']},
        'regraded_loso_window': {'accuracy': default['loso_accuracy'],
                                 'roc_auc': default['loso_auc']},
        'regraded_loso_subject': {
            k: v for k, v in detail_subj.items()
            if k not in ('subjects', 'subject_proba', 'subject_true')
        },
        'subject_detail': {
            'subjects': detail_subj['subjects'],
            'proba': detail_subj['subject_proba'],
            'true': detail_subj['subject_true'],
        },
        'inflation': default['inflation'],
        'leakage': [
            'Cửa sổ chồng lấn tới 92%: bản sao gần đúng của mẫu test nằm trong train.',
            'StandardScaler fit trên toàn bộ dữ liệu trước khi chia.',
            'Lọc IQR tính ngưỡng trên toàn bộ bảng, xóa cả ca khó trong test.',
            'Chia ngẫu nhiên theo cửa sổ, không theo bệnh nhân.',
            'Trích đặc trưng từ kênh ECG — không phải PPG như sản phẩm thật.',
        ],
    }
    store.save(VERSION_ID, payload)
    # Xuất model ở quy mô mặc định (bước 10s). Kênh ECG — ghi rõ vào thẻ model
    # vì đem áp lên PPG của vòng đeo là sai loại tín hiệu.
    export.export(VERSION_ID, biggest_df, FEATURES, make_model, payload,
                  train_filter=train_only_iqr_mask,
                  channel=CHANNEL, verbose=verbose)

    if verbose:
        print('\n' + scale_table[['label', 'n_windows', 'leaky_accuracy',
                                  'loso_accuracy', 'inflation']].to_string(index=False))
        print(f"\nĐã lưu: {store.results_path(VERSION_ID)}")
    return payload


if __name__ == '__main__':
    run()
