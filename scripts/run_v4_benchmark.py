"""Bước 2 — Benchmark v4: LOSO theo bệnh nhân, không data leakage.

Đọc data/features/mimic_features_v4.csv (chạy run_v4_extraction.py trước)
-> Leave-One-Subject-Out: 35 folds, mỗi fold giữ trọn 1 bệnh nhân làm test
-> Trong mỗi fold: lọc outlier IQR (train-only) + GridSearchCV GroupKFold(3)
-> Metrics 2 cấp: cửa sổ 30s & bệnh nhân; biểu đồ; lưu models/mimic/benchmark_v4/

Chạy:  python scripts/run_v4_benchmark.py [--full16]
       --full16: dùng đủ 16 đặc trưng (mặc định loại nhóm LF không đáng tin ở 30s)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd

from healthsense_ml import config
from healthsense_ml.feature_extraction import load_features
from healthsense_ml.training import run_loso_benchmark
from healthsense_ml import evaluation


def main():
    use_full = '--full16' in sys.argv
    feature_cols = config.ALL_FEATURES if use_full else config.CORE_FEATURES

    print('=' * 70)
    print('🔬 HEALTHSENSE ML v4 — LOSO BENCHMARK (chống data leakage)')
    print(f'   Split : Leave-One-Subject-Out theo record_id')
    print(f'   Tiền xử lý: IQR x{config.IQR_MULTIPLIER} + StandardScaler (fit train-only)')
    print(f'   Đặc trưng : {len(feature_cols)} '
          f'({"đủ 16" if use_full else "loại nhóm LF (30s không đủ tin cậy)"})')
    print('=' * 70)

    df = load_features()
    n_subj = df['record_id'].nunique()
    n_af = df[df['status'] == 1]['record_id'].nunique()
    print(f'\n📊 {len(df)} cửa sổ | {n_subj} bệnh nhân ({n_af} AFib, {n_subj - n_af} Normal)')

    predictions = run_loso_benchmark(df, feature_cols=feature_cols)

    os.makedirs(config.BENCHMARK_V4_DIR, exist_ok=True)
    pred_path = os.path.join(config.BENCHMARK_V4_DIR, 'loso_predictions.csv')
    predictions.to_csv(pred_path, index=False)

    # Bảng metrics
    summary = evaluation.summarize(predictions)
    summary_path = os.path.join(config.BENCHMARK_V4_DIR, 'benchmark_results_v4.csv')
    summary.to_csv(summary_path, index=False)

    print('\n' + '=' * 70)
    print('📋 KẾT QUẢ (không leakage — con số phản ánh bệnh nhân CHƯA TỪNG THẤY)')
    print('=' * 70)
    for level, label in [('window', 'MỨC CỬA SỔ 30s'), ('subject', 'MỨC BỆNH NHÂN')]:
        sub = summary[summary['Level'] == level]
        print(f'\n📊 {label}:')
        cols = ['Model', 'Accuracy', 'Recall (Sensitivity)', 'Specificity',
                'F1-Score', 'ROC-AUC', 'FN', 'FP']
        print(sub[cols].to_string(index=False,
              float_format=lambda v: f'{v:.4f}'))

    # Biểu đồ
    print('\n🎨 Tạo biểu đồ...')
    evaluation.plot_confusion_matrices(predictions, config.BENCHMARK_V4_DIR, 'subject')
    evaluation.plot_confusion_matrices(predictions, config.BENCHMARK_V4_DIR, 'window')
    evaluation.plot_roc_curves(predictions, config.BENCHMARK_V4_DIR, 'window')
    evaluation.plot_roc_curves(predictions, config.BENCHMARK_V4_DIR, 'subject')
    evaluation.plot_subject_probabilities(predictions, config.BENCHMARK_V4_DIR)

    print(f'\n✅ HOÀN TẤT! Kết quả tại: {config.BENCHMARK_V4_DIR}')


if __name__ == '__main__':
    main()
