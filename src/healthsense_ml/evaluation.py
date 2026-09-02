"""Đánh giá benchmark v4: metrics 2 cấp (cửa sổ & bệnh nhân) + biểu đồ.

Mức cửa sổ (window-level): gộp mọi dự đoán LOSO — phản ánh độ chính xác
từng lần đo 30 giây.

Mức bệnh nhân (subject-level): trung bình xác suất các cửa sổ của mỗi
bệnh nhân -> 1 điểm số/người. Đây là con số phản ánh đúng bài toán sàng lọc
("người này có bị AFib không?") và là metric nên dùng để báo cáo.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

from . import config

sns.set_theme(style='whitegrid')

MODEL_COLORS = {
    'Logistic Regression': '#3b82f6',
    'Random Forest': '#10b981',
    'XGBoost': '#f59e0b',
}


def _metrics_row(y_true, y_pred, y_prob, model, level):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'Model': model,
        'Level': level,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall (Sensitivity)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'FN': int(fn), 'FP': int(fp), 'TP': int(tp), 'TN': int(tn),
    }


def summarize(predictions):
    """Từ bảng dự đoán LOSO -> bảng metrics window-level & subject-level."""
    rows = []
    for model in predictions['model'].unique():
        p = predictions[predictions['model'] == model]

        # Window-level: gộp toàn bộ cửa sổ
        rows.append(_metrics_row(
            p['status'], p['pred'], p['prob'], model, 'window'))

        # Subject-level: trung bình prob theo bệnh nhân
        by_subj = p.groupby('record_id').agg(
            status=('status', 'first'), prob=('prob', 'mean'))
        by_subj['pred'] = (by_subj['prob'] >= 0.5).astype(int)
        rows.append(_metrics_row(
            by_subj['status'], by_subj['pred'], by_subj['prob'],
            model, 'subject'))
    return pd.DataFrame(rows)


def plot_confusion_matrices(predictions, save_dir, level='subject'):
    """3 confusion matrix cạnh nhau (mặc định mức bệnh nhân)."""
    models = list(predictions['model'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5.5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        p = predictions[predictions['model'] == model]
        if level == 'subject':
            by = p.groupby('record_id').agg(
                status=('status', 'first'), prob=('prob', 'mean'))
            y_true, y_pred = by['status'], (by['prob'] >= 0.5).astype(int)
        else:
            y_true, y_pred = p['status'], p['pred']

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'AFib'],
                    yticklabels=['Normal', 'AFib'],
                    annot_kws={'size': 16, 'weight': 'bold'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(model, fontweight='bold', color=MODEL_COLORS.get(model))

    unit = 'bệnh nhân' if level == 'subject' else 'cửa sổ 30s'
    fig.suptitle(f'Confusion Matrices — LOSO, mức {unit}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrices_{level}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_roc_curves(predictions, save_dir, level='window'):
    """ROC overlay 3 mô hình."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for model in predictions['model'].unique():
        p = predictions[predictions['model'] == model]
        if level == 'subject':
            by = p.groupby('record_id').agg(
                status=('status', 'first'), prob=('prob', 'mean'))
            y_true, y_prob = by['status'], by['prob']
        else:
            y_true, y_prob = p['status'], p['prob']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2.5, color=MODEL_COLORS.get(model),
                label=f'{model} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    unit = 'bệnh nhân' if level == 'subject' else 'cửa sổ 30s'
    ax.set_title(f'ROC Curve — LOSO, mức {unit}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(save_dir, f'roc_curves_{level}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_subject_probabilities(predictions, save_dir, model='Random Forest'):
    """Xác suất AFib trung bình từng bệnh nhân — nhìn rõ ai bị phân loại sai."""
    p = predictions[predictions['model'] == model]
    by = p.groupby('record_id').agg(
        status=('status', 'first'), prob=('prob', 'mean')).sort_values('prob')

    colors = ['#ef4444' if s == 1 else '#3b82f6' for s in by['status']]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(by)), by['prob'], color=colors, alpha=0.85)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(by)))
    ax.set_yticklabels(by.index, fontsize=7)
    ax.set_xlabel('P(AFib) trung bình các cửa sổ')
    ax.set_title(f'Xác suất AFib theo bệnh nhân — {model} (LOSO)\n'
                 f'Đỏ = AFib thật, Xanh = Normal thật',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'subject_probabilities.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
