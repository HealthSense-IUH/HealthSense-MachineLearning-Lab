"""Đo hiệu năng — dùng chung cho cả 4 phiên bản để so sánh công bằng.

Hai mức đo hoàn toàn khác nhau:
- Mức CỬA SỔ: mỗi cửa sổ 30 giây là một mẫu. Đây là con số các phiên bản
  cũ (v1-v3) hay báo cáo.
- Mức BỆNH NHÂN: gộp toàn bộ cửa sổ của một người thành một chẩn đoán duy
  nhất. Đây mới là thứ có ý nghĩa lâm sàng — bác sĩ chẩn đoán con người,
  không chẩn đoán từng đoạn 30 giây.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def window_metrics(y_true, y_pred, y_proba=None):
    """Metrics mức cửa sổ. Trả về dict đã làm tròn 4 chữ số."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        'n_samples': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Specificity = tỉ lệ người BÌNH THƯỜNG được nhận đúng. Quan trọng ngang
    # recall: specificity thấp nghĩa là báo động giả tràn lan.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    out['tn'], out['fp'], out['fn'], out['tp'] = int(tn), int(fp), int(fn), int(tp)

    if y_proba is not None and len(np.unique(y_true)) == 2:
        out['roc_auc'] = float(roc_auc_score(y_true, y_proba))
    else:
        out['roc_auc'] = None

    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def subject_metrics(record_ids, y_true, y_proba, threshold=0.5):
    """Gộp dự đoán từng cửa sổ thành chẩn đoán mức bệnh nhân.

    Quy tắc gộp: lấy TRUNG BÌNH xác suất các cửa sổ của một người, rồi so
    với ngưỡng. Cách này ổn định hơn "bỏ phiếu đa số" khi số cửa sổ mỗi
    người chênh lệch.
    """
    record_ids = np.asarray(record_ids)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    subjects, true_labels, mean_probas = [], [], []
    for rid in np.unique(record_ids):
        mask = record_ids == rid
        subjects.append(rid)
        # Nhãn của bệnh nhân là nhãn chung của mọi cửa sổ thuộc người đó
        true_labels.append(int(round(float(y_true[mask].mean()))))
        mean_probas.append(float(y_proba[mask].mean()))

    true_labels = np.array(true_labels)
    mean_probas = np.array(mean_probas)
    preds = (mean_probas >= threshold).astype(int)

    out = window_metrics(true_labels, preds, mean_probas)
    out['n_subjects'] = out.pop('n_samples')
    out['subjects'] = list(subjects)
    out['subject_proba'] = [round(p, 4) for p in mean_probas]
    out['subject_true'] = [int(v) for v in true_labels]
    return out


def summarize(name, window, subject=None):
    """Một dòng tóm tắt gọn để in ra trong notebook."""
    line = (f'{name:28} | cửa sổ: acc {window["accuracy"]:.4f} '
            f'recall {window["recall"]:.4f} '
            f'AUC {window["roc_auc"] if window["roc_auc"] is not None else float("nan"):.4f}')
    if subject:
        line += (f' || bệnh nhân: acc {subject["accuracy"]:.4f} '
                 f'({subject["n_subjects"]} người)')
    return line
