"""Hai cách "chấm điểm" cùng một bảng đặc trưng — để đo đúng cái giá của leakage.

Đây là công cụ trung tâm của bảo tàng phiên bản. Với BẤT KỲ bảng đặc trưng
nào (của v1, v2, v3 hay v4), ta chấm hai lần:

1. `leaky_random_split()` — cách các phiên bản cũ đã làm: trộn toàn bộ cửa
   sổ rồi bốc ngẫu nhiên 20% làm test. Vì mỗi bệnh nhân có hàng chục cửa sổ,
   gần như chắc chắn người xuất hiện ở test cũng có mặt trong train. Mô hình
   chỉ cần NHỚ MẶT bệnh nhân là đủ điểm cao.

2. `loso()` — Leave-One-Subject-Out: mỗi vòng giữ trọn một bệnh nhân ra
   ngoài, huấn luyện trên 34 người còn lại. Mọi bước tiền xử lý (chuẩn hóa,
   lọc outlier) chỉ được nhìn dữ liệu train của vòng đó. Đây là mô phỏng
   đúng tình huống thật: máy gặp một người CHƯA TỪNG THẤY.

Chênh lệch giữa (1) và (2) chính là phần điểm "ảo" do leakage tạo ra.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import subject_metrics, window_metrics

RANDOM_STATE = 42


def _as_pipeline(model_factory, scale=True):
    """Bọc mô hình trong Pipeline để scaler LUÔN fit trên train của fold."""
    model = model_factory()
    if not scale:
        return model
    return Pipeline([('scaler', StandardScaler()), ('model', model)])


def leaky_random_split(df, feature_cols, model_factory, label_col='status',
                       group_col='record_id', test_size=0.2, scale=True):
    """Chia ngẫu nhiên theo CỬA SỔ — tái dựng cách làm của v1/v2/v3.

    Cột định danh bệnh nhân bị bỏ đi trước khi chia (đúng như bản gốc), nên
    một người có thể vừa ở train vừa ở test.

    Trả về dict: metrics mức cửa sổ + thống kê chồng lấn bệnh nhân.
    """
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy(dtype=int)
    groups = df[group_col].to_numpy() if group_col in df.columns else None

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    pipe = _as_pipeline(model_factory, scale)
    pipe.fit(X[idx_tr], y[idx_tr])

    y_pred = pipe.predict(X[idx_te])
    y_proba = (pipe.predict_proba(X[idx_te])[:, 1]
               if hasattr(pipe, 'predict_proba') else None)

    result = window_metrics(y[idx_te], y_pred, y_proba)
    result['split'] = 'random_window'

    # Đo mức độ leakage: bao nhiêu bệnh nhân ở test cũng có mặt trong train?
    if groups is not None:
        tr_subjects = set(groups[idx_tr])
        te_subjects = set(groups[idx_te])
        overlap = te_subjects & tr_subjects
        result['n_test_subjects'] = len(te_subjects)
        result['n_subjects_also_in_train'] = len(overlap)
        result['subject_overlap_pct'] = round(
            100.0 * len(overlap) / len(te_subjects), 1) if te_subjects else 0.0
    return result


def loso(df, feature_cols, model_factory, label_col='status',
         group_col='record_id', scale=True, verbose=False, train_filter=None):
    """Leave-One-Subject-Out — cách chấm trung thực.

    train_filter: hàm nhận DataFrame train của fold, trả về mask giữ hàng.
        Dùng cho các bước làm sạch (luật theo nhãn của v2, lọc IQR của v3).
        Bản gốc áp luật lên TOÀN BỘ dữ liệu — tức là vứt luôn cả những ca khó
        trong test. Ở đây luật chỉ được nhìn train của từng fold; test giữ
        nguyên 100%, đúng như đời thực: không ai được phép vứt bỏ dữ liệu của
        bệnh nhân mới chỉ vì nó khó.

    Trả về (window_result, subject_result, predictions_df).
    predictions_df có cột: record_id, status, proba, pred — tiện vẽ biểu đồ.
    """
    if group_col not in df.columns:
        raise ValueError(
            f"Bảng đặc trưng thiếu cột '{group_col}'. Không có danh tính bệnh "
            f"nhân thì KHÔNG THỂ chia theo người — đây chính là lý do các "
            f"bảng đặc trưng v3 cũ không thể kiểm định lại được.")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy(dtype=int)
    groups = df[group_col].to_numpy()

    logo = LeaveOneGroupOut()
    rows = []
    n_folds = logo.get_n_splits(groups=groups)

    for i, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        # Làm sạch CHỈ trên train của fold (nếu phiên bản có bước này)
        if train_filter is not None:
            keep = np.asarray(train_filter(df.iloc[tr]), dtype=bool)
            if keep.sum() >= 10 and len(np.unique(y[tr][keep])) == 2:
                tr = tr[keep]

        # Chỉ huấn luyện được khi train có đủ 2 lớp
        if len(np.unique(y[tr])) < 2:
            continue

        pipe = _as_pipeline(model_factory, scale)
        pipe.fit(X[tr], y[tr])

        proba = (pipe.predict_proba(X[te])[:, 1]
                 if hasattr(pipe, 'predict_proba')
                 else pipe.predict(X[te]).astype(float))
        pred = (proba >= 0.5).astype(int)

        for j, k in enumerate(te):
            rows.append({
                'record_id': groups[k],
                'status': int(y[k]),
                'proba': float(proba[j]),
                'pred': int(pred[j]),
            })

        if verbose:
            held = groups[te][0]
            print(f'  [{i:>2}/{n_folds}] giữ ra {held}: '
                  f'{len(te)} cửa sổ, đúng {int((pred == y[te]).sum())}')

    preds = pd.DataFrame(rows)
    win = window_metrics(preds['status'], preds['pred'], preds['proba'])
    win['split'] = 'LOSO'
    win['n_folds'] = int(preds['record_id'].nunique())
    subj = subject_metrics(preds['record_id'], preds['status'], preds['proba'])
    return win, subj, preds


def compare(df, feature_cols, model_factory, label_col='status',
            group_col='record_id', scale=True, df_leaky=None, train_filter=None):
    """Chấm theo cả hai cách, trả về bảng so sánh gọn.

    df:       bảng ĐẦY ĐỦ — dùng cho LOSO (test không bị vứt hàng nào).
    df_leaky: bảng đã qua xử lý kiểu bản gốc (vd: đã lọc toàn cục). Nếu bỏ
              trống thì dùng chính `df`.

    Đây là hàm được dùng ở cuối mỗi notebook báo cáo v1/v2/v3.
    """
    leaky = leaky_random_split(df if df_leaky is None else df_leaky,
                               feature_cols, model_factory,
                               label_col, group_col, scale=scale)
    win, subj, preds = loso(df, feature_cols, model_factory,
                            label_col, group_col, scale=scale,
                            train_filter=train_filter)

    table = pd.DataFrame([
        {
            'Cách chấm': 'Ngẫu nhiên theo cửa sổ (bản gốc)',
            'Accuracy': leaky['accuracy'],
            'Recall': leaky['recall'],
            'Specificity': leaky['specificity'],
            'ROC-AUC': leaky['roc_auc'],
        },
        {
            'Cách chấm': 'LOSO theo bệnh nhân (trung thực)',
            'Accuracy': win['accuracy'],
            'Recall': win['recall'],
            'Specificity': win['specificity'],
            'ROC-AUC': win['roc_auc'],
        },
    ])
    gap = round(leaky['accuracy'] - win['accuracy'], 4)
    return {
        'table': table,
        'leaky': leaky,
        'loso_window': win,
        'loso_subject': subj,
        'predictions': preds,
        'inflation': gap,
    }
