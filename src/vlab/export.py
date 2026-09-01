"""Xuất mô hình cuối cùng của mỗi phiên bản ra file .pkl.

Ý tưởng: mỗi đời pipeline, nếu ngày đó đem đi triển khai thật, sẽ cho ra một
file model như thế nào? Đây là câu trả lời — 4 file, một cho mỗi đời.

HAI QUY TẮC AN TOÀN, học từ chính sai lầm có thật trong dự án
--------------------------------------------------------------
1. LUÔN gói trong sklearn Pipeline (scaler nằm bên trong).
   Trong `HealthSense-AI-Service/app/models/` có file `best_model_8165.pkl` —
   một MLPClassifier TRẦN, không kèm scaler, sinh ra từ pipeline v3 vốn chuẩn
   hóa dữ liệu toàn cục từ trước. Ai nạp nó rồi đưa đặc trưng thô vào sẽ nhận
   kết quả rác mà không có lỗi nào báo ra. Gói scaler vào trong Pipeline khiến
   chuyện đó không thể xảy ra.

2. LUÔN kèm file .json ghi ĐIỂM THẬT (LOSO), không phải điểm bản gốc công bố.
   Điểm bản gốc của v1-v3 bị thổi phồng 6-7 điểm do data leakage. File model
   đi tới đâu thì con số trung thực đi theo tới đó.

Lưu ý: model của v1, v2 KHÔNG phải "model hỏng" — chấm bằng LOSO chúng ngang
ngửa v4. Cái hỏng ngày đó là *thước đo*, không phải mô hình. Riêng v3 thì cần
cẩn thận thật: nó học trên kênh ECG, đem áp lên PPG của vòng đeo là sai loại
tín hiệu.
"""

import json
import os

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .raw import MODELS_DIR


def model_path(version):
    """models/<version>.pkl"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f'{version}.pkl')


def card_path(version):
    """models/<version>.json — thẻ model đi kèm."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f'{version}.json')


def export(version, df, feature_cols, model_factory, payload,
           train_filter=None, channel='PPG', verbose=True):
    """Huấn luyện model cuối của một phiên bản trên TOÀN BỘ dữ liệu và lưu.

    df           : bảng đặc trưng đầy đủ
    feature_cols : bộ đặc trưng của phiên bản đó (thứ tự cột có ý nghĩa!)
    model_factory: hàm không tham số trả về mô hình chưa huấn luyện
    payload      : dict kết quả của phiên bản (để lấy điểm LOSO trung thực)
    train_filter : luật làm sạch, CHỈ áp lên dữ liệu huấn luyện

    Khác với lúc benchmark: ở đây không cần giữ tập test, vì mục tiêu là ra
    một model dùng được. Điểm số trung thực đã được đo từ trước bằng LOSO và
    được ghi vào thẻ model.
    """
    train = df
    n_dropped = 0
    if train_filter is not None:
        keep = np.asarray(train_filter(df), dtype=bool)
        n_dropped = int((~keep).sum())
        train = df[keep]

    X = train[feature_cols].to_numpy(dtype=np.float64)
    y = train['status'].to_numpy(dtype=int)

    # Scaler LUÔN nằm trong Pipeline — xem quy tắc 1 ở đầu file
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model_factory())])
    pipe.fit(X, y)

    path = model_path(version)
    joblib.dump(pipe, path, compress=3)

    loso_w = payload.get('regraded_loso_window', {})
    loso_s = payload.get('regraded_loso_subject', {})
    old_way = (payload.get('reproduced_leaky')
               or payload.get('if_graded_the_old_way') or {})

    card = {
        'version': version,
        'title': payload.get('title'),
        'channel': channel,
        'input': f'Cửa sổ 30s -> {len(feature_cols)} đặc trưng HRV, '
                 f'ĐÚNG THỨ TỰ trong "features"',
        'features': list(feature_cols),
        'output': 'predict_proba[:, 1] = P(AFib); ngưỡng mặc định 0.5',
        'trained_on': {
            'dataset': 'MIMIC PERform AF',
            'n_subjects': int(df['record_id'].nunique()),
            'n_windows_available': int(len(df)),
            'n_windows_used': int(len(train)),
            'n_windows_dropped_by_cleaning': n_dropped,
        },
        'honest_score_loso': {
            'window_accuracy': loso_w.get('accuracy'),
            'window_roc_auc': loso_w.get('roc_auc'),
            'subject_accuracy': loso_s.get('accuracy'),
            'subject_recall': loso_s.get('recall'),
        },
        'score_if_graded_the_old_way': {
            'accuracy': old_way.get('accuracy'),
            'note': 'Chia ngẫu nhiên theo cửa sổ — con số bị thổi phồng.',
        },
        'inflation_points': payload.get('inflation'),
        'known_leakage': payload.get('leakage') or [],
        'warning': (
            'HIỆN VẬT HỌC TẬP — không dùng cho sản phẩm. '
            'Model triển khai duy nhất là models/healthsense_afib_pipeline.pkl.'
        ) if version != 'v4' else (
            'Model đối chứng của bảo tàng (Random Forest, 35 bệnh nhân MIMIC). '
            'KHÁC với model triển khai models/healthsense_afib_pipeline.pkl '
            '(XGBoost, 60 bệnh nhân, có cân bằng nguồn).'
        ),
    }
    with open(card_path(version), 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    if verbose:
        size = os.path.getsize(path)
        print(f'  Đã xuất model: models/{version}.pkl ({size / 1024:.0f} KB) '
              f'+ {version}.json')
    return path
