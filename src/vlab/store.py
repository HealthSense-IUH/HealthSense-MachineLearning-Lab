"""Lưu / nạp kết quả từng phiên bản.

Notebook báo cáo không nên huấn luyện lại mỗi lần mở. Quy ước:
- Script `src/vN/pipeline.py` chạy thật, ghi kết quả vào models/vN/results.json
- Notebook `src/report/vN_*.ipynb` nạp file đó và trình bày.
Nếu chưa có kết quả, notebook sẽ tự chạy pipeline (mất vài phút).
"""

import json
import os

from .raw import MODELS_DIR


def results_path(version):
    """models/<version>/results.json"""
    d = os.path.join(MODELS_DIR, version)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'results.json')


def save(version, payload):
    """Ghi kết quả phiên bản (dict) ra JSON, giữ tiếng Việt nguyên vẹn."""
    path = results_path(version)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_fallback)
    return path


def load(version):
    """Nạp kết quả phiên bản; trả về None nếu chưa chạy bao giờ."""
    path = results_path(version)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_or_run(version, run_fn):
    """Nạp kết quả có sẵn, chưa có thì chạy `run_fn()` để tạo."""
    cached = load(version)
    if cached is not None:
        return cached
    print(f'Chưa có kết quả {version} — đang chạy pipeline (có thể mất vài phút)...')
    run_fn()
    return load(version)


def _fallback(obj):
    """Chuyển kiểu numpy/pandas về kiểu Python thuần khi ghi JSON."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)
