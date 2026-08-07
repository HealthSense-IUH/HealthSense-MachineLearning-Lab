"""
MIMIC-III v3 Outlier Filter
============================
Lọc Outlier bằng phương pháp IQR (Interquartile Range) cho tất cả 4 quy mô dữ liệu.
Sau đó thực hiện chuẩn hóa Z-Score & Min-Max Scaling và lưu file sạch.

Dữ liệu đầu vào: data/features/mimic_features_{scale}.csv
Dữ liệu đầu ra:  data/processed/mimic_zscore_{scale}_clean.csv
                  data/processed/mimic_minmax_{scale}_clean.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Fix Unicode encoding on Windows (cp1252 -> utf-8)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================================
# Cấu hình
# ============================================================
SCALES = ['1360', '4083', '8165', '16358']
IQR_MULTIPLIER = 1.5  # Hệ số IQR chuẩn (1.5 = mild outlier, 3.0 = extreme)

# Tìm thư mục gốc project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def filter_outliers_iqr(df, feature_cols, multiplier=1.5):
    """
    Lọc outlier bằng phương pháp IQR.
    Loại bỏ các hàng có BẤT KỲ feature nào nằm ngoài [Q1 - k*IQR, Q3 + k*IQR].

    Parameters:
        df: DataFrame chứa dữ liệu
        feature_cols: danh sách cột feature (không bao gồm 'status')
        multiplier: hệ số IQR (mặc định 1.5)

    Returns:
        DataFrame đã lọc outlier
    """
    mask = pd.Series([True] * len(df), index=df.index)

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        mask = mask & (df[col] >= lower) & (df[col] <= upper)

    return df[mask].reset_index(drop=True)


def process_scale(tag):
    """Xử lý 1 quy mô dữ liệu: Lọc Outlier -> Chuẩn hóa -> Lưu file."""

    feature_file = os.path.join(FEATURES_DIR, f'mimic_features_{tag}.csv')
    if not os.path.exists(feature_file):
        print(f'  ❌ Không tìm thấy: {feature_file}')
        return None

    # 1. Đọc dữ liệu features thô
    df = pd.read_csv(feature_file)
    original_count = len(df)
    feature_cols = [c for c in df.columns if c != 'status']

    print(f'\n{"="*60}')
    print(f'📊 QUY MÔ {tag} MẪU')
    print(f'{"="*60}')
    print(f'  Tổng mẫu ban đầu: {original_count}')

    # 2. Phân bố nhãn trước lọc
    label_counts_before = df['status'].value_counts()
    print(f'  Phân bố nhãn trước lọc:')
    for label, count in label_counts_before.items():
        print(f'    - {label}: {count} ({count/original_count*100:.1f}%)')

    # 3. Lọc outlier bằng IQR
    df_clean = filter_outliers_iqr(df, feature_cols, multiplier=IQR_MULTIPLIER)
    clean_count = len(df_clean)
    removed_count = original_count - clean_count

    print(f'  Mẫu sau khi lọc outlier (IQR x{IQR_MULTIPLIER}): {clean_count}')
    print(f'  Số mẫu bị loại: {removed_count} ({removed_count/original_count*100:.1f}%)')

    # 4. Phân bố nhãn sau lọc
    label_counts_after = df_clean['status'].value_counts()
    print(f'  Phân bố nhãn sau lọc:')
    for label, count in label_counts_after.items():
        print(f'    - {label}: {count} ({count/clean_count*100:.1f}%)')

    # 5. Chuẩn hóa Z-Score
    X = df_clean.drop(columns=['status'])
    y = df_clean['status']

    scaler_z = StandardScaler()
    X_z = pd.DataFrame(scaler_z.fit_transform(X), columns=feature_cols)
    df_zscore = pd.concat([X_z, y.reset_index(drop=True)], axis=1)

    # 6. Chuẩn hóa Min-Max
    scaler_mm = MinMaxScaler()
    X_mm = pd.DataFrame(scaler_mm.fit_transform(X), columns=feature_cols)
    df_minmax = pd.concat([X_mm, y.reset_index(drop=True)], axis=1)

    # 7. Lưu file
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    zscore_path = os.path.join(PROCESSED_DIR, f'mimic_zscore_{tag}_clean.csv')
    minmax_path = os.path.join(PROCESSED_DIR, f'mimic_minmax_{tag}_clean.csv')

    df_zscore.to_csv(zscore_path, index=False)
    df_minmax.to_csv(minmax_path, index=False)

    print(f'  ✅ Đã lưu Z-Score: {zscore_path}')
    print(f'  ✅ Đã lưu Min-Max: {minmax_path}')

    return {
        'scale': tag,
        'original': original_count,
        'clean': clean_count,
        'removed': removed_count,
        'removed_pct': removed_count / original_count * 100
    }


def main():
    print('=' * 60)
    print('🔬 MIMIC-III v3 OUTLIER FILTER (IQR Method)')
    print(f'   Hệ số IQR: {IQR_MULTIPLIER}')
    print('=' * 60)

    results = []
    for tag in SCALES:
        result = process_scale(tag)
        if result:
            results.append(result)

    # Tổng kết
    if results:
        print(f'\n{"="*60}')
        print('📋 TỔNG KẾT LỌC OUTLIER')
        print(f'{"="*60}')
        print(f'{"Quy mô":<12} {"Ban đầu":>10} {"Sau lọc":>10} {"Bị loại":>10} {"Tỉ lệ":>8}')
        print('-' * 52)
        for r in results:
            print(f'{r["scale"]:<12} {r["original"]:>10} {r["clean"]:>10} {r["removed"]:>10} {r["removed_pct"]:>7.1f}%')

    print('\n✅ Hoàn tất lọc Outlier!')


if __name__ == '__main__':
    main()
