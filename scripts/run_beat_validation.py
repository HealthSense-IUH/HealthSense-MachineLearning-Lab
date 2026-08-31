"""Bước 5 — Kiểm chứng dò nhịp PPG bằng ECG đồng bộ (MIMIC PERform).

Chấm điểm bộ dò nhịp PPG của pipeline v4 với đáp án chuẩn là R-peak trên
cột ECG ghi song song. Đồng thời tính RMSSD/pNN50 từ CHÍNH ECG của từng
bệnh nhân — thẩm định luôn 2 ca "Normal" nghi nhãn sai.

Chạy:  python scripts/run_beat_validation.py
Kết quả: models/beat_validation/beat_validation.csv + tóm tắt in màn hình
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd

from healthsense_ml import config
from healthsense_ml.beat_validation import validate_all

OUTPUT_DIR = os.path.join(config.MODELS_DIR, 'beat_validation')


def main():
    print('=' * 70)
    print('🔬 KIỂM CHỨNG DÒ NHỊP PPG BẰNG ECG ĐỒNG BỘ (MIMIC PERform)')
    print(f'   Dung sai khớp nhịp: ±150 ms (sau khi bù PTT)')
    print('=' * 70)

    df = validate_all(verbose=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, 'beat_validation.csv')
    df.to_csv(out_csv, index=False)

    ok = df[~df['ecg_unreliable']].copy()
    print('\n' + '=' * 70)
    print(f'📋 TỔNG KẾT ({len(ok)}/{len(df)} bệnh nhân có ECG đủ tin cậy để chấm)')
    print('=' * 70)

    for grp, sub in ok.groupby('label'):
        print(f'\n  Nhóm {grp} ({len(sub)} người):')
        print(f'    F1 dò nhịp   : median {sub.f1.median():.3f} | min {sub.f1.min():.3f} | max {sub.f1.max():.3f}')
        print(f'    Sensitivity  : median {sub.sensitivity.median():.3f}')
        print(f'    PPV          : median {sub.ppv.median():.3f}')
        print(f'    HR MAE (bpm) : median {sub.hr_mae_bpm.median():.2f}')

    worst = ok.nsmallest(5, 'f1')[['record_id', 'label', 'f1', 'sensitivity', 'ppv', 'hr_mae_bpm']]
    print('\n  ⚠️ 5 bệnh nhân dò nhịp kém nhất:')
    print(worst.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # Thẩm định 2 ca Normal nghi nhãn sai bằng chính ECG
    suspects = ['mimic_perform_non_af_012', 'mimic_perform_non_af_014']
    print('\n  🔎 THẨM ĐỊNH NHÃN bằng RMSSD/pNN50 tính từ ECG (không qua PPG):')
    normals = df[(df.label == 'Normal') & (~df.record_id.isin(suspects))]
    afs = df[df.label == 'AFib']
    print(f'    Normal khác (median): RMSSD={normals.ecg_RMSSD.median():.0f}ms, pNN50={normals.ecg_pNN50.median():.0f}%')
    print(f'    AFib (median)       : RMSSD={afs.ecg_RMSSD.median():.0f}ms, pNN50={afs.ecg_pNN50.median():.0f}%')
    for rid in suspects:
        r = df[df.record_id == rid]
        if len(r):
            r = r.iloc[0]
            flag = ' (ECG unreliable!)' if r.ecg_unreliable else ''
            print(f'    {rid}: RMSSD={r.ecg_RMSSD:.0f}ms, pNN50={r.ecg_pNN50:.0f}%{flag}')

    print(f'\n💾 Đã lưu: {out_csv}')
    print('\n✅ HOÀN TẤT KIỂM CHỨNG DÒ NHỊP!')


if __name__ == '__main__':
    main()
