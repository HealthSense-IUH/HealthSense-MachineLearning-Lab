"""Biểu đồ dùng lại trong các notebook báo cáo.

Bảng màu thống nhất với bộ slide và tài liệu dự án:
navy = kết quả trung thực, cherry = kết quả bị thổi phồng / AFib.
"""

import matplotlib.pyplot as plt
import numpy as np

NAVY = '#2F3C7E'
CHERRY = '#990011'
GREEN = '#1B7A43'
SLATE = '#64748B'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.25,
})


def leakage_gap(leaky_acc, honest_acc, title='Cái giá của data leakage',
                labels=('Chia ngẫu nhiên\n(bản gốc)', 'LOSO theo bệnh nhân\n(trung thực)')):
    """Hai cột: điểm bản gốc vs điểm sau khi chấm lại trung thực."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    values = [leaky_acc, honest_acc]
    bars = ax.bar(labels, values, color=[CHERRY, NAVY], width=0.55)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f'{v * 100:.1f}%', ha='center', fontweight='bold', fontsize=13)

    gap = leaky_acc - honest_acc
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\nphần điểm ảo: {gap * 100:+.1f} điểm phần trăm',
                 color=CHERRY if gap > 0.02 else GREEN)
    ax.axhline(0.5, color=SLATE, linestyle=':', linewidth=1)
    plt.tight_layout()
    return fig


def subject_probabilities(subject_result, title='Xác suất AFib theo từng bệnh nhân'):
    """Mỗi bệnh nhân một chấm — thấy ngay ai bị chẩn đoán sai."""
    subjects = subject_result['subjects']
    proba = np.array(subject_result['subject_proba'])
    true = np.array(subject_result['subject_true'])

    order = np.argsort(proba)
    proba, true = proba[order], true[order]
    subjects = [subjects[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = [CHERRY if t == 1 else NAVY for t in true]
    ax.scatter(range(len(proba)), proba, c=colors, s=55, zorder=3)
    ax.axhline(0.5, color=SLATE, linestyle='--', linewidth=1.2,
               label='ngưỡng 0.5')

    wrong = [(i, p) for i, (p, t) in enumerate(zip(proba, true))
             if (p >= 0.5) != (t == 1)]
    if wrong:
        ax.scatter([w[0] for w in wrong], [w[1] for w in wrong],
                   facecolors='none', edgecolors='black', s=220, linewidths=1.6,
                   zorder=4, label='chẩn đoán sai')

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([s.replace('mimic_perform_', '') for s in subjects],
                       rotation=90, fontsize=7)
    ax.set_ylabel('P(AFib) trung bình')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'{title}  (đỏ = AFib thật, xanh = bình thường)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig


def version_comparison(rows, title='Bốn phiên bản, chấm bằng cùng một thước đo'):
    """Biểu đồ cột kép so sánh điểm 'bản gốc' và điểm LOSO của cả 4 phiên bản.

    rows: list dict {'version','claimed','honest'}
    """
    versions = [r['version'] for r in rows]
    claimed = [r['claimed'] for r in rows]
    honest = [r['honest'] for r in rows]

    x = np.arange(len(versions))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - w / 2, claimed, w, label='Điểm bản gốc công bố', color=CHERRY)
    b2 = ax.bar(x + w / 2, honest, w, label='Chấm lại bằng LOSO', color=NAVY)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h * 100:.1f}', ha='center', fontsize=9.5,
                        fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
