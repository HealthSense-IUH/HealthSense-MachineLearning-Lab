"""
MIMIC-III v3 Benchmark Pipeline
================================
Pipeline hoàn chỉnh theo gợi ý mentor:
1. Chia dữ liệu Train / Validation / Test (70/15/15)
2. 3 Mô hình: Logistic Regression, Random Forest, XGBoost
3. Hyperparameter Tuning trên Validation set
4. Loss Curve, Confusion Matrix Heatmap, ROC Curve
5. Bảng tổng hợp benchmark kết quả

Sử dụng dữ liệu đã lọc Outlier (_clean files).
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Fix Unicode encoding on Windows (cp1252 -> utf-8)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ============================================================
# Cấu hình
# ============================================================
SCALES = ['1360', '4083', '8165', '16358']
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # 15% of total -> ~17.6% of remaining after test split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models', 'mimic', 'benchmark_v3')

# Plotting config
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

# Colors
COLORS = {
    'Logistic Regression': '#3b82f6',
    'Random Forest': '#10b981',
    'XGBoost': '#f59e0b'
}


def load_data(scale):
    """Load dữ liệu đã lọc outlier (clean). Fallback sang dữ liệu chưa lọc nếu không có."""

    # Ưu tiên file clean
    mm_clean = os.path.join(PROCESSED_DIR, f'mimic_minmax_{scale}_clean.csv')
    zs_clean = os.path.join(PROCESSED_DIR, f'mimic_zscore_{scale}_clean.csv')

    if os.path.exists(mm_clean) and os.path.exists(zs_clean):
        df_mm = pd.read_csv(mm_clean)
        df_zs = pd.read_csv(zs_clean)
        data_source = 'clean (outlier-filtered)'
    else:
        # Fallback
        mm_file = os.path.join(PROCESSED_DIR, f'mimic_minmax_{scale}.csv')
        zs_file = os.path.join(PROCESSED_DIR, f'mimic_zscore_{scale}.csv')
        if not os.path.exists(mm_file):
            return None
        df_mm = pd.read_csv(mm_file)
        df_zs = pd.read_csv(zs_file)
        data_source = 'original (no outlier filter)'

    return df_mm, df_zs, data_source


SCALE_OVERLAP = {'1360': 0.0, '4083': 0.66, '8165': 0.83, '16358': 0.91}


def split_data_anti_leakage(df_mm, df_zs, scale_tag):
    """
    Chia dữ liệu chống Data Leakage:
    - Scale 1360 (0% overlap): random stratified split
    - Scale khác (có overlap): stratified contiguous block split (chia 70/15/15 theo block liên tục cho từng nhãn)
    """
    y = df_mm['status']
    X_mm = df_mm.drop(columns=['status'])
    X_zs = df_zs.drop(columns=['status'])

    overlap = SCALE_OVERLAP.get(scale_tag, 0)

    if overlap == 0:
        # Random stratified split (không có overlap -> không bị leakage)
        Xmm_temp, Xmm_test, y_temp, y_test = train_test_split(
            X_mm, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        Xzs_temp, Xzs_test, _, _ = train_test_split(
            X_zs, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        Xmm_train, Xmm_val, y_train, y_val = train_test_split(
            Xmm_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp)
        Xzs_train, Xzs_val, _, _ = train_test_split(
            Xzs_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp)
        split_method = 'Random Stratified Split'
    else:
        # Stratified Contiguous Block Split (chống data leakage nhưng vẫn đảm bảo đủ 2 nhãn)
        train_idx, val_idx, test_idx = [], [], []

        for label in sorted(y.unique()):
            label_indices = y[y == label].index.tolist()
            n_label = len(label_indices)
            t_end = int(n_label * 0.70)
            v_end = int(n_label * 0.85)

            train_idx.extend(label_indices[:t_end])
            val_idx.extend(label_indices[t_end:v_end])
            test_idx.extend(label_indices[v_end:])

        Xmm_train, Xmm_val, Xmm_test = X_mm.loc[train_idx], X_mm.loc[val_idx], X_mm.loc[test_idx]
        Xzs_train, Xzs_val, Xzs_test = X_zs.loc[train_idx], X_zs.loc[val_idx], X_zs.loc[test_idx]
        y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]

        split_method = f'Stratified Contiguous Block Split (overlap={overlap*100:.0f}%)'

    return {
        'mm': {'train': Xmm_train, 'val': Xmm_val, 'test': Xmm_test},
        'zs': {'train': Xzs_train, 'val': Xzs_val, 'test': Xzs_test},
        'y':  {'train': y_train,   'val': y_val,    'test': y_test},
        'split_method': split_method
    }


def train_logistic_regression(data):
    """Huấn luyện Logistic Regression trên Z-Score data với Grid Search."""

    X_train = data['zs']['train']
    X_val   = data['zs']['val']
    X_test  = data['zs']['test']
    y_train = data['y']['train']
    y_val   = data['y']['val']
    y_test  = data['y']['test']

    # Grid Search trên Validation
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [1000],
        'solver': ['saga'],
        'random_state': [RANDOM_STATE]
    }

    best_score = -1
    best_model = None
    best_params = None
    loss_history = []

    for C in param_grid['C']:
        model = LogisticRegression(
            C=C, max_iter=1000, solver='saga',
            random_state=RANDOM_STATE, penalty='l2'
        )
        model.fit(X_train, y_train)

        val_score = accuracy_score(y_val, model.predict(X_val))
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        loss_history.append({
            'C': C, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_accuracy': val_score
        })

        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_params = {'C': C}

    # Evaluation trên Test set
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': best_model,
        'name': 'Logistic Regression',
        'best_params': best_params,
        'test_preds': test_preds,
        'test_probs': test_probs,
        'y_test': y_test,
        'loss_history': loss_history,
        'X_data': 'zscore'
    }


def train_random_forest(data):
    """Huấn luyện Random Forest trên MinMax data với Grid Search."""

    X_train = data['mm']['train']
    X_val   = data['mm']['val']
    X_test  = data['mm']['test']
    y_train = data['y']['train']
    y_val   = data['y']['val']
    y_test  = data['y']['test']

    # Grid Search trên Validation
    param_grid = [
        {'n_estimators': 50,  'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 150, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 7},
        {'n_estimators': 150, 'max_depth': 7},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 150, 'max_depth': 10},
    ]

    best_score = -1
    best_model = None
    best_params = None
    loss_history = []

    for params in param_grid:
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        val_score = accuracy_score(y_val, model.predict(X_val))
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        loss_history.append({
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_score,
            'oob_score': model.oob_score_
        })

        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_params = params

    # Evaluation trên Test set
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': best_model,
        'name': 'Random Forest',
        'best_params': best_params,
        'test_preds': test_preds,
        'test_probs': test_probs,
        'y_test': y_test,
        'loss_history': loss_history,
        'X_data': 'minmax'
    }


def train_xgboost(data):
    """Huấn luyện XGBoost trên MinMax data với Grid Search + Early Stopping."""

    X_train = data['mm']['train']
    X_val   = data['mm']['val']
    X_test  = data['mm']['test']
    y_train = data['y']['train']
    y_val   = data['y']['val']
    y_test  = data['y']['test']

    # Grid Search trên Validation
    param_grid = [
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 200},
        {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 150},
        {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 200},
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 150},
        {'max_depth': 5, 'learning_rate': 0.1,  'n_estimators': 100},
        {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 150},
    ]

    best_score = -1
    best_model = None
    best_params = None
    best_evals = None
    all_evals = []

    for params in param_grid:
        model = xgb.XGBClassifier(
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            early_stopping_rounds=20,
            verbosity=0
        )

        # Train with eval set for loss curve
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        val_score = accuracy_score(y_val, model.predict(X_val))

        evals_result = model.evals_result()
        all_evals.append({
            'params': params,
            'val_accuracy': val_score,
            'train_loss': evals_result['validation_0']['logloss'],
            'val_loss': evals_result['validation_1']['logloss']
        })

        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_params = params
            best_evals = evals_result

    # Evaluation trên Test set
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': best_model,
        'name': 'XGBoost',
        'best_params': best_params,
        'test_preds': test_preds,
        'test_probs': test_probs,
        'y_test': y_test,
        'loss_history': all_evals,
        'best_evals': best_evals,
        'X_data': 'minmax'
    }


def compute_metrics(result):
    """Tính toàn bộ metrics đánh giá từ kết quả predict."""

    y_true = result['y_test']
    y_pred = result['test_preds']
    y_prob = result['test_probs']

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'Model': result['name'],
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall (Sensitivity)': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'False Negative (FN)': fn,
        'True Positive (TP)': tp,
        'True Negative (TN)': tn,
        'False Positive (FP)': fp,
        'Best Params': str(result['best_params']),
        'Data Type': result['X_data']
    }


def plot_confusion_matrix(result, scale, save_dir):
    """Vẽ Confusion Matrix heatmap cho 1 mô hình."""

    y_true = result['y_test']
    y_pred = result['test_preds']
    name = result['name']

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal (0)', 'AFib (1)'],
                yticklabels=['Normal (0)', 'AFib (1)'],
                annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(f'Confusion Matrix - {name}\n(Scale: {scale} samples)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    filename = f'confusion_matrix_{name.lower().replace(" ", "_")}_{scale}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    📊 Saved: {filename}')
    return filepath


def plot_xgboost_loss_curve(result, scale, save_dir):
    """Vẽ Loss Curve (Train vs Validation) cho XGBoost."""

    evals = result['best_evals']
    if not evals:
        return None

    train_loss = evals['validation_0']['logloss']
    val_loss = evals['validation_1']['logloss']
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label='Train Loss', color=COLORS['XGBoost'], linewidth=2)
    ax.plot(epochs, val_loss, label='Validation Loss', color='#ef4444', linewidth=2, linestyle='--')
    ax.set_xlabel('Boosting Rounds', fontsize=13)
    ax.set_ylabel('Log Loss', fontsize=13)
    ax.set_title(f'XGBoost Loss Curve - Train vs Validation\n(Scale: {scale} samples)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'loss_curve_xgboost_{scale}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    📈 Saved: {filename}')
    return filepath


def plot_lr_tuning_curve(result, scale, save_dir):
    """Vẽ Loss curve theo hyperparameter C cho Logistic Regression."""

    history = result['loss_history']
    if not history:
        return None

    cs = [h['C'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(cs)), train_losses, 'o-', label='Train Loss',
            color=COLORS['Logistic Regression'], linewidth=2, markersize=8)
    ax.plot(range(len(cs)), val_losses, 's--', label='Validation Loss',
            color='#ef4444', linewidth=2, markersize=8)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([f'C={c}' for c in cs], fontsize=11)
    ax.set_xlabel('Regularization Parameter C', fontsize=13)
    ax.set_ylabel('Log Loss', fontsize=13)
    ax.set_title(f'Logistic Regression - Hyperparameter Tuning\n(Scale: {scale} samples)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'loss_curve_logistic_regression_{scale}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    📈 Saved: {filename}')
    return filepath


def plot_rf_tuning_curve(result, scale, save_dir):
    """Vẽ Loss curve theo hyperparameter cho Random Forest."""

    history = result['loss_history']
    if not history:
        return None

    labels = [f'n={h["n_estimators"]}\nd={h["max_depth"]}' for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    oob_scores = [h['oob_score'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curve
    ax1.plot(range(len(labels)), train_losses, 'o-', label='Train Loss',
             color=COLORS['Random Forest'], linewidth=2, markersize=8)
    ax1.plot(range(len(labels)), val_losses, 's--', label='Validation Loss',
             color='#ef4444', linewidth=2, markersize=8)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_xlabel('Hyperparameters', fontsize=13)
    ax1.set_ylabel('Log Loss', fontsize=13)
    ax1.set_title('Loss: Train vs Validation', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # OOB Score
    ax2.bar(range(len(labels)), oob_scores, color=COLORS['Random Forest'], alpha=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel('Hyperparameters', fontsize=13)
    ax2.set_ylabel('OOB Score', fontsize=13)
    ax2.set_title('Out-of-Bag (OOB) Score', fontsize=13, fontweight='bold')
    ax2.set_ylim(min(oob_scores) - 0.02, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Random Forest - Hyperparameter Tuning\n(Scale: {scale} samples)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = f'loss_curve_random_forest_{scale}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    📈 Saved: {filename}')
    return filepath


def plot_roc_curves(results, scale, save_dir):
    """Vẽ ROC Curve overlay 3 mô hình trên cùng 1 đồ thị."""

    fig, ax = plt.subplots(figsize=(10, 8))

    for result in results:
        name = result['name']
        y_true = result['y_test']
        y_prob = result['test_probs']

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})',
                color=COLORS[name], linewidth=2.5)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5000)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title(f'ROC Curve Comparison - 3 Models\n(Scale: {scale} samples)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    filename = f'roc_curve_comparison_{scale}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    📊 Saved: {filename}')
    return filepath


def plot_all_confusion_matrices(results, scale, save_dir):
    """Vẽ 3 Confusion Matrix cạnh nhau trên 1 figure."""

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, result in enumerate(results):
        y_true = result['y_test']
        y_pred = result['test_preds']
        name = result['name']

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Normal', 'AFib'],
                    yticklabels=['Normal', 'AFib'],
                    annot_kws={'size': 16, 'weight': 'bold'})
        axes[idx].set_xlabel('Predicted', fontsize=12)
        axes[idx].set_ylabel('True', fontsize=12)
        axes[idx].set_title(f'{name}', fontsize=13, fontweight='bold',
                           color=COLORS[name])

    fig.suptitle(f'Confusion Matrices - All 3 Models (Scale: {scale} samples)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    filename = f'confusion_matrices_all_{scale}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    📊 Saved: {filename}')
    return filepath


def benchmark_one_scale(scale):
    """Chạy toàn bộ benchmark cho 1 quy mô dữ liệu."""

    print(f'\n{"="*70}')
    print(f'🚀 BENCHMARK QUY MÔ {scale} MẪU')
    print(f'{"="*70}')

    # 1. Load data
    result = load_data(scale)
    if result is None:
        print(f'  ❌ Không tìm thấy dữ liệu cho quy mô {scale}!')
        return None

    df_mm, df_zs, data_source = result
    print(f'  📂 Nguồn dữ liệu: {data_source}')
    print(f'  📊 Tổng mẫu: {len(df_mm)}')

    # 2. Split data
    data = split_data_anti_leakage(df_mm, df_zs, scale)
    print(f'  ✂️  [{data["split_method"]}] Train: {len(data["y"]["train"])} | Val: {len(data["y"]["val"])} | Test: {len(data["y"]["test"])}')

    # 3. Train 3 models
    print(f'\n  🏋️ Huấn luyện Logistic Regression...')
    lr_result = train_logistic_regression(data)
    print(f'    Best params: {lr_result["best_params"]}')

    print(f'  🏋️ Huấn luyện Random Forest...')
    rf_result = train_random_forest(data)
    print(f'    Best params: {rf_result["best_params"]}')

    print(f'  🏋️ Huấn luyện XGBoost...')
    xgb_result = train_xgboost(data)
    print(f'    Best params: {xgb_result["best_params"]}')

    all_results = [lr_result, rf_result, xgb_result]

    # 4. Compute metrics
    metrics_list = []
    for res in all_results:
        m = compute_metrics(res)
        m['Scale'] = scale
        metrics_list.append(m)
        print(f'  ✅ {res["name"]}: Accuracy={m["Accuracy"]:.4f}, Recall={m["Recall (Sensitivity)"]:.4f}, '
              f'F1={m["F1-Score"]:.4f}, ROC-AUC={m["ROC-AUC"]:.4f}')

    # 5. Generate plots
    scale_dir = os.path.join(OUTPUT_DIR, f'scale_{scale}')
    os.makedirs(scale_dir, exist_ok=True)

    print(f'\n  🎨 Tạo biểu đồ...')

    # Confusion matrices (individual + combined)
    for res in all_results:
        plot_confusion_matrix(res, scale, scale_dir)
    plot_all_confusion_matrices(all_results, scale, scale_dir)

    # Loss curves
    plot_lr_tuning_curve(lr_result, scale, scale_dir)
    plot_rf_tuning_curve(rf_result, scale, scale_dir)
    plot_xgboost_loss_curve(xgb_result, scale, scale_dir)

    # ROC Curve overlay
    plot_roc_curves(all_results, scale, scale_dir)

    # 6. Save best models
    for res in all_results:
        model_filename = f'{res["name"].lower().replace(" ", "_")}_{scale}.pkl'
        model_path = os.path.join(scale_dir, model_filename)
        joblib.dump(res['model'], model_path)
        print(f'    💾 Model saved: {model_filename}')

    return metrics_list


def create_summary_table(all_metrics):
    """Tạo bảng tổng hợp kết quả benchmark."""

    df = pd.DataFrame(all_metrics)

    # Sắp xếp theo Scale và Accuracy
    df = df.sort_values(by=['Scale', 'Accuracy'], ascending=[True, False]).reset_index(drop=True)

    # Thêm Rank trong từng Scale
    ranks = []
    for scale in SCALES:
        scale_df = df[df['Scale'] == scale]
        for i, (idx, _) in enumerate(scale_df.iterrows()):
            badges = ['🥇', '🥈', '🥉']
            ranks.append(f'{badges[i]} #{i+1}' if i < 3 else f'#{i+1}')
    df['Rank'] = ranks

    # Reorder columns
    display_cols = ['Scale', 'Rank', 'Model', 'Accuracy', 'Recall (Sensitivity)',
                    'Precision', 'F1-Score', 'ROC-AUC', 'Specificity',
                    'False Negative (FN)', 'Best Params']
    df_display = df[display_cols]

    return df, df_display


def plot_cross_scale_comparison(all_metrics, save_dir):
    """Vẽ biểu đồ so sánh 3 model qua 4 quy mô dữ liệu."""

    df = pd.DataFrame(all_metrics)

    metrics_to_plot = ['Accuracy', 'Recall (Sensitivity)', 'F1-Score', 'ROC-AUC']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            model_df = df[df['Model'] == model_name]
            ax.plot(model_df['Scale'].astype(str), model_df[metric],
                    'o-', label=model_name, color=COLORS[model_name],
                    linewidth=2.5, markersize=10)

        ax.set_xlabel('Dataset Scale (Samples)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(min(df[metric].min() - 0.02, 0.90), 1.01)

    fig.suptitle('Cross-Scale Performance Comparison\n(3 Models × 4 Scales)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = 'cross_scale_comparison.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  📊 Saved: {filename}')
    return filepath


def main():
    print('=' * 70)
    print('🔬 MIMIC-III v3 BENCHMARK PIPELINE')
    print('   Models: Logistic Regression, Random Forest, XGBoost')
    print('   Split:  Train 70% | Validation 15% | Test 15%')
    print('   Data:   Outlier-filtered (IQR)')
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []

    for scale in SCALES:
        metrics = benchmark_one_scale(scale)
        if metrics:
            all_metrics.extend(metrics)

    if not all_metrics:
        print('\n❌ Không có kết quả nào! Kiểm tra lại dữ liệu.')
        return

    # Tạo bảng tổng hợp
    print(f'\n{"="*70}')
    print('📋 BẢNG TỔNG HỢP BENCHMARK')
    print(f'{"="*70}')

    df_full, df_display = create_summary_table(all_metrics)

    # Hiển thị bảng
    for scale in SCALES:
        scale_df = df_display[df_display['Scale'] == scale]
        if len(scale_df) == 0:
            continue
        print(f'\n📊 QUY MÔ {scale} MẪU:')
        print(scale_df.to_string(index=False))

    # Lưu CSV
    results_csv = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
    df_full.to_csv(results_csv, index=False)
    print(f'\n  💾 Kết quả đã lưu: {results_csv}')

    # Vẽ biểu đồ so sánh chéo
    print(f'\n  🎨 Tạo biểu đồ so sánh chéo...')
    plot_cross_scale_comparison(all_metrics, OUTPUT_DIR)

    # Tìm model tốt nhất tổng thể
    best = df_full.loc[df_full['Accuracy'].idxmax()]
    print(f'\n🏆 MODEL TỐT NHẤT TỔNG THỂ:')
    print(f'   {best["Model"]} @ Scale {best["Scale"]}')
    print(f'   Accuracy: {best["Accuracy"]:.4f}')
    print(f'   Recall:   {best["Recall (Sensitivity)"]:.4f}')
    print(f'   F1-Score: {best["F1-Score"]:.4f}')
    print(f'   ROC-AUC:  {best["ROC-AUC"]:.4f}')

    print(f'\n✅ HOÀN TẤT BENCHMARK!')
    print(f'   Kết quả tại: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
