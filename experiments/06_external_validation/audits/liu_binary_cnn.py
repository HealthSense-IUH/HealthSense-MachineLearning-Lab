from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd

from scipy.io import loadmat

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
)


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


ROOT = Path(
    "data/raw/PPGArrhythmiaDetection-main"
)

DATA_DIR = (
    ROOT / "valid_testDataset"
)


LABEL_NAMES = {
    0: "SR",
    1: "PVC",
    2: "PAC",
    3: "VT",
    4: "SVT",
    5: "AF",
}


TASKS = {
    "AF_vs_SR": {
        "positive": {5},
        "negative": {0},
    },

    "AF_vs_ALL": {
        "positive": {5},
        "negative": {0, 1, 2, 3, 4},
    },

    "AF_vs_PAC_PVC": {
        "positive": {5},
        "negative": {1, 2},
    },
}


def read_list(path):

    return [
        x.strip()
        .strip("'")
        .strip('"')

        for x in (
            path.read_text()
            .splitlines()
        )

        if x.strip()
    ]


VALID_SUBJECTS = read_list(
    ROOT / "validsubjects.txt"
)

TEST_SUBJECTS = read_list(
    ROOT / "testsubjects.txt"
)


def load_subjects(subjects):

    rows = []

    for filename in subjects:

        path = DATA_DIR / filename

        mat = loadmat(
            path,
            squeeze_me=True,
        )

        ppg = np.asarray(
            mat["ppgseg"],
            dtype=np.float32,
        )

        labels = np.asarray(
            mat["labels"]
        ).reshape(-1).astype(int)

        for i, label in enumerate(
            labels
        ):

            rows.append({
                "subject":
                    path.stem,

                "segment":
                    i,

                "label":
                    label,

                "ppg":
                    ppg[i],
            })

    return rows


print(
    "Loading validation subjects..."
)

VALID_ROWS = load_subjects(
    VALID_SUBJECTS
)


print(
    "Loading test subjects..."
)

TEST_ROWS = load_subjects(
    TEST_SUBJECTS
)


class PPGDataset(Dataset):

    def __init__(
        self,
        rows,
        task,
    ):

        positive = task[
            "positive"
        ]

        negative = task[
            "negative"
        ]

        self.items = []

        for r in rows:

            label = r["label"]

            if label in positive:
                y = 1

            elif label in negative:
                y = 0

            else:
                continue

            self.items.append(
                (
                    r["ppg"],
                    y,
                    r["subject"],
                )
            )

    def __len__(self):
        return len(
            self.items
        )

    def __getitem__(self, idx):

        x, y, subject = (
            self.items[idx]
        )

        # Already public-dataset
        # normalized to [0,1].
        x = torch.from_numpy(
            x
        ).float().unsqueeze(0)

        y = torch.tensor(
            y,
            dtype=torch.float32,
        )

        return (
            x,
            y,
            subject,
        )


class SmallPPGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1, 32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32, 64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64, 128,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                128, 128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        return (
            self.classifier(x)
            .squeeze(1)
        )


def evaluate(
    model,
    loader,
):

    model.eval()

    ys = []
    probs = []
    subjects = []

    with torch.no_grad():

        for x, y, subject in loader:

            x = x.to(
                DEVICE
            )

            logits = model(x)

            prob = torch.sigmoid(
                logits
            )

            ys.extend(
                y.numpy()
            )

            probs.extend(
                prob.cpu().numpy()
            )

            subjects.extend(
                subject
            )

    return (
        np.asarray(ys),
        np.asarray(probs),
        np.asarray(subjects),
    )


def metrics(
    y,
    prob,
):

    pred = (
        prob >= 0.5
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        ).ravel()
    )

    return {
        "Accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "Sensitivity":
            recall_score(
                y,
                pred,
                zero_division=0,
            ),

        "Specificity":
            (
                tn
                / (tn + fp)
                if tn + fp
                else np.nan
            ),

        "Precision":
            precision_score(
                y,
                pred,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y,
                pred,
                zero_division=0,
            ),

        "ROC_AUC":
            roc_auc_score(
                y,
                prob,
            ),

        "PR_AUC":
            average_precision_score(
                y,
                prob,
            ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def subject_macro_auc(
    y,
    prob,
    subjects,
):

    aucs = []

    for subject in np.unique(
        subjects
    ):

        mask = (
            subjects == subject
        )

        ys = y[mask]

        if len(
            np.unique(ys)
        ) < 2:
            continue

        aucs.append(
            roc_auc_score(
                ys,
                prob[mask],
            )
        )

    return (
        float(
            np.mean(aucs)
        ),
        float(
            np.median(aucs)
        ),
        len(aucs),
    )


def train_task(
    task_name,
    task,
):

    print()
    print("=" * 100)
    print(task_name)
    print("=" * 100)

    # --------------------------------------------------
    # Split ONLY the 46 official validation subjects
    # into development-train and early-stop validation.
    # Official 45 test subjects stay untouched.
    # --------------------------------------------------

    subjects = np.array(
        VALID_SUBJECTS
    )

    dummy = np.zeros(
        len(subjects)
    )

    splitter = (
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=SEED,
        )
    )

    tr_idx, es_idx = next(
        splitter.split(
            subjects,
            dummy,
            groups=subjects,
        )
    )

    train_subjects = set(
        subjects[tr_idx]
    )

    early_subjects = set(
        subjects[es_idx]
    )

    train_rows = [
        x
        for x in VALID_ROWS
        if (
            x["subject"] + ".mat"
        ) in train_subjects
    ]

    early_rows = [
        x
        for x in VALID_ROWS
        if (
            x["subject"] + ".mat"
        ) in early_subjects
    ]

    train_ds = PPGDataset(
        train_rows,
        task,
    )

    early_ds = PPGDataset(
        early_rows,
        task,
    )

    test_ds = PPGDataset(
        TEST_ROWS,
        task,
    )

    print(
        "Development train segments:",
        len(train_ds),
    )

    print(
        "Early-stop segments:",
        len(early_ds),
    )

    print(
        "Official test segments:",
        len(test_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    early_loader = DataLoader(
        early_ds,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ----------------------------------------------
    # Calculate class weight on development train only
    # ----------------------------------------------

    train_labels = np.array(
        [
            item[1]
            for item in train_ds.items
        ]
    )

    pos = np.sum(
        train_labels == 1
    )

    neg = np.sum(
        train_labels == 0
    )

    pos_weight = torch.tensor(
        neg / pos,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        "Train positive:",
        pos,
    )

    print(
        "Train negative:",
        neg,
    )

    print(
        "pos_weight:",
        float(pos_weight),
    )

    model = SmallPPGCNN().to(
        DEVICE
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_auc = -np.inf
    best_epoch = None
    best_state = None

    patience = 10
    stale = 0

    MAX_EPOCHS = 80

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        for x, y, _ in (
            train_loader
        ):

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(y)
            )

        val_y, val_prob, _ = (
            evaluate(
                model,
                early_loader,
            )
        )

        val_auc = roc_auc_score(
            val_y,
            val_prob,
        )

        train_loss = (
            total_loss
            / len(train_ds)
        )

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.5f} "
            f"val_auc={val_auc:.5f}"
        )

        if val_auc > (
            best_auc + 1e-4
        ):

            best_auc = val_auc
            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            stale = 0

        else:
            stale += 1

        if stale >= patience:
            break

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best development AUC:",
        best_auc,
    )

    # ----------------------------------------------
    # Do NOT tune anything using official test.
    # Evaluate best development checkpoint once.
    # ----------------------------------------------

    model.load_state_dict(
        best_state
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    y, prob, subjects = evaluate(
        model,
        test_loader,
    )

    result = metrics(
        y,
        prob,
    )

    macro_mean, macro_median, n_subjects = (
        subject_macro_auc(
            y,
            prob,
            subjects,
        )
    )

    result[
        "Subject_Macro_AUC_Mean"
    ] = macro_mean

    result[
        "Subject_Macro_AUC_Median"
    ] = macro_median

    result[
        "Subjects_With_Both_Classes"
    ] = n_subjects

    print()
    print("OFFICIAL TEST RESULTS")

    for k, v in result.items():
        print(
            f"{k}: {v}"
        )

    pred_df = pd.DataFrame({
        "status":
            y.astype(int),

        "prob":
            prob,

        "subject":
            subjects,
    })

    pred_df.to_csv(
        "experiments/"
        "06_external_validation/"
        f"liu_{task_name}_predictions.csv",
        index=False,
    )

    return {
        "Task": task_name,
        **result,
    }


results = []

for task_name, task in (
    TASKS.items()
):

    results.append(
        train_task(
            task_name,
            task,
        )
    )


summary = pd.DataFrame(
    results
)

summary.to_csv(
    "experiments/"
    "06_external_validation/"
    "liu_binary_cnn_summary.csv",
    index=False,
)


print()
print("=" * 120)
print("LIU BINARY CNN SUMMARY")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)
