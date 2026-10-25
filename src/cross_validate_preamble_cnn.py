
from pathlib import Path
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.001
PATIENCE = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SESSIONS = [
    "2025_12_08_12_09_21",
    "2025_12_09_15_58_20",
    "2025_12_09_16_02_52",
    "2025_12_19_08_48_36",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# ============================================================
# LOAD FULL PREAMBLES: ONE EXAMPLE PER RECORDING
# ============================================================
records = []

for path in sorted(DATA_DIR.glob("*.npz")):
    stem = path.stem.removesuffix("_anechoic_chamber")
    parts = stem.split("_")

    if len(parts) < 8 or not parts[-1].isdigit():
        raise ValueError(f"Unexpected filename: {path.name}")

    session = "_".join(parts[1:-1])

    with np.load(path, allow_pickle=False) as data:
        required = {
            "BSSID", "CFO", "beaconPreambleI", "beaconPreambleQ"
        }
        missing = required - set(data.files)

        if missing:
            raise KeyError(f"{path.name}: missing {sorted(missing)}")

        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())
        i = data["beaconPreambleI"].astype(np.float32).reshape(-1)
        q = data["beaconPreambleQ"].astype(np.float32).reshape(-1)

    if len(i) != len(q) or len(i) < 256:
        raise ValueError(f"Invalid preamble length in {path.name}")

    if not (np.isfinite(i).all() and np.isfinite(q).all()):
        raise ValueError(f"Non-finite I/Q values in {path.name}")

    if not np.isfinite(cfo):
        raise ValueError(f"Invalid CFO in {path.name}")

    # Two channels: I and Q.
    signal = np.stack([i, q], axis=0)
    rms = np.sqrt(np.mean(signal ** 2))

    if rms < 1e-8:
        raise ValueError(f"Near-zero preamble in {path.name}")

    # Normalize signal gain, preserving relative I/Q waveform structure.
    signal = (signal / rms).astype(np.float32)

    records.append({
        "filename": path.name,
        "label": label,
        "session": session,
        "cfo": cfo,
        "signal": signal,
    })

if not records:
    raise RuntimeError(f"No NPZ recordings found in {DATA_DIR}")

classes = sorted({r["label"] for r in records})
class_id = {name: i for i, name in enumerate(classes)}
num_classes = len(classes)

lengths = {r["signal"].shape[1] for r in records}
if len(lengths) != 1:
    raise ValueError(
        f"Preamble lengths vary: {sorted(lengths)}. "
        "Add a documented alignment/resampling step before training."
    )

print("Recordings:", len(records))
print("Classes:", classes)
print("Preamble samples per recording:", next(iter(lengths)))

for session in SESSIONS:
    found = {r["label"] for r in records if r["session"] == session}
    if found != set(classes):
        raise ValueError(
            f"Session {session} lacks one or more BSSID classes."
        )

# ============================================================
# MODEL
# ============================================================
class PreambleCNN(nn.Module):
    def __init__(self, class_count):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(8, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(32, class_count),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loader(X, y, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def build_arrays(items):
    X = np.stack([r["signal"] for r in items]).astype(np.float32)
    y = np.array([class_id[r["label"]] for r in items], dtype=np.int64)
    return X, y


# ============================================================
# FOUR-FOLD SESSION-LEVEL EVALUATION
# ============================================================
actual_all = []
cfo_all = []
cnn_all = []
prediction_rows = []

for fold, test_session in enumerate(SESSIONS):
    val_session = SESSIONS[(fold + 1) % len(SESSIONS)]

    excluded = {test_session, val_session}

    train_records = [
        r for r in records if r["session"] not in excluded
    ]
    val_records = [
        r for r in records if r["session"] == val_session
    ]
    test_records = [
        r for r in records if r["session"] == test_session
    ]

    # Two additional sessions contain a single BSSID and are
    # permitted in training only.
    if {r["label"] for r in train_records} != set(classes):
        raise ValueError("Training fold is missing a class.")

    print(f"\n{'=' * 60}")
    print(f"Fold {fold + 1}/4")
    print("Validation session:", val_session)
    print("Test session:", test_session)
    print(
        f"Recordings: train={len(train_records)}, "
        f"validation={len(val_records)}, test={len(test_records)}"
    )

    # ---------------- CFO baseline ----------------
    cfo_train = np.array([r["cfo"] for r in train_records])
    y_train_rec = np.array([
        class_id[r["label"]] for r in train_records
    ])
    cfo_test = np.array([r["cfo"] for r in test_records])

    mu, sd = cfo_train.mean(), cfo_train.std()
    if sd < 1e-8:
        sd = 1.0

    A = (cfo_train - mu) / sd
    B = (cfo_test - mu) / sd

    centers = np.array([
        A[y_train_rec == c].mean() for c in range(num_classes)
    ])

    cfo_predictions = (
        ((B[:, None] - centers[None, :]) ** 2).argmin(axis=1)
    )

    # ---------------- CNN training ----------------
    X_train, y_train = build_arrays(train_records)
    X_val, y_val = build_arrays(val_records)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)

    fold_seed = SEED + fold
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)

    model = PreambleCNN(num_classes).to(device)

    counts = torch.bincount(
        torch.from_numpy(y_train),
        minlength=num_classes,
    ).float()

    weights = counts.sum() / (
        num_classes * counts.clamp(min=1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(y)

        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                loss = criterion(model(X), y)
                val_loss_sum += loss.item() * len(y)

        val_loss = val_loss_sum / len(val_loader.dataset)
        scheduler.step(val_loss)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"train loss={train_loss:.4f} | "
                f"val loss={val_loss:.4f}"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)
    model.eval()

    # One prediction per full preamble/recording.
    cnn_predictions = []

    with torch.no_grad():
        for record in test_records:
            signal = torch.from_numpy(
                record["signal"][None, ...]
            ).to(device)

            probabilities = torch.softmax(model(signal), dim=1)
            cnn_predictions.append(int(probabilities.argmax(dim=1).item()))

    y_true = np.array([
        class_id[r["label"]] for r in test_records
    ])

    print("Best epoch:", best_epoch)
    print(
        f"CFO accuracy: "
        f"{accuracy_score(y_true, cfo_predictions) * 100:.2f}%"
    )
    print(
        f"Preamble CNN accuracy: "
        f"{accuracy_score(y_true, cnn_predictions) * 100:.2f}%"
    )

    actual_all.extend(y_true.tolist())
    cfo_all.extend(cfo_predictions.tolist())
    cnn_all.extend(cnn_predictions)

    for record, actual, cfo_pred, cnn_pred in zip(
        test_records, y_true, cfo_predictions, cnn_predictions
    ):
        prediction_rows.append({
            "filename": record["filename"],
            "session": test_session,
            "actual": classes[actual],
            "cfo_prediction": classes[cfo_pred],
            "preamble_cnn_prediction": classes[cnn_pred],
        })

# ============================================================
# POOLED OUT-OF-FOLD RESULTS
# ============================================================
actual_all = np.asarray(actual_all)
cfo_all = np.asarray(cfo_all)
cnn_all = np.asarray(cnn_all)

for name, predictions in [
    ("CFO-only metadata baseline", cfo_all),
    ("Preamble-only CNN", cnn_all),
]:
    print(f"\n========== {name} ==========")
    print(f"Accuracy: {accuracy_score(actual_all, predictions) * 100:.2f}%")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(actual_all, predictions) * 100:.2f}%"
    )

    print(classification_report(
        actual_all,
        predictions,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    ))

    cm = confusion_matrix(
        actual_all,
        predictions,
        labels=list(range(num_classes)),
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )
    disp.plot(xticks_rotation=45, values_format="d")
    plt.title(f"{name}: Session-Level CV")
    plt.tight_layout()

    filename = (
        "preamble_cfo_cv_confusion.png"
        if name == "CFO-only metadata baseline"
        else "preamble_cnn_cv_confusion.png"
    )

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

csv_path = RESULTS_DIR / "preamble_cnn_cv_predictions.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "session",
            "actual",
            "cfo_prediction",
            "preamble_cnn_prediction",
        ],
    )
    writer.writeheader()
    writer.writerows(prediction_rows)

print("\nSaved predictions:", csv_path)
print("Session-level preamble CNN evaluation completed.")
