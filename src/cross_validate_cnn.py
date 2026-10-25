
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
MAX_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.001
PATIENCE = 4
SEGMENT_LENGTH = 2048

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# These four sessions contain all five candidate BSSIDs.
EVALUATION_SESSIONS = [
    "2025_12_08_12_09_21",
    "2025_12_09_15_58_20",
    "2025_12_09_16_02_52",
    "2025_12_19_08_48_36",
]

# The two additional sessions contain recordings of one BSSID.
# They are used for training, never as a five-class test fold.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# ============================================================
# LOAD RECORDING METADATA
# ============================================================
records = []

for path in sorted(DATA_DIR.glob("*.npz")):
    stem = path.stem.removesuffix("_anechoic_chamber")
    parts = stem.split("_")

    if len(parts) < 8 or not parts[-1].isdigit():
        raise ValueError(f"Unexpected filename: {path.name}")

    session = "_".join(parts[1:-1])

    with np.load(path, allow_pickle=False) as data:
        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())

    if not np.isfinite(cfo):
        raise ValueError(f"Invalid CFO in {path.name}")

    records.append({
        "path": path,
        "label": label,
        "session": session,
        "cfo": cfo,
    })

classes = sorted({r["label"] for r in records})
class_to_index = {name: i for i, name in enumerate(classes)}
num_classes = len(classes)

if len(records) != 206 or num_classes != 5:
    raise ValueError(
        f"Expected 206 recordings and 5 classes; "
        f"found {len(records)} recordings and {num_classes} classes."
    )

for session in EVALUATION_SESSIONS:
    found = {r["label"] for r in records if r["session"] == session}
    if found != set(classes):
        raise ValueError(
            f"Session {session} does not contain all five classes: {found}"
        )

# ============================================================
# SIGNAL SEGMENTATION
# One recording's segments always stay in the same fold.
# ============================================================
def get_segments(record):
    with np.load(record["path"], allow_pickle=False) as data:
        i = data["beaconDataDownSampledI"].astype(np.float32)
        q = data["beaconDataDownSampledQ"].astype(np.float32)

    segments = []

    for start in range(0, len(i) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):
        segment = np.stack([
            i[start:start + SEGMENT_LENGTH],
            q[start:start + SEGMENT_LENGTH],
        ])

        rms = np.sqrt(np.mean(segment ** 2))

        if not np.isfinite(rms) or rms < 1e-8:
            continue

        # Same per-segment RMS normalization used previously.
        segments.append(segment / rms)

    if not segments:
        raise ValueError(f"No valid segments in {record['path'].name}")

    return np.asarray(segments, dtype=np.float32)


def build_segment_arrays(record_list):
    X, y = [], []

    for record in record_list:
        segments = get_segments(record)
        X.extend(segments)
        y.extend(
            [class_to_index[record["label"]]] * len(segments)
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
    )


def make_loader(X, y, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )

# ============================================================
# SMALL REGULARIZED I/Q CNN
# ============================================================
class RFClassifier(nn.Module):
    def __init__(self, output_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, output_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def predict_recording(model, record):
    """Average segment probabilities to make one prediction per recording."""
    model.eval()
    segments = torch.from_numpy(get_segments(record))
    all_probabilities = []

    with torch.no_grad():
        for start in range(0, len(segments), BATCH_SIZE):
            batch = segments[start:start + BATCH_SIZE].to(device)
            probabilities = torch.softmax(model(batch), dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    return np.concatenate(all_probabilities, axis=0).mean(axis=0)


def predict_cfo(train_records, test_records):
    """One CFO value per recording; scaling uses training recordings only."""
    X_train = np.array([r["cfo"] for r in train_records])
    y_train = np.array([
        class_to_index[r["label"]] for r in train_records
    ])
    X_test = np.array([r["cfo"] for r in test_records])

    mu = X_train.mean()
    sd = X_train.std()
    if sd < 1e-8:
        sd = 1.0

    A = (X_train - mu) / sd
    B = (X_test - mu) / sd

    centers = np.array([
        A[y_train == c].mean() for c in range(num_classes)
    ])

    distances = (B[:, None] - centers[None, :]) ** 2
    return distances.argmin(axis=1)


# ============================================================
# FOUR-FOLD SESSION-LEVEL EVALUATION
# ============================================================
all_actual = []
all_cfo_predictions = []
all_cnn_predictions = []
all_results = []

for fold, test_session in enumerate(EVALUATION_SESSIONS):
    # Use a different complete session for early stopping.
    val_session = EVALUATION_SESSIONS[
        (fold + 1) % len(EVALUATION_SESSIONS)
    ]

    test_records = [
        r for r in records if r["session"] == test_session
    ]
    val_records = [
        r for r in records if r["session"] == val_session
    ]
    excluded_sessions = {test_session, val_session}

    # Include the two single-class sessions in training only.
    train_records = [
        r for r in records
        if r["session"] not in excluded_sessions
    ]

    train_classes = {r["label"] for r in train_records}
    if train_classes != set(classes):
        raise ValueError(f"Fold {fold + 1}: training classes incomplete")

    if {r["session"] for r in train_records} & excluded_sessions:
        raise AssertionError("Session leakage detected")

    print(f"\n{'=' * 65}")
    print(f"Fold {fold + 1}/{len(EVALUATION_SESSIONS)}")
    print("Validation session:", val_session)
    print("Test session:", test_session)
    print("Train recordings:", len(train_records))
    print("Validation recordings:", len(val_records))
    print("Test recordings:", len(test_records))

    # CFO baseline: one prediction per test recording.
    cfo_predictions = predict_cfo(train_records, test_records)

    # Build segments for CNN training and validation.
    X_train, y_train = build_segment_arrays(train_records)
    X_val, y_val = build_segment_arrays(val_records)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)

    # Re-seed model initialization for reproducibility.
    fold_seed = SEED + fold
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)

    model = RFClassifier(num_classes).to(device)

    counts = torch.bincount(
        torch.from_numpy(y_train), minlength=num_classes
    ).float()
    weights = counts.sum() / (
        num_classes * counts.clamp(min=1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=0.05,
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

    for epoch in range(MAX_EPOCHS):
        model.train()
        loss_sum = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item() * len(y)

        train_loss = loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        val_actual = []
        val_predicted = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * len(y)

                val_actual.extend(y.cpu().tolist())
                val_predicted.extend(
                    logits.argmax(dim=1).cpu().tolist()
                )

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc = accuracy_score(val_actual, val_predicted)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train loss={train_loss:.4f} | "
                f"val loss={val_loss:.4f} | "
                f"val segment acc={val_acc:.3f}"
            )

        if no_improvement >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced")

    model.load_state_dict(best_state)
    print("Best validation epoch:", best_epoch)

    # Aggregate segment probabilities into one prediction per test recording.
    cnn_predictions = []
    for record in test_records:
        probabilities = predict_recording(model, record)
        cnn_predictions.append(int(np.argmax(probabilities)))

    y_true = np.array([
        class_to_index[r["label"]] for r in test_records
    ])

    cfo_acc = accuracy_score(y_true, cfo_predictions)
    cnn_acc = accuracy_score(y_true, cnn_predictions)

    print(f"CFO recording accuracy: {cfo_acc * 100:.2f}%")
    print(f"CNN recording accuracy: {cnn_acc * 100:.2f}%")

    for record, actual, cfo_pred, cnn_pred in zip(
        test_records, y_true, cfo_predictions, cnn_predictions
    ):
        all_actual.append(actual)
        all_cfo_predictions.append(cfo_pred)
        all_cnn_predictions.append(cnn_pred)

        all_results.append({
            "filename": record["path"].name,
            "session": test_session,
            "actual": classes[actual],
            "cfo_prediction": classes[cfo_pred],
            "cnn_prediction": classes[cnn_pred],
        })

# ============================================================
# AGGREGATE OUT-OF-FOLD RECORDING-LEVEL RESULTS
# ============================================================
all_actual = np.asarray(all_actual)
all_cfo_predictions = np.asarray(all_cfo_predictions)
all_cnn_predictions = np.asarray(all_cnn_predictions)

print("\n" + "=" * 65)
print("POOLED OUT-OF-FOLD RESULTS — RECORDING LEVEL")

for name, predictions in [
    ("CFO-only", all_cfo_predictions),
    ("I/Q CNN", all_cnn_predictions),
]:
    acc = accuracy_score(all_actual, predictions)
    bal_acc = balanced_accuracy_score(all_actual, predictions)

    print(f"\n{name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Balanced accuracy: {bal_acc * 100:.2f}%")
    print(classification_report(
        all_actual,
        predictions,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    ))

    cm = confusion_matrix(
        all_actual, predictions, labels=list(range(num_classes))
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )
    disp.plot(xticks_rotation=45, values_format="d")
    plt.title(f"{name}: Session-Level CV, Recording-Level Predictions")
    plt.tight_layout()

    output_name = (
        "cfo_session_cv_confusion.png"
        if name == "CFO-only"
        else "cnn_session_cv_confusion.png"
    )

    plt.savefig(
        RESULTS_DIR / output_name,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

csv_path = RESULTS_DIR / "session_cv_recording_predictions.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "session",
            "actual",
            "cfo_prediction",
            "cnn_prediction",
        ],
    )
    writer.writeheader()
    writer.writerows(all_results)

print("\nSaved recording-level predictions:", csv_path)
print("Saved confusion matrices in:", RESULTS_DIR)
print("\nSession-level cross-validation completed.")
