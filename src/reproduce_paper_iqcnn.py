
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
# PAPER-ALIGNED IQCNN CONFIGURATION: FIVE DEVICES
# ============================================================
SEED = 42
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 0.000635

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
print("Architecture: 2-layer IQCNN")
print("Filters: 12, 8 | Kernels: 22, 39")
print("Fully connected layer: 128 neurons")
print(f"Adam learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE} | Epochs: {EPOCHS}")

# ============================================================
# LOAD COMPLETE EXTRACTED BEACONS
# ============================================================
records = []

for path in sorted(DATA_DIR.glob("*.npz")):
    stem = path.stem.removesuffix("_anechoic_chamber")
    parts = stem.split("_")

    if len(parts) < 8 or not parts[-1].isdigit():
        raise ValueError(f"Unexpected filename format: {path.name}")

    session = "_".join(parts[1:-1])

    with np.load(path, allow_pickle=False) as data:
        required = {
            "BSSID",
            "CFO",
            "beaconDataDownSampledI",
            "beaconDataDownSampledQ",
        }

        missing = required - set(data.files)
        if missing:
            raise KeyError(
                f"{path.name}: missing fields {sorted(missing)}"
            )

        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())

        i = data["beaconDataDownSampledI"].astype(np.float32).reshape(-1)
        q = data["beaconDataDownSampledQ"].astype(np.float32).reshape(-1)

    if len(i) != len(q) or len(i) < 1000:
        raise ValueError(f"Invalid beacon I/Q length: {path.name}")

    if not (np.isfinite(i).all() and np.isfinite(q).all()):
        raise ValueError(f"Non-finite I/Q samples: {path.name}")

    if not np.isfinite(cfo):
        raise ValueError(f"Invalid CFO metadata: {path.name}")

    signal = np.stack([i, q], axis=0)

    # Normalize once per complete beacon, not every 2048-sample segment.
    rms = np.sqrt(np.mean(signal ** 2))
    if not np.isfinite(rms) or rms < 1e-8:
        raise ValueError(f"Invalid beacon RMS: {path.name}")

    signal = (signal / rms).astype(np.float32)

    records.append({
        "filename": path.name,
        "label": label,
        "session": session,
        "cfo": cfo,
        "signal": signal,
    })

if not records:
    raise FileNotFoundError(f"No NPZ recordings found in {DATA_DIR}")

classes = sorted({r["label"] for r in records})
class_to_index = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

lengths = {r["signal"].shape[1] for r in records}
if len(lengths) != 1:
    raise ValueError(
        "Beacon lengths differ. A defined alignment/resampling "
        f"procedure is required. Found lengths: {sorted(lengths)}"
    )

print("\nRecordings:", len(records))
print("Classes:", classes)
print("Samples per complete beacon:", next(iter(lengths)))

if num_classes != 5:
    raise ValueError(f"Expected five BSSID classes; found {num_classes}")

for session in SESSIONS:
    found = {r["label"] for r in records if r["session"] == session}
    if found != set(classes):
        raise ValueError(
            f"Session {session} does not contain all five classes."
        )

# ============================================================
# TWO-CONVOLUTION IQCNN
# Paper architecture: Conv1d -> ReLU -> MaxPool twice,
# followed by a 128-neuron fully connected layer.
# ============================================================
class IQCNN(nn.Module):
    def __init__(self, input_length, output_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=12,
            kernel_size=22,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=12,
            out_channels=8,
            kernel_size=39,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dynamically determine the flattened feature dimension.
        with torch.no_grad():
            dummy = torch.zeros(1, 2, input_length)
            flattened_size = self._extract_features(dummy).shape[1]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_classes)

    def _extract_features(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        x = self._extract_features(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


# ============================================================
# DATA HELPERS
# ============================================================
def make_arrays(items):
    X = np.stack([r["signal"] for r in items]).astype(np.float32)
    y = np.array(
        [class_to_index[r["label"]] for r in items],
        dtype=np.int64,
    )
    return X, y


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


def predict_cfo(train_records, test_records):
    """Nearest-centroid baseline using supplied CFO metadata."""
    x_train = np.array([r["cfo"] for r in train_records])
    y_train = np.array([
        class_to_index[r["label"]] for r in train_records
    ])
    x_test = np.array([r["cfo"] for r in test_records])

    mean = x_train.mean()
    std = x_train.std()
    if std < 1e-12:
        std = 1.0

    A = (x_train - mean) / std
    B = (x_test - mean) / std

    centroids = np.array([
        A[y_train == c].mean() for c in range(num_classes)
    ])

    distances = (B[:, None] - centroids[None, :]) ** 2
    return distances.argmin(axis=1)


# ============================================================
# FOUR-FOLD SESSION-LEVEL EVALUATION
# Keep test sessions out of both training and validation.
# ============================================================
pooled_actual = []
pooled_cnn_predictions = []
pooled_cfo_predictions = []
prediction_rows = []
fold_summaries = []

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

    if {r["label"] for r in train_records} != set(classes):
        raise ValueError(f"Fold {fold + 1}: training classes incomplete")

    assert not (
        {r["session"] for r in train_records} & excluded
    )

    print(f"\n{'=' * 64}")
    print(f"Fold {fold + 1}/{len(SESSIONS)}")
    print("Validation session:", val_session)
    print("Test session:", test_session)
    print(
        f"Recordings: train={len(train_records)}, "
        f"validation={len(val_records)}, test={len(test_records)}"
    )

    # CFO baseline on exactly the same held-out recordings.
    cfo_predictions = predict_cfo(train_records, test_records)

    X_train, y_train = make_arrays(train_records)
    X_val, y_val = make_arrays(val_records)
    X_test, y_test = make_arrays(test_records)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)

    torch.manual_seed(SEED + fold)
    model = IQCNN(X_train.shape[2], num_classes).to(device)

    # Paper-aligned optimizer and learning rate.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    # Train up to the paper's reported eight epochs.
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y)

        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        val_actual = []
        val_predicted = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)

                total_val_loss += loss.item() * len(y)
                val_actual.extend(y.cpu().tolist())
                val_predicted.extend(
                    logits.argmax(dim=1).cpu().tolist()
                )

        val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_actual, val_predicted)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val accuracy={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No checkpoint selected.")

    model.load_state_dict(best_state)
    model.eval()

    # One complete beacon = one test prediction.
    with torch.no_grad():
        logits = []
        for start in range(0, len(X_test), BATCH_SIZE):
            batch = torch.from_numpy(
                X_test[start:start + BATCH_SIZE]
            ).to(device)

            logits.append(model(batch).cpu().numpy())

    cnn_predictions = np.concatenate(logits, axis=0).argmax(axis=1)

    cnn_accuracy = accuracy_score(y_test, cnn_predictions)
    cfo_accuracy = accuracy_score(y_test, cfo_predictions)

    print("Selected validation epoch:", best_epoch)
    print(f"CFO metadata accuracy: {cnn_accuracy * 0 + cfo_accuracy:.4f}")
    print(f"IQCNN recording accuracy: {cnn_accuracy:.4f}")

    # Save a fold-specific checkpoint for reproducibility.
    checkpoint_path = MODEL_DIR / f"paper_iqcnn_fold_{fold + 1}.pt"
    torch.save(best_state, checkpoint_path)

    pooled_actual.extend(y_test.tolist())
    pooled_cnn_predictions.extend(cnn_predictions.tolist())
    pooled_cfo_predictions.extend(cfo_predictions.tolist())

    fold_summaries.append({
        "fold": fold + 1,
        "validation_session": val_session,
        "test_session": test_session,
        "train_recordings": len(train_records),
        "validation_recordings": len(val_records),
        "test_recordings": len(test_records),
        "best_epoch": best_epoch,
        "cfo_accuracy": cfo_accuracy,
        "iqcnn_accuracy": cnn_accuracy,
    })

    for record, actual, cfo_pred, cnn_pred in zip(
        test_records, y_test, cfo_predictions, cnn_predictions
    ):
        prediction_rows.append({
            "filename": record["filename"],
            "session": test_session,
            "actual": classes[actual],
            "cfo_prediction": classes[cfo_pred],
            "iqcnn_prediction": classes[cnn_pred],
        })


# ============================================================
# POOLED RECORDING-LEVEL RESULTS
# ============================================================
actual = np.asarray(pooled_actual)
cnn_pred = np.asarray(pooled_cnn_predictions)
cfo_pred = np.asarray(pooled_cfo_predictions)

print("\n" + "=" * 64)
print("POOLED SESSION-LEVEL RESULTS — RECORDING LEVEL")

for name, predictions in [
    ("CFO metadata baseline", cfo_pred),
    ("Paper-aligned IQCNN", cnn_pred),
]:
    accuracy = accuracy_score(actual, predictions)
    balanced = balanced_accuracy_score(actual, predictions)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced * 100:.2f}%")
    print(classification_report(
        actual,
        predictions,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    ))

    cm = confusion_matrix(
        actual,
        predictions,
        labels=list(range(num_classes)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )
    display.plot(xticks_rotation=45, values_format="d")
    plt.title(f"{name} — Session-Level Evaluation")
    plt.tight_layout()

    filename = (
        "paper_iqcnn_confusion_matrix.png"
        if name == "Paper-aligned IQCNN"
        else "paper_iqcnn_cfo_baseline_confusion.png"
    )

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

with open(
    RESULTS_DIR / "paper_iqcnn_fold_results.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(fold_summaries[0].keys()),
    )
    writer.writeheader()
    writer.writerows(fold_summaries)

with open(
    RESULTS_DIR / "paper_iqcnn_predictions.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "session",
            "actual",
            "cfo_prediction",
            "iqcnn_prediction",
        ],
    )
    writer.writeheader()
    writer.writerows(prediction_rows)

print("\nSaved fold results:", RESULTS_DIR / "paper_iqcnn_fold_results.csv")
print("\nSaved predictions:", RESULTS_DIR / "paper_iqcnn_predictions.csv")
print("Saved confusion matrices in:", RESULTS_DIR)
print("\nPaper-aligned IQCNN evaluation completed.")
