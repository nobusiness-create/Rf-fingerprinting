
from pathlib import Path
import json
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
MAX_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001
PATIENCE = 7

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "rf_dataset_grouped.npz"
LABEL_PATH = ROOT / "data" / "processed" / "label_map.json"

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "rf_hybrid_cnn_best.pt"
HISTORY_PATH = RESULTS_DIR / "hybrid_cnn_history.png"
CM_PATH = RESULTS_DIR / "hybrid_cnn_confusion_matrix.png"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# ============================================================
# LOAD GROUPED DATASET
# ============================================================
if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run src/prepare_dataset.py first."
    )

with np.load(DATA_PATH) as data:
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    cfo_train = data["cfo_train"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)
    cfo_val = data["cfo_val"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    cfo_test = data["cfo_test"].astype(np.float32)

with open(LABEL_PATH, encoding="utf-8") as f:
    label_map = json.load(f)

class_names = [
    label for label, index in
    sorted(label_map.items(), key=lambda item: item[1])
]
num_classes = len(class_names)

print("Classes:", class_names)

for name, X, y, cfo in [
    ("Train", X_train, y_train, cfo_train),
    ("Validation", X_val, y_val, cfo_val),
    ("Test", X_test, y_test, cfo_test),
]:
    print(
        f"{name}: X={X.shape}, y={y.shape}, CFO={cfo.shape}"
    )

    if len(X) != len(y) or len(y) != len(cfo):
        raise ValueError(f"{name}: signal, label, and CFO lengths differ")

    if not np.isfinite(X).all() or not np.isfinite(cfo).all():
        raise ValueError(f"{name}: non-finite input values detected")

    if set(np.unique(y)) != set(range(num_classes)):
        raise ValueError(f"{name}: one or more classes are missing")

# ============================================================
# STANDARDIZE CFO USING TRAINING DATA ONLY
# ============================================================
cfo_mean = float(cfo_train.mean())
cfo_std = float(cfo_train.std())

if cfo_std < 1e-8:
    cfo_std = 1.0

cfo_train_z = ((cfo_train - cfo_mean) / cfo_std).reshape(-1, 1)
cfo_val_z = ((cfo_val - cfo_mean) / cfo_std).reshape(-1, 1)
cfo_test_z = ((cfo_test - cfo_mean) / cfo_std).reshape(-1, 1)

print(f"\nCFO training mean: {cfo_mean:.4f}")
print(f"CFO training standard deviation: {cfo_std:.4f}")

# ============================================================
# DATA LOADERS
# ============================================================
def make_loader(X, cfo, y, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(cfo.astype(np.float32)),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = make_loader(
    X_train, cfo_train_z, y_train, shuffle=True
)
val_loader = make_loader(X_val, cfo_val_z, y_val)
test_loader = make_loader(X_test, cfo_test_z, y_test)

# ============================================================
# CFO-ONLY BASELINE ON THE SAME GROUPED SEGMENT SPLIT
# ============================================================
centroids = np.vstack([
    cfo_train_z[y_train == c].mean(axis=0)
    for c in range(num_classes)
])

cfo_distances = (
    (cfo_test_z[:, None, :] - centroids[None, :, :]) ** 2
).sum(axis=2)

cfo_predictions = cfo_distances.argmin(axis=1)

print("\n========== CFO-ONLY BASELINE ==========")
print(
    f"Test segment accuracy: "
    f"{accuracy_score(y_test, cfo_predictions):.4f}"
)
print(
    f"Test balanced accuracy: "
    f"{balanced_accuracy_score(y_test, cfo_predictions):.4f}"
)

# ============================================================
# HYBRID MODEL: 1D CNN FOR I/Q + SMALL MLP FOR CFO
# ============================================================
class HybridRFClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.signal_branch = nn.Sequential(
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

        self.cfo_branch = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 8, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, signal, cfo):
        signal_features = self.signal_branch(signal)
        cfo_features = self.cfo_branch(cfo)

        combined = torch.cat(
            [signal_features, cfo_features],
            dim=1,
        )

        return self.classifier(combined)


model = HybridRFClassifier(num_classes).to(device)

# Class-weighted loss helps account for unequal training-segment counts.
class_counts = torch.bincount(
    torch.from_numpy(y_train),
    minlength=num_classes,
).float()

class_weights = class_counts.sum() / (
    num_classes * class_counts.clamp(min=1)
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),
    label_smoothing=0.02,
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

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate(loader):
    model.eval()

    loss_sum = 0.0
    actual = []
    predictions = []

    with torch.no_grad():
        for signal, cfo, labels in loader:
            signal = signal.to(device)
            cfo = cfo.to(device)
            labels = labels.to(device)

            logits = model(signal, cfo)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * len(labels)

            actual.extend(labels.cpu().numpy().tolist())
            predictions.extend(
                logits.argmax(dim=1).cpu().numpy().tolist()
            )

    avg_loss = loss_sum / len(loader.dataset)
    acc = accuracy_score(actual, predictions)

    return avg_loss, acc, actual, predictions

# ============================================================
# TRAIN WITH EARLY STOPPING
# ============================================================
history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

print("\n========== HYBRID CNN TRAINING ==========")

for epoch in range(MAX_EPOCHS):
    model.train()

    loss_sum = 0.0
    actual_train = []
    predictions_train = []

    for signal, cfo, labels in train_loader:
        signal = signal.to(device)
        cfo = cfo.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(signal, cfo)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item() * len(labels)

        actual_train.extend(labels.cpu().numpy().tolist())
        predictions_train.extend(
            logits.argmax(dim=1).detach().cpu().numpy().tolist()
        )

    train_loss = loss_sum / len(train_loader.dataset)
    train_acc = accuracy_score(actual_train, predictions_train)

    val_loss, val_acc, _, _ = evaluate(val_loader)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1:02d}/{MAX_EPOCHS} | "
        f"train loss={train_loss:.4f}, "
        f"train acc={train_acc:.4f} | "
        f"val loss={val_loss:.4f}, "
        f"val acc={val_acc:.4f} | "
        f"lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    # Select the best checkpoint using validation loss only.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch + 1}; "
            f"best epoch was {best_epoch}."
        )
        break

print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Best checkpoint epoch: {best_epoch}")
print("Saved model:", MODEL_PATH)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

_, _, test_actual, test_predictions = evaluate(test_loader)

test_acc = accuracy_score(test_actual, test_predictions)
test_balanced_acc = balanced_accuracy_score(
    test_actual, test_predictions
)

print("\n========== HYBRID CNN TEST RESULTS ==========")
print(f"Test segment accuracy: {test_acc:.4f}")
print(f"Test balanced accuracy: {test_balanced_acc:.4f}")

print("\nClassification report:")
print(classification_report(
    test_actual,
    test_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0,
))

cm = confusion_matrix(
    test_actual,
    test_predictions,
    labels=list(range(num_classes)),
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
disp.plot(xticks_rotation=45, values_format="d")

plt.title("Hybrid I/Q + CFO CNN: Grouped Test Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# SAVE TRAINING CURVES
# ============================================================
epochs = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, history["train_loss"], label="Train")
axes[0].plot(epochs, history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    epochs, history["train_accuracy"], label="Train"
)
axes[1].plot(
    epochs, history["val_accuracy"], label="Validation"
)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(HISTORY_PATH, dpi=150, bbox_inches="tight")
plt.show()

print("\nTraining history saved:", HISTORY_PATH)
print("Confusion matrix saved:", CM_PATH)
print("\nHybrid CNN experiment completed.")
