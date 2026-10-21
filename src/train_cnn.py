
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

DATA_PATH = Path("data/processed/rf_dataset.npz")
LABEL_PATH = Path("data/processed/label_map.json")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# ---------------- Load dataset ----------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}. "
        "Run src/prepare_dataset.py first."
    )

with np.load(DATA_PATH) as data:
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

with open(LABEL_PATH, encoding="utf-8") as f:
    label_map = json.load(f)

class_names = [
    bssid for bssid, index in
    sorted(label_map.items(), key=lambda item: item[1])
]
num_classes = len(class_names)

print("Classes:", class_names)
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# ---------------- PyTorch data loaders ----------------
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

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

# ---------------- Define 1D CNN ----------------
class RFClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = RFClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# ---------------- Training and validation ----------------
best_val_accuracy = -1.0
history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

best_model_path = MODEL_DIR / "rf_cnn_best.pt"

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_true, train_pred = [], []

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(y)
        train_true.extend(y.cpu().tolist())
        train_pred.extend(logits.argmax(dim=1).cpu().tolist())

    train_loss = train_loss_sum / len(train_loader.dataset)
    train_acc = accuracy_score(train_true, train_pred)

    model.eval()
    val_loss_sum = 0.0
    val_true, val_pred = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            val_loss_sum += loss.item() * len(y)
            val_true.extend(y.cpu().tolist())
            val_pred.extend(logits.argmax(dim=1).cpu().tolist())

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_acc = accuracy_score(val_true, val_pred)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"train loss={train_loss:.4f}, "
        f"train acc={train_acc:.4f} | "
        f"val loss={val_loss:.4f}, "
        f"val acc={val_acc:.4f}"
    )

print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
print("Best model saved:", best_model_path.resolve())

# ---------------- Evaluate held-out test set ----------------
# Load the checkpoint selected using validation accuracy.
model.load_state_dict(
    torch.load(best_model_path, map_location=device, weights_only=True)
)
model.eval()

test_true, test_pred = [], []

with torch.no_grad():
    for X, y in test_loader:
        logits = model(X.to(device))
        test_true.extend(y.tolist())
        test_pred.extend(logits.argmax(dim=1).cpu().tolist())

test_accuracy = accuracy_score(test_true, test_pred)

print(f"\nTest segment accuracy: {test_accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        test_true,
        test_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
)

cm = confusion_matrix(
    test_true,
    test_pred,
    labels=list(range(num_classes)),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
display.plot(xticks_rotation=45, values_format="d")
plt.title("RF Fingerprinting: Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "cnn_confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# ---------------- Plot training history ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["train_accuracy"], label="Train")
axes[1].plot(history["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "cnn_training_history.png", dpi=150)
plt.show()

print("\nTraining history saved:", (RESULTS_DIR / "cnn_training_history.png").resolve())
print("Confusion matrix saved:", (RESULTS_DIR / "cnn_confusion_matrix.png").resolve())
