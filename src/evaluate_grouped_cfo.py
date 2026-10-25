
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"

TRAIN_SESSIONS = {
    "2025_12_08_12_09_21",
    "2025_12_08_12_24_58",
    "2025_12_08_12_26_46",
    "2025_12_09_15_58_20",
}
VAL_SESSIONS = {"2025_12_09_16_02_52"}
TEST_SESSIONS = {"2025_12_19_08_48_36"}

records = []

for path in sorted(DATA_DIR.glob("*.npz")):
    stem = path.stem.removesuffix("_anechoic_chamber")
    parts = stem.split("_")

    if len(parts) < 8 or not parts[-1].isdigit():
        print("Skipping unexpected filename:", path.name)
        continue

    session = "_".join(parts[1:-1])

    with np.load(path, allow_pickle=False) as data:
        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())

    if not np.isfinite(cfo):
        continue

    if session in TRAIN_SESSIONS:
        split = "train"
    elif session in VAL_SESSIONS:
        split = "validation"
    elif session in TEST_SESSIONS:
        split = "test"
    else:
        raise ValueError(f"Unassigned session: {session}")

    records.append({
        "label": label,
        "cfo": cfo,
        "session": session,
        "split": split,
    })

train = [r for r in records if r["split"] == "train"]
test = [r for r in records if r["split"] == "test"]

classes = np.array(sorted({r["label"] for r in records}))

print("All recordings:", len(records))
print("Training recordings:", len(train))
print("Test recordings:", len(test))
print("Candidate BSSIDs:", len(classes))

assert len(train) > 0 and len(test) > 0
assert {r["session"] for r in train}.isdisjoint(
    {r["session"] for r in test}
)

y_train = np.array([r["label"] for r in train])
y_test = np.array([r["label"] for r in test])

X_train = np.array([r["cfo"] for r in train]).reshape(-1, 1)
X_test = np.array([r["cfo"] for r in test]).reshape(-1, 1)

if set(y_train) != set(classes) or set(y_test) != set(classes):
    raise ValueError("Training and test sets must each contain all classes.")

# Fit scaling on training recordings only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0

A = (X_train - mean) / std
B = (X_test - mean) / std

# Train a nearest-centroid classifier using CFO alone.
centroids = np.vstack([
    A[y_train == label].mean(axis=0)
    for label in classes
])

distances = ((B[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
predicted = classes[distances.argmin(axis=1)]

print("\n=== CFO-ONLY, SESSION-SEPARATED RESULTS ===")
print(f"Recording-level accuracy: {accuracy_score(y_test, predicted) * 100:.2f}%")
print(
    "Balanced accuracy: "
    f"{balanced_accuracy_score(y_test, predicted) * 100:.2f}%"
)

print("\nClassification report:")
print(classification_report(
    y_test,
    predicted,
    labels=classes,
    target_names=classes,
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predicted, labels=classes))
