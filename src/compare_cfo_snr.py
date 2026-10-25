
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"

records = []

for path in sorted(DATA_DIR.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())
        snr = float(np.asarray(data["SNR"]).squeeze())

        if np.isfinite(cfo) and np.isfinite(snr):
            records.append((label, cfo, snr))

labels = np.array([r[0] for r in records])
X = np.array([[r[1], r[2]] for r in records], dtype=float)

classes = np.unique(labels)
indices = np.arange(len(labels))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=labels,
)

feature_sets = {
    "CFO only": [0],
    "SNR only": [1],
    "CFO + SNR": [0, 1],
}

print(f"Recordings: {len(records)}")
print(f"Candidate BSSIDs: {len(classes)}")
print("\nExploratory nearest-centroid results:")

for name, cols in feature_sets.items():
    A = X[train_idx][:, cols]
    B = X[test_idx][:, cols]

    # Fit standardization on training data only.
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    std[std == 0] = 1

    A = (A - mean) / std
    B = (B - mean) / std

    # One centroid per BSSID.
    centers = np.vstack([
        A[labels[train_idx] == c].mean(axis=0)
        for c in classes
    ])

    distances = ((B[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    predicted = classes[distances.argmin(axis=1)]
    actual = labels[test_idx]

    acc = accuracy_score(actual, predicted) * 100
    bal_acc = balanced_accuracy_score(actual, predicted) * 100

    print(
        f"{name:12s} | Accuracy: {acc:6.2f}%"
        f" | Balanced accuracy: {bal_acc:6.2f}%"
    )
