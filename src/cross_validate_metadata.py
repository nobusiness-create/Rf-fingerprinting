
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"

SESSIONS = [
    "2025_12_08_12_09_21",
    "2025_12_09_15_58_20",
    "2025_12_09_16_02_52",
    "2025_12_19_08_48_36",
]

records = []

for path in sorted(DATA_DIR.glob("*.npz")):
    stem = path.stem.removesuffix("_anechoic_chamber")
    parts = stem.split("_")

    if len(parts) < 8 or not parts[-1].isdigit():
        continue

    session = "_".join(parts[1:-1])

    with np.load(path, allow_pickle=False) as data:
        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())
        snr = float(np.asarray(data["SNR"]).squeeze())

    records.append({
        "label": label,
        "session": session,
        "cfo": cfo,
        "snr": snr,
    })

classes = np.array(sorted({r["label"] for r in records}))
class_index = {name: i for i, name in enumerate(classes)}

feature_sets = {
    "CFO only": ["cfo"],
    "SNR only": ["snr"],
    "CFO + SNR": ["cfo", "snr"],
}

pooled = {
    name: {"actual": [], "predicted": []}
    for name in feature_sets
}

print("Recordings:", len(records))
print("Classes:", len(classes))
print("\n=== SESSION-LEVEL CROSS-VALIDATION ===")

for session in SESSIONS:
    train = [r for r in records if r["session"] != session]
    test = [r for r in records if r["session"] == session]

    train_classes = {r["label"] for r in train}
    test_classes = {r["label"] for r in test}

    if train_classes != set(classes) or test_classes != set(classes):
        print("Skipping session with missing classes:", session)
        continue

    y_train = np.array([class_index[r["label"]] for r in train])
    y_test = np.array([class_index[r["label"]] for r in test])

    print(f"\nTest session: {session} ({len(test)} recordings)")

    for name, columns in feature_sets.items():
        X_train = np.array([
            [r[column] for column in columns]
            for r in train
        ], dtype=float)

        X_test = np.array([
            [r[column] for column in columns]
            for r in test
        ], dtype=float)

        # Fit standardization on training recordings only.
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd[sd < 1e-12] = 1.0

        A = (X_train - mu) / sd
        B = (X_test - mu) / sd

        # Nearest-centroid classifier.
        centroids = np.vstack([
            A[y_train == c].mean(axis=0)
            for c in range(len(classes))
        ])

        distances = (
            (B[:, None, :] - centroids[None, :, :]) ** 2
        ).sum(axis=2)

        predictions = distances.argmin(axis=1)

        pooled[name]["actual"].extend(y_test.tolist())
        pooled[name]["predicted"].extend(predictions.tolist())

        acc = accuracy_score(y_test, predictions) * 100
        bal = balanced_accuracy_score(y_test, predictions) * 100

        print(f"  {name:12s}: accuracy={acc:6.2f}%  balanced={bal:6.2f}%")

print("\n=== POOLED OUT-OF-FOLD RESULTS ===")

for name, result in pooled.items():
    if not result["actual"]:
        continue

    acc = accuracy_score(
        result["actual"], result["predicted"]
    ) * 100

    bal = balanced_accuracy_score(
        result["actual"], result["predicted"]
    ) * 100

    print(
        f"{name:12s}: accuracy={acc:6.2f}%  "
        f"balanced accuracy={bal:6.2f}%"
    )
