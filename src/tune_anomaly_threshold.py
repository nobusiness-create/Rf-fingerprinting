
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"

SESSIONS = [
    "2025_12_08_12_09_21",
    "2025_12_09_15_58_20",
    "2025_12_09_16_02_52",
    "2025_12_19_08_48_36",
]

# Explore a wider range, including more permissive thresholds.
QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

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

    if np.isfinite(cfo):
        records.append({
            "label": label,
            "session": session,
            "cfo": cfo,
        })

classes = sorted({r["label"] for r in records})
results = {q: [] for q in QUANTILES}

for fold, test_session in enumerate(SESSIONS):
    val_session = SESSIONS[(fold + 1) % len(SESSIONS)]

    for unknown_label in classes:
        train = [
            r for r in records
            if r["session"] not in {test_session, val_session}
            and r["label"] != unknown_label
        ]
        val = [
            r for r in records
            if r["session"] == val_session
            and r["label"] != unknown_label
        ]
        test = [
            r for r in records
            if r["session"] == test_session
        ]

        known_classes = sorted({r["label"] for r in train})

        if set(known_classes) != set(classes) - {unknown_label}:
            raise ValueError("Training classes are incomplete.")

        if {r["label"] for r in val} != set(known_classes):
            raise ValueError("Validation classes are incomplete.")

        X_train = np.array([r["cfo"] for r in train])
        X_val = np.array([r["cfo"] for r in val])
        X_test = np.array([r["cfo"] for r in test])

        mean = X_train.mean()
        std = X_train.std()
        if std < 1e-12:
            std = 1.0

        A = (X_train - mean) / std
        V = (X_val - mean) / std
        T = (X_test - mean) / std

        train_labels = np.array([r["label"] for r in train])
        val_labels = np.array([r["label"] for r in val])
        y_unknown = np.array([
            int(r["label"] == unknown_label) for r in test
        ])

        centers = np.array([
            A[train_labels == label].mean()
            for label in known_classes
        ])

        val_distances = np.abs(V[:, None] - centers[None, :])

        # Distance from each known validation sample to its
        # correctly labelled class centroid.
        class_distances = {
            label: val_distances[
                val_labels == label, class_idx
            ]
            for class_idx, label in enumerate(known_classes)
        }

        test_distances = np.abs(T[:, None] - centers[None, :])
        nearest_idx = test_distances.argmin(axis=1)
        nearest_distance = test_distances.min(axis=1)

        for q in QUANTILES:
            thresholds = {
                label: np.quantile(values, q)
                for label, values in class_distances.items()
            }

            predicted_unknown = np.array([
                int(
                    nearest_distance[i]
                    > thresholds[known_classes[nearest_idx[i]]]
                )
                for i in range(len(test))
            ])

            known_mask = y_unknown == 0
            unknown_mask = y_unknown == 1

            known_acceptance = np.mean(
                predicted_unknown[known_mask] == 0
            )
            unknown_detection = np.mean(
                predicted_unknown[unknown_mask] == 1
            )
            balanced_acc = balanced_accuracy_score(
                y_unknown, predicted_unknown
            )

            results[q].append((
                known_acceptance,
                unknown_detection,
                balanced_acc,
            ))

print("\nCFO-only anomaly threshold sensitivity")
print("Each threshold is calibrated on known validation recordings.")
print("\nPercentile | Known accepted | Unknown detected | Balanced accuracy")

for q, values in results.items():
    known = np.mean([v[0] for v in values]) * 100
    unknown = np.mean([v[1] for v in values]) * 100
    balanced = np.mean([v[2] for v in values]) * 100

    print(
        f"{q * 100:9.1f}% | {known:13.2f}% | "
        f"{unknown:15.2f}% | {balanced:18.2f}%"
    )

print("\nThreshold sweep completed.")
print("Treat these results as exploratory, not final validation.")
