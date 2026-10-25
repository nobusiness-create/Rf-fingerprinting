
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

FEATURE_SETS = {
    "CFO only": ["cfo"],
    "CFO + SNR": ["cfo", "snr"],
}

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

    if not np.isfinite(cfo) or not np.isfinite(snr):
        continue

    records.append({
        "label": label,
        "session": session,
        "cfo": cfo,
        "snr": snr,
    })

classes = sorted({r["label"] for r in records})

results = {name: [] for name in FEATURE_SETS}

print("Recordings:", len(records))
print("Candidate BSSIDs:", len(classes))
print("Threshold calibration: per known class, 95th percentile")

for test_idx, test_session in enumerate(SESSIONS):
    val_session = SESSIONS[(test_idx + 1) % len(SESSIONS)]

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
            raise ValueError("Unexpected training classes.")

        if {r["label"] for r in val} != set(known_classes):
            raise ValueError("Validation set missing known classes.")

        y_binary = np.array([
            int(r["label"] == unknown_label) for r in test
        ])

        for method, columns in FEATURE_SETS.items():

            def matrix(items):
                return np.array([
                    [r[col] for col in columns] for r in items
                ], dtype=float)

            X_train = matrix(train)
            X_val = matrix(val)
            X_test = matrix(test)

            # Fit scaling on training data only.
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std[std < 1e-12] = 1.0

            A = (X_train - mean) / std
            V = (X_val - mean) / std
            T = (X_test - mean) / std

            train_labels = np.array([r["label"] for r in train])
            val_labels = np.array([r["label"] for r in val])

            # One centroid per known class.
            centroids = np.vstack([
                A[train_labels == label].mean(axis=0)
                for label in known_classes
            ])

            # Validation distance to the correct class centroid.
            val_distances = np.sqrt(
                ((V[:, None, :] - centroids[None, :, :]) ** 2)
                .sum(axis=2)
            )

            thresholds = {}

            for class_idx, label in enumerate(known_classes):
                class_mask = val_labels == label
                correct_class_distances = val_distances[
                    class_mask, class_idx
                ]

                thresholds[label] = np.quantile(
                    correct_class_distances, 0.95
                )

            # Test: find the nearest known class first.
            test_distances = np.sqrt(
                ((T[:, None, :] - centroids[None, :, :]) ** 2)
                .sum(axis=2)
            )

            nearest_indices = test_distances.argmin(axis=1)
            nearest_distances = test_distances.min(axis=1)

            predicted_unknown = np.array([
                nearest_distances[i] >
                thresholds[known_classes[nearest_indices[i]]]
                for i in range(len(test))
            ], dtype=int)

            known_mask = y_binary == 0
            unknown_mask = y_binary == 1

            known_acceptance = np.mean(
                predicted_unknown[known_mask] == 0
            )
            unknown_detection = np.mean(
                predicted_unknown[unknown_mask] == 1
            )

            balanced_acc = balanced_accuracy_score(
                y_binary, predicted_unknown
            )

            results[method].append({
                "known_acceptance": known_acceptance,
                "unknown_detection": unknown_detection,
                "balanced_accuracy": balanced_acc,
            })

print("\n========== PER-CLASS THRESHOLD RESULTS ==========")

for method, rows in results.items():
    print(f"\n{method}")
    print(
        "Known acceptance rate: "
        f"{np.mean([r['known_acceptance'] for r in rows]) * 100:.2f}%"
    )
    print(
        "Unknown detection rate: "
        f"{np.mean([r['unknown_detection'] for r in rows]) * 100:.2f}%"
    )
    print(
        "Balanced accuracy: "
        f"{np.mean([r['balanced_accuracy'] for r in rows]) * 100:.2f}%"
    )

print("\nCompleted. Results are exploratory, not a final validation.")
