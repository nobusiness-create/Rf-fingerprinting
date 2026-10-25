
from pathlib import Path
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"

records = []

for path in sorted(DATA_DIR.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())

    if not np.isfinite(cfo):
        continue

    # Expected pattern:
    # 001_2025_12_08_12_09_21_0_anechoic_chamber
    # Remove the final recording index to define a capture group.
    stem = path.stem.removesuffix("_anechoic_chamber")
    parts = stem.split("_")

    if len(parts) < 8 or not parts[-1].isdigit():
        print("Check filename pattern:", path.name)
        continue

    # Group all transmitter recordings captured at the same timestamp.
# Exclude the transmitter/file prefix and final recording index.
    session_id = "_".join(parts[1:-1])
    records.append((label, cfo, session_id))

if not records:
    raise RuntimeError("No valid recordings found.")

labels = np.array([r[0] for r in records])
X = np.array([[r[1]] for r in records], dtype=float)
groups = np.array([r[2] for r in records])

classes = np.unique(labels)

print("Recordings:", len(records))
print("Candidate BSSIDs:", len(classes))
print("Capture groups:", len(np.unique(groups)))
print("\nGroups per BSSID:")

for c in classes:
    print(f"  {c}: {len(np.unique(groups[labels == c]))}")

# Verify whether CFO is constant within each inferred session.
print("\nUnique CFO values within sessions:")
for group in np.unique(groups):
    idx = groups == group
    print(
        f"  {group}: {idx.sum()} recordings, "
        f"{len(np.unique(X[idx, 0]))} CFO value(s)"
    )

# Leave one entire capture group out at a time.
logo = LeaveOneGroupOut()
actual_all = []
predicted_all = []

for train_idx, test_idx in logo.split(X, labels, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = labels[train_idx]

    # Fit scaling on training data only.
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1

    A = (X_train - mu) / sd
    B = (X_test - mu) / sd

    centers = np.vstack([
        A[y_train == c].mean(axis=0)
        for c in classes
    ])

    distances = ((B[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    predictions = classes[distances.argmin(axis=1)]

    actual_all.extend(labels[test_idx])
    predicted_all.extend(predictions)

print("\n=== SESSION-HOLDOUT CFO RESULTS ===")
print(
    f"Accuracy: "
    f"{accuracy_score(actual_all, predicted_all) * 100:.2f}%"
)
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(actual_all, predicted_all) * 100:.2f}%"
)
