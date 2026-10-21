
from pathlib import Path
from collections import Counter
import json

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------- Configuration ----------------
DATA_DIR = Path("data/anechoic_chamber")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LENGTH = 2048
RANDOM_SEED = 42

# ---------------- Find recordings ----------------
files = sorted(DATA_DIR.rglob("*.npz"))

if not files:
    raise FileNotFoundError(
        f"No NPZ recordings found in {DATA_DIR.resolve()}"
    )

# Read labels from each recording's BSSID metadata.
recordings = []
failed = []

for path in files:
    try:
        with np.load(path, allow_pickle=False) as data:
            bssid = str(data["BSSID"].item()).strip().upper()

            i = data["beaconDataDownSampledI"].astype(np.float32)
            q = data["beaconDataDownSampledQ"].astype(np.float32)

            if len(i) != len(q) or len(i) < SEGMENT_LENGTH:
                raise ValueError("Invalid or too-short I/Q arrays")

            if not (np.isfinite(i).all() and np.isfinite(q).all()):
                raise ValueError("Signal contains non-finite values")

            recordings.append({
                "path": path,
                "label": bssid,
            })

    except Exception as exc:
        failed.append((path.name, str(exc)))

if failed:
    print(f"WARNING: {len(failed)} files could not be read.")
    for name, error in failed[:5]:
        print(f"  {name}: {error}")

if not recordings:
    raise RuntimeError("No valid recordings found.")

# ---------------- Encode class labels ----------------
classes = sorted({r["label"] for r in recordings})
label_map = {bssid: index for index, bssid in enumerate(classes)}

print(f"\nValid recordings: {len(recordings)}")
print(f"Candidate classes: {len(classes)}")

for bssid in classes:
    count = sum(r["label"] == bssid for r in recordings)
    print(f"  Class {label_map[bssid]} | {bssid}: {count} recordings")

# ---------------- Split by recording FIRST ----------------
# Segments from a single recording must never cross splits.
labels = [r["label"] for r in recordings]

train_records, remaining_records = train_test_split(
    recordings,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=labels,
)

remaining_labels = [r["label"] for r in remaining_records]

val_records, test_records = train_test_split(
    remaining_records,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=remaining_labels,
)

print("\nRecording-level split:")
print(f"  Train: {len(train_records)} recordings")
print(f"  Validation: {len(val_records)} recordings")
print(f"  Test: {len(test_records)} recordings")

# ---------------- Convert recordings into segments ----------------
def make_segments(record_list):
    X, y = [], []

    for record in record_list:
        with np.load(record["path"], allow_pickle=False) as data:
            i = data["beaconDataDownSampledI"].astype(np.float32)
            q = data["beaconDataDownSampledQ"].astype(np.float32)

        # Non-overlapping segments: no segment crosses a recording boundary.
        total = len(i) // SEGMENT_LENGTH

        for index in range(total):
            start = index * SEGMENT_LENGTH
            end = start + SEGMENT_LENGTH

            segment_i = i[start:end]
            segment_q = q[start:end]

            # Two input channels: I and Q.
            segment = np.stack([segment_i, segment_q], axis=0)

            # Remove near-zero segments if encountered.
            rms = np.sqrt(np.mean(segment ** 2))
            if rms < 1e-8:
                continue

            # Initial baseline: normalize each segment by its RMS.
            # This reduces gain differences; it may also remove useful
            # amplitude information, so we will compare this choice later.
            segment = segment / rms

            X.append(segment.astype(np.float32))
            y.append(label_map[record["label"]])

    if not X:
        raise RuntimeError("No usable signal segments were produced.")

    return np.stack(X), np.asarray(y, dtype=np.int64)


X_train, y_train = make_segments(train_records)
X_val, y_val = make_segments(val_records)
X_test, y_test = make_segments(test_records)

# ---------------- Save processed data ----------------
output_path = OUTPUT_DIR / "rf_dataset.npz"

np.savez_compressed(
    output_path,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
)

with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2)

# ---------------- Report ----------------
print("\nSegment arrays:")
for name, X, y in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    print(f"\n{name}: X={X.shape}, y={y.shape}")
    print("  Segments per class:", dict(sorted(Counter(y.tolist()).items())))

print("\nSaved processed dataset:", output_path.resolve())
print("Saved label map:", (OUTPUT_DIR / "label_map.json").resolve())
print("\nDataset preparation completed successfully.")
