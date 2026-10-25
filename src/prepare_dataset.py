
from pathlib import Path
from collections import Counter
import json

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LENGTH = 2048

# ============================================================
# SESSION-BASED SPLIT
# Recordings from one timestamp stay in a single split.
# ============================================================
TRAIN_SESSIONS = {
    "2025_12_08_12_09_21",
    "2025_12_08_12_24_58",
    "2025_12_08_12_26_46",
    "2025_12_09_15_58_20",
}

VAL_SESSIONS = {
    "2025_12_09_16_02_52",
}

TEST_SESSIONS = {
    "2025_12_19_08_48_36",
}

assert TRAIN_SESSIONS.isdisjoint(VAL_SESSIONS)
assert TRAIN_SESSIONS.isdisjoint(TEST_SESSIONS)
assert VAL_SESSIONS.isdisjoint(TEST_SESSIONS)

# ============================================================
# FIND RECORDINGS AND READ METADATA
# ============================================================
files = sorted(DATA_DIR.glob("*.npz"))

if not files:
    raise FileNotFoundError(
        f"No NPZ recordings found in {DATA_DIR}"
    )

recordings = []
failed = []

for path in files:
    try:
        with np.load(path, allow_pickle=False) as data:
            bssid = str(data["BSSID"].item()).strip().upper()

            cfo = float(np.asarray(data["CFO"]).squeeze())

            i = data["beaconDataDownSampledI"].astype(np.float32)
            q = data["beaconDataDownSampledQ"].astype(np.float32)

            if len(i) != len(q) or len(i) < SEGMENT_LENGTH:
                raise ValueError("Invalid or too-short I/Q arrays")

            if not (np.isfinite(i).all() and np.isfinite(q).all()):
                raise ValueError("Signal contains non-finite values")

            if not np.isfinite(cfo):
                raise ValueError("CFO is not finite")

        # Expected filename:
        # 001_2025_12_08_12_09_21_0_anechoic_chamber.npz
        stem = path.stem.removesuffix("_anechoic_chamber")
        parts = stem.split("_")

        if len(parts) < 8 or not parts[-1].isdigit():
            raise ValueError(
                f"Filename does not match expected format: {path.name}"
            )

        # Exclude device prefix and final recording index.
        session_id = "_".join(parts[1:-1])

        recordings.append({
            "path": path,
            "label": bssid,
            "cfo": cfo,
            "session": session_id,
        })

    except Exception as exc:
        failed.append((path.name, str(exc)))

if failed:
    print(f"WARNING: {len(failed)} recordings could not be read.")
    for name, error in failed[:10]:
        print(f"  {name}: {error}")

if not recordings:
    raise RuntimeError("No valid recordings found.")

# ============================================================
# ENCODE BSSID LABELS
# ============================================================
classes = sorted({r["label"] for r in recordings})
label_map = {bssid: idx for idx, bssid in enumerate(classes)}

print(f"\nValid recordings: {len(recordings)}")
print(f"Candidate classes: {len(classes)}")

for bssid in classes:
    count = sum(r["label"] == bssid for r in recordings)
    print(f"  Class {label_map[bssid]} | {bssid}: {count} recordings")

if len(classes) != 5:
    raise ValueError(f"Expected 5 BSSID classes, found {len(classes)}")

# ============================================================
# SPLIT BY CAPTURE TIMESTAMP BEFORE SEGMENTING
# ============================================================
all_sessions = {r["session"] for r in recordings}
assigned_sessions = TRAIN_SESSIONS | VAL_SESSIONS | TEST_SESSIONS

missing_sessions = all_sessions - assigned_sessions
unused_sessions = assigned_sessions - all_sessions

if missing_sessions or unused_sessions:
    raise ValueError(
        f"Session configuration mismatch.\n"
        f"Unassigned sessions: {sorted(missing_sessions)}\n"
        f"Configured but absent sessions: {sorted(unused_sessions)}"
    )

train_records = [
    r for r in recordings if r["session"] in TRAIN_SESSIONS
]
val_records = [
    r for r in recordings if r["session"] in VAL_SESSIONS
]
test_records = [
    r for r in recordings if r["session"] in TEST_SESSIONS
]

splits = {
    "Train": train_records,
    "Validation": val_records,
    "Test": test_records,
}

for name, split_records in splits.items():
    present_classes = {r["label"] for r in split_records}
    session_names = sorted({r["session"] for r in split_records})

    print(
        f"\n{name}: {len(split_records)} recordings, "
        f"{len(present_classes)} classes, "
        f"{len(session_names)} sessions"
    )
    print(f"  Sessions: {session_names}")

    if present_classes != set(classes):
        raise ValueError(
            f"{name} split does not contain every BSSID class. "
            f"Found: {sorted(present_classes)}"
        )

# Verify recording assignments are disjoint.
train_paths = {r["path"] for r in train_records}
val_paths = {r["path"] for r in val_records}
test_paths = {r["path"] for r in test_records}

assert train_paths.isdisjoint(val_paths)
assert train_paths.isdisjoint(test_paths)
assert val_paths.isdisjoint(test_paths)

assert len(train_paths | val_paths | test_paths) == len(recordings)

# ============================================================
# CONVERT EACH RECORDING INTO NON-OVERLAPPING I/Q SEGMENTS
# Return the segment, class label, and associated CFO.
# ============================================================
def make_segments(record_list):
    X = []
    y = []
    cfo_values = []

    for record in record_list:
        with np.load(record["path"], allow_pickle=False) as data:
            i = data["beaconDataDownSampledI"].astype(np.float32)
            q = data["beaconDataDownSampledQ"].astype(np.float32)

        total_segments = len(i) // SEGMENT_LENGTH

        for segment_idx in range(total_segments):
            start = segment_idx * SEGMENT_LENGTH
            end = start + SEGMENT_LENGTH

            segment_i = i[start:end]
            segment_q = q[start:end]

            # Two input channels: I and Q.
            segment = np.stack([segment_i, segment_q], axis=0)

            # Normalize the I/Q segment by its RMS.
            rms = np.sqrt(np.mean(segment ** 2))

            if not np.isfinite(rms) or rms < 1e-8:
                continue

            segment = segment / rms

            X.append(segment.astype(np.float32))
            y.append(label_map[record["label"]])
            cfo_values.append(record["cfo"])

    if not X:
        raise RuntimeError("No usable signal segments were produced.")

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(cfo_values, dtype=np.float32),
    )

# ============================================================
# BUILD THE THREE DATA SPLITS
# ============================================================
X_train, y_train, cfo_train = make_segments(train_records)
X_val, y_val, cfo_val = make_segments(val_records)
X_test, y_test, cfo_test = make_segments(test_records)

# ============================================================
# SAVE TO A NEW FILE — PRESERVE THE ORIGINAL DATASET
# ============================================================
output_path = OUTPUT_DIR / "rf_dataset_grouped.npz"

np.savez_compressed(
    output_path,
    X_train=X_train,
    y_train=y_train,
    cfo_train=cfo_train,
    X_val=X_val,
    y_val=y_val,
    cfo_val=cfo_val,
    X_test=X_test,
    y_test=y_test,
    cfo_test=cfo_test,
)

label_path = OUTPUT_DIR / "label_map.json"

with open(label_path, "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2)

# ============================================================
# REPORT
# ============================================================
print("\n========== SEGMENT DATASET SUMMARY ==========")

for name, X, y, cfo in [
    ("Train", X_train, y_train, cfo_train),
    ("Validation", X_val, y_val, cfo_val),
    ("Test", X_test, y_test, cfo_test),
]:
    print(f"\n{name}:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  CFO shape: {cfo.shape}")
    print(f"  Segments per class: {dict(sorted(Counter(y.tolist()).items()))}")

    assert len(X) == len(y) == len(cfo)

print("\nSaved grouped dataset:", output_path)
print("Saved label map:", label_path)
print("\nDataset preparation completed successfully.")
