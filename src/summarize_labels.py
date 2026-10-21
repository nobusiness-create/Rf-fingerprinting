
from pathlib import Path
from collections import Counter
from scipy.io import loadmat
import numpy as np

DATA_DIR = Path("data/anechoic_chamber")

mat_files = sorted(DATA_DIR.rglob("*.mat"))

if not mat_files:
    raise FileNotFoundError(f"No .mat files found in {DATA_DIR.resolve()}")

bssid_counts = Counter()
ssid_counts = Counter()
filename_prefix_counts = Counter()
failed_files = []

for path in mat_files:
    try:
        data = loadmat(path)

        bssid = str(np.asarray(data["BSSID"]).squeeze()).strip()
        ssid = str(np.asarray(data["SSID"]).squeeze()).strip()

        bssid_counts[bssid] += 1
        ssid_counts[ssid] += 1

        # Record the filename prefix separately.
        # Do not assume it identifies a transmitter yet.
        filename_prefix_counts[path.name.split("_")[0]] += 1

    except Exception as exc:
        failed_files.append((path.name, str(exc)))

print(f"Total MATLAB files: {len(mat_files)}")

print("\n--- Unique BSSIDs ---")
print(f"Count: {len(bssid_counts)}")
for label, count in sorted(bssid_counts.items()):
    print(f"{label}: {count} recordings")

print("\n--- Unique SSIDs ---")
for label, count in ssid_counts.most_common():
    print(f"{label}: {count} recordings")

print("\n--- Filename prefixes (not confirmed device labels) ---")
for prefix, count in sorted(filename_prefix_counts.items()):
    print(f"{prefix}: {count} recordings")

# Inspect the first NPZ archive's structure
npz_files = sorted(DATA_DIR.rglob("*.npz"))

if npz_files:
    print("\n--- First NPZ archive ---")
    with np.load(npz_files[0], allow_pickle=False) as archive:
        for key in archive.files:
            array = archive[key]
            print(
                f"{key}: shape={array.shape}, "
                f"dtype={array.dtype}"
            )

if failed_files:
    print(f"\nFiles that could not be inspected: {len(failed_files)}")
    for name, error in failed_files[:5]:
        print(name, error)
