
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "anechoic_chamber"

print("Dataset folder:", DATA_DIR)
print("Dataset folder exists:", DATA_DIR.exists())

files = sorted(DATA_DIR.rglob("*.npz"))
print("NPZ files found:", len(files))

if not files:
    raise FileNotFoundError(
        f"No NPZ files found under {DATA_DIR}"
    )

stats = defaultdict(lambda: {
    "cfo": [],
    "snr": [],
    "files": 0,
})

for path in files:
    try:
        with np.load(path, allow_pickle=False) as data:
            bssid = str(data["BSSID"].item()).strip().upper()

            cfo = float(np.asarray(data["CFO"]).squeeze())
            snr = float(np.asarray(data["SNR"]).squeeze())

            stats[bssid]["cfo"].append(cfo)
            stats[bssid]["snr"].append(snr)
            stats[bssid]["files"] += 1

    except Exception as exc:
        print(f"Skipping {path.name}: {exc}")

print("\n=== CFO AND SNR BY CANDIDATE BSSID ===")

for bssid, values in sorted(stats.items()):
    cfo = np.asarray(values["cfo"])
    snr = np.asarray(values["snr"])

    print(f"\nBSSID: {bssid}")
    print(f"Recordings: {values['files']}")

    print(
        f"CFO: mean={cfo.mean():.4f}, "
        f"std={cfo.std():.4f}, "
        f"unique values={len(np.unique(cfo))}"
    )

    print(
        f"SNR: mean={snr.mean():.4f}, "
        f"std={snr.std():.4f}, "
        f"unique values={len(np.unique(snr))}"
    )
