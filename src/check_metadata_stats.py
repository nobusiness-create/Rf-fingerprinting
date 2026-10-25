
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"

stats = defaultdict(lambda: {"CFO": [], "SNR": []})

for path in sorted(DATA_DIR.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        bssid = str(data["BSSID"].item()).strip().upper()

        for field in ("CFO", "SNR"):
            value = float(np.asarray(data[field]).squeeze())
            if np.isfinite(value):
                stats[bssid][field].append(value)

for bssid, features in sorted(stats.items()):
    print(f"\n{'=' * 55}")
    print(f"BSSID: {bssid}")

    for name, values in features.items():
        x = np.asarray(values)

        print(f"\n{name} (n={len(x)})")
        print(f"  Min:    {np.min(x):.3f}")
        print(f"  Q1:     {np.percentile(x, 25):.3f}")
        print(f"  Median: {np.median(x):.3f}")
        print(f"  Q3:     {np.percentile(x, 75):.3f}")
        print(f"  Max:    {np.max(x):.3f}")
        print(f"  Unique values: {len(np.unique(x))}")
