
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data/anechoic_chamber")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

files = sorted(DATA_DIR.rglob("*.npz"))
if not files:
    raise FileNotFoundError(f"No NPZ files found in {DATA_DIR.resolve()}")

# Select one recording per BSSID
recordings = {}

for path in files:
    with np.load(path, allow_pickle=False) as data:
        bssid = str(data["BSSID"].item())

        if bssid in recordings:
            continue

        i = data["beaconDataDownSampledI"].astype(np.float32)
        q = data["beaconDataDownSampledQ"].astype(np.float32)
        sr = float(data["baseBandSampleRate"].item())

        recordings[bssid] = (i + 1j * q, sr, path.name)

    if len(recordings) == 5:
        break

fig, axes = plt.subplots(5, 1, figsize=(12, 12))

for ax, (bssid, (signal, sr, filename)) in zip(
    axes, sorted(recordings.items())
):
    count = min(1000, len(signal))
    time_us = np.arange(count) / sr * 1e6

    # Normalize each recording to compare shape, not absolute gain.
    magnitude = np.abs(signal[:count])
    magnitude = magnitude / (np.mean(magnitude) + 1e-12)

    ax.plot(time_us, magnitude)
    ax.set_title(f"BSSID: {bssid}")
    ax.set_ylabel("Normalized magnitude")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (microseconds)")
fig.suptitle("One example recording per candidate transmitter", y=1.01)
fig.tight_layout()

output = OUT_DIR / "device_signal_comparison.png"
fig.savefig(output, dpi=150, bbox_inches="tight")
print("Devices plotted:", len(recordings))
print("Saved:", output.resolve())

plt.show()
