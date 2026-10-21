
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data/anechoic_chamber")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load one recording
files = sorted(DATA_DIR.rglob("*.npz"))
if not files:
    raise FileNotFoundError(f"No NPZ files found in {DATA_DIR.resolve()}")

file_path = files[0]

with np.load(file_path, allow_pickle=False) as data:
    I = data["beaconDataDownSampledI"].astype(np.float32)
    Q = data["beaconDataDownSampledQ"].astype(np.float32)
    bssid = str(data["BSSID"].item())
    sample_rate = float(data["baseBandSampleRate"].item())

signal = I + 1j * Q

print("File:", file_path.name)
print("BSSID:", bssid)
print("Number of samples:", len(signal))
print("Sample rate:", sample_rate / 1e6, "MHz")
print("Signal finite:", np.isfinite(I).all() and np.isfinite(Q).all())

# Plot I/Q waveform and frequency spectrum
fig, axes = plt.subplots(3, 1, figsize=(11, 9))

count = min(1000, len(signal))
time_us = np.arange(count) / sample_rate * 1e6

axes[0].plot(time_us, I[:count], label="I")
axes[0].plot(time_us, Q[:count], label="Q", alpha=0.8)
axes[0].set_title("Wi-Fi signal: I and Q components")
axes[0].set_xlabel("Time (microseconds)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_us, np.abs(signal[:count]))
axes[1].set_title("Signal magnitude")
axes[1].set_xlabel("Time (microseconds)")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True, alpha=0.3)

# Centered FFT spectrum
spectrum = np.fft.fftshift(np.fft.fft(signal))
frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(signal), d=1 / sample_rate)
) / 1e6

power_db = 20 * np.log10(np.abs(spectrum) / len(signal) + 1e-12)

axes[2].plot(frequencies, power_db)
axes[2].set_title("Signal frequency spectrum")
axes[2].set_xlabel("Frequency offset (MHz)")
axes[2].set_ylabel("Magnitude (dB)")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()

output_path = RESULTS_DIR / "first_signal.png"
fig.savefig(output_path, dpi=150)
print("Plot saved to:", output_path.resolve())

plt.show()
