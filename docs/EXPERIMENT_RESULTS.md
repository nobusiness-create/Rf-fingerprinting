
# RF Fingerprinting — Experimental Results

## 1. Project overview

The project investigates classification and anomaly detection using RF
recordings from five candidate Wi-Fi transmitters.

Dataset: RF Fingerprinting WLAN Dataset
Source: https://zenodo.org/records/21237610

The dataset describes five ESP32 devices transmitting 802.11b beacon
frames, measured in an anechoic chamber and an office environment.

This experiment uses the local anechoic-chamber subset containing
206 recordings across five BSSID labels.

## 2. Classification experiments

### Preliminary random-split feature baseline

| Method | Accuracy |
|---|---:|
| Magnitude features | 36.92% |
| CFO + SNR metadata | 78.46% |
| CFO-only random-split experiment | 93.55% |

These results are preliminary and should not be compared directly
with session-separated evaluation results.

### Session-separated evaluation

Four capture sessions were evaluated using leave-one-session-out
testing. Predictions were pooled at the recording level.

| Method | Accuracy | Balanced accuracy |
|---|---:|---:|
| CFO metadata only | 75.00% | 74.99% |
| SNR metadata only | 29.79% | 29.74% |
| CFO + SNR metadata | 75.00% | 75.16% |
| I/Q CNN | 19.15% | 19.16% |
| Preamble-only CNN | 19.68% | 19.86% |

The metadata baseline outperformed the current CNN implementations.
The CNN results were near the 20% chance-level reference for five
classes, showing poor generalization across capture sessions.

## 3. CFO estimation diagnostic

A preliminary phase-correlation estimator was tested on the complex
preamble signal.

Results:
- Median correlation coherence: 0.0146
- Median absolute error against supplied CFO metadata: approximately
  432,195 Hz

The estimator was not validated and must not be used as a reliable
CFO measurement. The supplied CFO metadata was not modified.

## 4. Anomaly-detection experiment

A preliminary open-set detector used the supplied CFO metadata and
nearest-centroid distances. One BSSID at a time was treated as unknown.

With per-class thresholds:

| Metric | CFO only | CFO + SNR |
|---|---:|---:|
| Known-device acceptance rate | 55.05% | 53.19% |
| Unknown-device detection rate | 85.00% | 55.00% |
| Balanced accuracy | 70.02% | 54.10% |

A threshold-percentile sweep from 50% to 99% did not change the
results. This indicates that threshold tuning using the current
validation data does not provide a useful improvement.

The detector is an exploratory metadata-based baseline, not a
validated anomaly detector operating directly on raw I/Q signals.

## 5. Main conclusions

1. CFO metadata contains useful information for distinguishing some
   candidate BSSIDs, but performance varies across capture sessions.
2. SNR alone performs poorly in the current experiments.
3. The current raw-I/Q and preamble CNNs do not generalize adequately
   across held-out sessions.
4. The direct CFO estimator did not reproduce the supplied metadata
   reliably.
5. The current anomaly detector has a substantial trade-off between
   unknown-device detection and accepting known devices.

## 6. Limitations and future work

- Verify the definition, units, and calculation method of the
  supplied CFO and SNR fields.
- Develop and validate a signal-derived CFO estimator.
- Improve CNN preprocessing and test session-independent generalization.
- Evaluate anomaly detection using raw signal features.
- Collect or use more independent capture sessions.
- Avoid using the same held-out test sessions repeatedly for model tuning.

All reported results are exploratory and should be interpreted in
light of the small number of capture sessions.
