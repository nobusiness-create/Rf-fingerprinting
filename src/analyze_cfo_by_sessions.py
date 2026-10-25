
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber" / "npy_files"

TRAIN_SESSIONS = {
    "2025_12_08_12_09_21",
    "2025_12_08_12_24_58",
    "2025_12_08_12_26_46",
    "2025_12_09_15_58_20",
}
TEST_SESSIONS = {"2025_12_19_08_48_36"}

rows = []

for path in sorted(DATA_DIR.glob("*.npz")):
    stem = path.stem.removesuffix("_anechoic_chamber")
    parts = stem.split("_")

    if len(parts) < 8 or not parts[-1].isdigit():
        print("Skipping unexpected filename:", path.name)
        continue

    session = "_".join(parts[1:-1])
    date = "_".join(parts[1:4])

    with np.load(path, allow_pickle=False) as data:
        label = str(data["BSSID"].item()).strip().upper()
        cfo = float(np.asarray(data["CFO"]).squeeze())

    if not np.isfinite(cfo):
        continue

    rows.append({
        "label": label,
        "session": session,
        "date": date,
        "cfo": cfo,
    })

classes = sorted({r["label"] for r in rows})

print(f"Total recordings: {len(rows)}")
print("\n=== CFO STATISTICS BY BSSID AND DATE ===")

for label in classes:
    print(f"\nBSSID: {label}")

    for date in sorted({r["date"] for r in rows}):
        values = np.array([
            r["cfo"] for r in rows
            if r["label"] == label and r["date"] == date
        ])

        if len(values) == 0:
            continue

        print(
            f"  {date}: n={len(values):2d}, "
            f"mean={values.mean():.3f}, "
            f"median={np.median(values):.3f}, "
            f"std={values.std():.3f}, "
            f"unique={len(np.unique(values))}"
        )

print("\n=== TRAIN VERSUS TEST CFO SHIFT ===")

for label in classes:
    train_values = np.array([
        r["cfo"] for r in rows
        if r["label"] == label and r["session"] in TRAIN_SESSIONS
    ])

    test_values = np.array([
        r["cfo"] for r in rows
        if r["label"] == label and r["session"] in TEST_SESSIONS
    ])

    if len(train_values) == 0 or len(test_values) == 0:
        print(f"{label}: missing training or test recordings")
        continue

    train_mean = train_values.mean()
    test_mean = test_values.mean()

    print(
        f"{label}: train_mean={train_mean:.3f}, "
        f"test_mean={test_mean:.3f}, "
        f"shift={test_mean - train_mean:+.3f}"
    )

print("\nCFO session analysis completed.")
