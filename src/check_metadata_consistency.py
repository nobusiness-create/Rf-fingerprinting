
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "anechoic_chamber"

matched = 0
mismatches = []

for npz_path in sorted(DATA_DIR.rglob("*.npz")):
    mat_path = npz_path.with_suffix(".mat")

    if not mat_path.exists():
        continue

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            bssid_npz = str(npz["BSSID"].item()).strip().upper()
            cfo_npz = float(np.asarray(npz["CFO"]).squeeze())
            snr_npz = float(np.asarray(npz["SNR"]).squeeze())

        mat = loadmat(mat_path, squeeze_me=True)
        bssid_mat = str(np.asarray(mat["BSSID"]).item()).strip().upper()
        cfo_mat = float(np.asarray(mat["CFO"]).squeeze())
        snr_mat = float(np.asarray(mat["SNR"]).squeeze())

        matched += 1

        differences = []
        if bssid_npz != bssid_mat:
            differences.append("BSSID")
        if not np.isclose(cfo_npz, cfo_mat, equal_nan=True):
            differences.append("CFO")
        if not np.isclose(snr_npz, snr_mat, equal_nan=True):
            differences.append("SNR")

        if differences:
            mismatches.append((npz_path.name, differences,
                               cfo_npz, cfo_mat, snr_npz, snr_mat))

    except Exception as exc:
        print(f"Could not compare {npz_path.name}: {exc}")

print(f"\nMatched MAT/NPZ pairs: {matched}")
print(f"Pairs with metadata differences: {len(mismatches)}")

for row in mismatches[:15]:
    name, fields, c_npz, c_mat, s_npz, s_mat = row
    print(f"\n{name}")
    print("Different fields:", ", ".join(fields))
    print(f"CFO: NPZ={c_npz}, MAT={c_mat}")
    print(f"SNR: NPZ={s_npz}, MAT={s_mat}")

if not mismatches:
    print("All matched files have consistent BSSID, CFO and SNR.")

