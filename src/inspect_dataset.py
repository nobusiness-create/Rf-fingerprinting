
from pathlib import Path
from collections import Counter
import json
from scipy.io import loadmat

DATA_DIR = Path("data/anechoic_chamber")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR.resolve()}")

files = [p for p in DATA_DIR.rglob("*") if p.is_file()]

print(f"Dataset folder: {DATA_DIR.resolve()}")
print(f"Total files: {len(files)}")

print("\n--- File extensions ---")
for extension, count in Counter(
    p.suffix.lower() for p in files
).most_common():
    print(f"{extension or '[no extension]'}: {count}")

# Inspect one MATLAB file
mat_files = list(DATA_DIR.rglob("*.mat"))

if mat_files:
    mat_path = mat_files[0]
    print("\n--- First MATLAB file ---")
    print("Name:", mat_path.name)
    print(f"Size: {mat_path.stat().st_size / (1024**2):.2f} MB")

    try:
        contents = loadmat(mat_path)
        for key, value in contents.items():
            if not key.startswith("__"):
                print(f"{key}: shape={getattr(value, 'shape', None)}, "
                      f"dtype={getattr(value, 'dtype', None)}")
    except Exception as error:
        print("Could not inspect MATLAB file:", error)
else:
    print("\nNo .mat files found.")

# Inspect one SigMF metadata file
metadata_files = [
    p for p in DATA_DIR.rglob("*")
    if p.name.lower().endswith(".sigmf-meta")
]

if metadata_files:
    meta_path = metadata_files[0]
    print("\n--- First SigMF metadata file ---")
    print("Name:", meta_path.name)

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        print(json.dumps(metadata, indent=2)[:5000])
    except Exception as error:
        print("Could not read metadata:", error)
else:
    print("\nNo .sigmf-meta files found.")
