"""Train only the models included in the public release profile.

This intentionally skips CC BY-NC soil photos and aggregate FAOSTAT forecast models.
It never creates training examples; both reviewed local datasets must already exist.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
JOBS=[
    ("crop benchmark",ROOT/"data/raw/crop_recommendation_kaggle/Crop_recommendation.csv",[
        sys.executable,"scripts/train_tabular.py","--task","crop","--csv",
        "data/raw/crop_recommendation_kaggle/Crop_recommendation.csv","--target","label",
        "--output","models/crop_latest.npz","--epochs","200"]),
    ("EuroSAT land cover",ROOT/"data/raw/eurosat_rgb/EuroSAT_RGB",[
        sys.executable,"scripts/train_image.py","--task","landcover","--data",
        "data/raw/eurosat_rgb/EuroSAT_RGB","--epochs","12","--augmentation-copies","2",
        "--image-size","32","--out","models/landcover_latest.npz"]),
]

missing=[(name,source) for name,source,_ in JOBS if not (source.is_file() if source.suffix else source.is_dir())]
if missing:
    print("Missing reviewed public-profile datasets:")
    for name,path in missing: print(f"- {name}: {path}")
    print("Download/review sources using DATA_SOURCES.md and config/datasets.yaml, then rerun. No synthetic data is generated.")
    raise SystemExit(2)

for name,source,command in JOBS:
    print(f"\nTraining public-profile model: {name}",flush=True)
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode: raise SystemExit(result.returncode)
