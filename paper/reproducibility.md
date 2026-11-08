# Reproducibility and submission checklist

## Data and licenses

- Dataset URLs, license/terms notes, download timestamps, sizes, and checksums or pinned commit: `DATA_SOURCES.md` and `data/raw/MANIFEST.json`.
- Raw data are ignored by Git because of size. Re-download through `python scripts/download_data.py` using `config/datasets.yaml`.
- Soil photo data are CC BY-NC 4.0 and should only be used for non-commercial research/education under the repository terms. Verify underlying photo attribution before redistribution.
- To reproduce the supplemental CyAUG experiment, run `python scripts/download_soil_cyaug.py`, then `python scripts/audit_soil_cyaug.py`, then `python scripts/train_soil_transfer.py --epochs 10 --batch-size 32 --image-size 160 --cyaug-data data/raw/soil_classification_nc/cyaug_source --out models/soil_transfer_cyaug_candidate.pt`. The generated files were appended to training only; the original-only checkpoint remains the selected application model because the CyAUG candidate scored 83.15% validation accuracy and 0.768 macro-F1 versus 88.20% and 0.844 for the original-only validation run. The candidate's test score is exploratory: this test split was reused in prior experiments, and generated-image provenance is incomplete.
- Crop Recommendation Dataset is used as a benchmark; its own description says source tables were augmented.
- FAOSTAT price experiment predicts an annual relative index, not a monetary price. FAOSTAT yield is national kg/ha, not field yield.

## Reproduction commands

From `D:\CropAdvisor`, on Windows with Python 3.11:

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/clean_faostat.py
python scripts/clean_faostat_prices.py
python scripts/prepare_faostat_forecasts.py
python scripts/train_all.py
python scripts/train_soil_transfer.py
python scripts/evaluate_soil_transfer.py
python scripts/download_soil_core.py
python scripts/train_soil_core.py
pytest -q
cd frontend
npm ci
npm run build
```

The NumPy reference training scripts use seed 7, CPU execution, training-only normalization, validation-loss checkpoint selection (patience 15), and held-out test scoring. Their image inputs are 32×32 RGB. Soil transfer learning uses torchvision ResNet-18 ImageNet1K_V1 weights, a seeded stratified file-level split, 224×224 crops, and validation macro-F1 selection. `python scripts/evaluate_soil_transfer.py` compares deterministic flip averaging on validation before reporting the selected test method; its test set had already been scored once, so this second comparison is exploratory and must be replaced with an external test before a formal claim. The time-series split holds out complete years. Exact splits and metrics are written in model cards beside each model. Runtime can vary with CPU and Python/PyTorch versions.

The current soil run uses 30 requested epochs with three independently sampled train-only augmentation copies; its restored checkpoint is epoch 16. EuroSAT uses 12 requested epochs with two train-only augmentation copies. The full local research pipeline is `python scripts/train_all.py`. The later public-profile pipeline is `python scripts/train_public.py`; it trains only crop and EuroSAT models and does not fetch or train the non-commercial soil or annual FAOSTAT forecast models.

The soil-core experiment is a separate geotechnical task. It trains only on four frequent DIN main-fraction labels and groups all images by borehole identifier before assigning train, validation, and test partitions. Do not merge those labels or scores with the agricultural field-soil task.

## Current reported results

The manuscript reports results from `models/crop_latest_card.json`, `models/image_latest_card.json`, `models/soil_transfer_latest_card.json`, `models/soil_core_latest_card.json`, `models/landcover_latest_card.json`, `models/yield_latest_card.json`, and `models/price_latest_card.json`. Regenerate these artifacts before submission, then check the tables in `manuscript.md` against the cards. The crop and image scores are split-specific benchmark measures. Annual yield and price MLPs are inferior to persistence on the held-out test periods.

## Environment to record

Before paper submission, record exact Windows version, CPU/GPU model (these NumPy reference models currently train on CPU), Python, NumPy, Pandas, Pillow, FastAPI, and Node versions; repository commit; exact run date and command; and whether scores match the checked-in model cards. The project currently does not claim independent replication across hardware.

## Submission review

- Replace author, affiliation, contact, venue, and repository placeholders.
- Apply the conference's current template, page limit, and citation style.
- Verify all dataset licenses, source-image attribution, and repository visibility.
- Add the public code/data access statement that applies to the actual release.
- Preserve limitations and clearly distinguish benchmark predictions from field advice.
- Avoid presenting the mock product catalog as market data.
- Re-run the application and UI build from a clean environment.
