# CropAdvisor

CropAdvisor is a local-first India-focused crop planning prototype. It combines a React/Vite interface, a FastAPI service, transparent farm economics, sourced cultivation guidance, NumPy models trained from scratch, and a locally run PyTorch transfer-learning soil-photo model. Model outputs are experimental decision support and require local agronomic verification.

## Start the application (Windows)

```powershell
cd D:\CropAdvisor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config/.env.example .env
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In another PowerShell window:

```powershell
cd D:\CropAdvisor\frontend
npm install
npm run dev
```

Open http://localhost:5173. Interactive API documentation is at http://127.0.0.1:8000/docs. For container deployment see [DEPLOYMENT.md](DEPLOYMENT.md). Current product prices are labelled MOCK. Mandi API access requires a current data.gov.in resource ID and API key.

## Current trained models

The crop, satellite, and time-series reference models are trained from scratch using NumPy. Soil-photo and soil-core experiments use ImageNet-pretrained ResNet-18 weights through PyTorch. Exact settings, split definitions, class metrics, and caveats are recorded in `models/*_latest_card.json`.

| Task | Data and held-out evaluation | Intended interpretation |
|---|---|---|
| Crop suitability | Kaggle Crop Recommendation Dataset, 2,200 rows and 22 classes; stratified 70/15/15 split; current test accuracy 98.18%, macro-F1 0.982 | Benchmark classification only. Source describes augmented records. Not evidence of farm-level generalization. |
| Soil-photo classes | 1,189 original photos, 7 repository labels; selected ImageNet-pretrained ResNet-18. Exploratory flip-averaged test accuracy 89.01%, macro-F1 0.854 | Photo label helper only. Non-commercial dataset with uncertain underlying photo provenance; not a soil test or moisture estimate. Not included in the public container. Score reuses a previously scored test split. A separate 3,909-image CyAUG experiment reached 83.15% validation accuracy (macro-F1 0.768) and was not promoted because it underperformed the original-only model. |
| Soil core main fractions | 3,604 MIT-labeled soil-core images; ResNet-18; 10/2/3 boreholes held out for train/validation/test; grouped test accuracy 69.46%, macro-F1 0.555 | Separate geotechnical research task. Does not share labels with the field-soil classifier. |
| Satellite land cover | EuroSAT RGB, 27,000 images, 10 classes; official 3,000/4,050/4,050 split; 12 epochs with two train-only augmented views per image; test accuracy 71.31%, macro-F1 0.703 | Satellite land-cover helper only; it does not classify soil or recommend crops. |
| National annual yield trend | FAOSTAT India yield records, 14 crop series; chronological split, test starts 2015; test MAE 1,871 kg/ha (MLP) vs 529 kg/ha (last-value persistence) | The MLP loses to the baseline; API identifies persistence as validation-selected. National annual trend, not farm yield. |
| National producer-price index | FAOSTAT annual producer-price index; 8 items; chronological test begins 2021; test MAE 16.78 index points (MLP) vs 9.35 (persistence) | The MLP loses to the baseline; API identifies persistence as validation-selected. Index (2014-2016=100), not INR/kg or mandi price. |

Metrics are benchmark results, not claims of agronomic validity. Random image/crop splits may share source or collection context, and the small soil set is imbalanced. The price/yield time series are national aggregates. More data alone does not make those models field-grade.

The selected soil-photo model gained about 4.4 percentage points of test accuracy versus the earlier one-augmentation run; some minority-class performance remains poor. A supplemental CyAUG run with 3,909 generated images did not improve validation and is not selected. The selected model is for non-commercial research only and is excluded from the public build because its source dataset has CC BY-NC terms. Two train-only augmented views raised EuroSAT test accuracy from 70.86% to 71.31% and macro-F1 from 0.694 to 0.703 on the fixed official split; the increase is modest.

## Reproduce data and training

Raw datasets are excluded from Git due to size. Their sources, licenses, checksums, pinned soil repository commit, download manifest, and caveats are recorded in [DATA_SOURCES.md](DATA_SOURCES.md) and `data/raw/MANIFEST.json`. The dataset download registry is `config/datasets.yaml`.

```powershell
python scripts/download_data.py
python scripts/clean_faostat.py
python scripts/clean_faostat_prices.py
python scripts/prepare_faostat_forecasts.py
python scripts/train_all.py
```

The NumPy reference models use CPU. The local soil-photo experiment uses PyTorch transfer learning with torchvision ResNet-18 and also runs on CPU; it uses validation-selected checkpoints. Run `python scripts/train_soil_transfer.py` to train and `python scripts/evaluate_soil_transfer.py` to compare deterministic flip averaging. The latter reuses a test split previously scored once, so it is exploratory and requires a fresh external test before a formal claim. Soil-photo accuracy remains below 90%. Do not select synthetic-data runs or models using the test split. `train_all.py` skips a task when its reviewed data are absent.

The separate soil-core dataset/model is downloaded and retrained with `python scripts/download_soil_core.py` and `python scripts/train_soil_core.py`. It uses borehole-group splits because multiple photos from each borehole appear across the dataset's supplied splits. Its labels describe geotechnical fractions and are not agricultural soil types. The result is available from `POST /classify/soil-core` only in local mode and only as a clearly separate research output.

## API predictions

- `POST /analyze`: field photo plus coordinates, area, and optional complete N/P/K, temperature, humidity, pH, rainfall inputs; returns live-context availability, crop ranking if inputs/model are present, and warnings. The local research mode classifies soil photos; the public build disables that non-commercial model and retains satellite land-cover classification.
- `POST /classify/soil-core`: standalone geotechnical core main-fraction classifier. Its output is separate from agricultural field-soil classification.
- `POST /forecast/yield` and `/forecast/price`: experimental local-only annual models. The public build omits these endpoints. In local mode, supply model features and the baseline lag column (`yield_persistence` or `price_lag_1`) to receive `recommended_estimate` and the raw neural-network prediction. The validation-selected persistence predictor is preferred for both current datasets.
- `GET /plan/{crop}`: sourced plans for rice and wheat, with region caveats.
- `GET /products/{crop}`: demonstration-only mock products; not live purchase data.
- `/economics` and `/economics/compare`: calculations from editable user assumptions, reported in INR and for acres/hectares.

The crop benchmark's measurement units and regional applicability must be checked before supplying inputs. Open-Meteo/NASA/SoilGrids availability is not guaranteed. An unavailable data service is reported as unavailable rather than replaced with invented values.

The Analyze screen reads `/health` and displays which model artifacts are available or disabled; artifact availability is not a quality or field-validation claim.

## Project and paper materials

- Conference-paper draft: [`paper/manuscript.md`](paper/manuscript.md)
- Reproducibility checklist and limits: [`paper/reproducibility.md`](paper/reproducibility.md)
- Dataset and license inventory: [`DATA_SOURCES.md`](DATA_SOURCES.md)
- Public bundle notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- API/deployment notes: [`DEPLOYMENT.md`](DEPLOYMENT.md)

The paper is a draft for adaptation: add the real author names/affiliations, select a conference template, verify its current citation format, and re-render the final copy before submission. It deliberately reports weak baselines and limitations rather than implying field validation.

## Local public-build preview (not deployed)

After the teacher approves the project, the single-container profile can be checked locally without exposing it to the network:

```powershell
python scripts/download_data.py --dataset crop_recommendation_kaggle --dataset eurosat_rgb
python scripts/train_public.py
cd frontend
npm ci
npm run build
cd ..
docker compose -f docker-compose.public.yml up --build
```

Open http://127.0.0.1:8000. This profile packages only the crop and satellite models, disables the non-commercial soil model and annual FAOSTAT forecast endpoints, serves the UI and API from one origin, and binds to loopback. It is a launch candidate, not a live public deployment. Review [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md) before choosing a public host.

## Checks

```powershell
pytest -q
cd frontend
npm run build
```
