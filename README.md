# AI-Based Live Crop Recommendation System with Profitability and Rotation Planning

**CropAdvisor** is an India-focused project that brings crop suitability, local weather and soil context, cultivation notes, crop-rotation suggestions, and farm-budget estimates into one web application. It is designed as a local research prototype; its predictions are not a substitute for field trials or advice from an agricultural expert.

## What it does

- Ranks crops from soil and climate measurements using a trained suitability model.
- Retrieves available weather and soil context for coordinates supplied by the user.
- Shows sourced cultivation plans for rice and wheat, including rotation notes.
- Estimates costs, revenue, profit, break-even yield, and sensitivity scenarios from user-entered assumptions. Values are shown in INR and can be calculated for acres or hectares.
- Classifies selected field-soil or satellite images with separate models and labels.
- Provides annual yield and producer-price-index estimates for research; both are compared with simple persistence baselines.

The application reports unavailable data instead of filling gaps with invented values. Product listings are demonstration examples, not live offers or market prices.

## Run locally

### Requirements

- Windows with Python 3.11
- Node.js 20 or newer
- Internet access for live weather and soil lookups; those services may be unavailable at times

From the project folder, create and activate a Python environment, then install the API dependencies:

```powershell
cd D:\CropAdvisor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config/.env.example .env
```

Start the API in that terminal:

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the web interface:

```powershell
cd D:\CropAdvisor\frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The local API is at [http://127.0.0.1:8000](http://127.0.0.1:8000), with interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The optional mandi connector needs a current data.gov.in resource ID and API key in `.env`. Keep credentials out of source control. Without them, the app reports mandi data as unavailable.

## Model results and limits

The table summarizes the current recorded evaluations. These are dataset-specific results, not evidence of farm-level performance.

| Task | Recorded result | Interpretation |
|---|---|---|
| Crop suitability | 98.18% accuracy; 0.982 macro-F1 on a 330-row test split | Benchmark classification across 22 labels. The dataset description reports augmented source records. |
| Soil-photo classification | 89.01% accuracy; 0.854 macro-F1 with flip averaging | Exploratory result on a 182-image split that had already been evaluated without averaging. The underlying dataset is imbalanced and has non-commercial terms. |
| Satellite land cover | 71.31% accuracy; 0.703 macro-F1 on the official EuroSAT test split | Land-cover classification only; this model does not identify soil type. |
| Soil-core fractions | 69.46% accuracy; 0.555 macro-F1 on grouped borehole holdouts | Separate geotechnical task, not an agricultural soil classifier. |
| Annual yield and producer-price index | MLP test MAE is higher than persistence for both tasks | National annual trends only; not farm yield or local mandi prices. |

The soil-photo score remains below 90%. Adding 3,909 CycleGAN-generated images did not improve validation results, so that candidate was not selected. The soil-photo model is excluded from the public container because the dataset is licensed for non-commercial research and education. See the model cards for class-level results and split details, and [DATA_SOURCES.md](DATA_SOURCES.md) for source and license notes.

## Data and model reproduction

Raw datasets are excluded from Git. Dataset URLs, checksums, license notes, the pinned soil repository version, and download status are documented in [DATA_SOURCES.md](DATA_SOURCES.md) and `data/raw/MANIFEST.json`. Dataset download settings are in `config/datasets.yaml`.

To download and prepare the main datasets and train the available models:

```powershell
python scripts/download_data.py
python scripts/clean_faostat.py
python scripts/clean_faostat_prices.py
python scripts/prepare_faostat_forecasts.py
python scripts/train_all.py
```

The soil-photo ResNet-18 experiment is separate and uses PyTorch transfer learning:

```powershell
python scripts/train_soil_transfer.py
python scripts/evaluate_soil_transfer.py
```

The flip-averaging comparison reuses a previously scored test split and must be treated as exploratory. Do not select a model using the test set. See [paper/reproducibility.md](paper/reproducibility.md) for detailed commands, split methods, and submission checks.

## API overview

| Route | Purpose |
|---|---|
| `GET /health` | API status and available model artifacts |
| `POST /analyze` | Image, location, area, and optional soil/climate measurements; returns available context and model outputs |
| `GET /plan/{crop}` | Sourced cultivation plan for a supported crop |
| `GET /products/{crop}` | Demonstration product list; prices are mock values |
| `POST /economics` | Profitability estimate using supplied costs, yield, and price assumptions |
| `POST /economics/compare` | Compare supplied economics for several crops |
| `POST /forecast/yield` and `POST /forecast/price` | Local research forecasts with baseline comparison |
| `POST /classify/soil-core` | Separate geotechnical soil-core classification in local mode |

## Project structure

```text
backend/       API endpoints and external-data adapters
config/        project, dataset, training, and market settings
data/crops/    sourced cultivation plans
frontend/      React and Vite application
licenses/      third-party license notes
ml/            model implementations and evaluation metrics
models/        selected model artifacts and model cards
models/runs/   dated training checkpoints and their cards
paper/         manuscript draft and reproducibility guide
reports/       generated evaluation summaries
scripts/       data preparation, training, and evaluation commands
tests/         API, model, product, and economics tests
```

## Checks

```powershell
pytest -q
cd frontend
npm run build
```

## Public-build preview

The project is configured for local use and has not been deployed publicly. After review and approval, the public-profile container can be previewed on loopback:

```powershell
python scripts/download_data.py --dataset crop_recommendation_kaggle --dataset eurosat_rgb
python scripts/train_public.py
cd frontend
npm ci
npm run build
cd ..
docker compose -f docker-compose.public.yml up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). This profile omits the non-commercial soil-photo model and the annual aggregate forecast endpoints. Review [DEPLOYMENT.md](DEPLOYMENT.md), [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before selecting a host.

## Paper

The conference-paper draft is in [paper/manuscript.md](paper/manuscript.md). It still needs author and affiliation details, a target conference template, and a final check of the results against the model cards before submission.
