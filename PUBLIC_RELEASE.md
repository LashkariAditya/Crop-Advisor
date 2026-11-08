# Public release preparation (not deployed)

The project remains local until the teacher accepts it. No public service, domain, or repository has been created. The release profile below is packaged for a local preview and later review.

## Local preview of the production profile

Train only the two model artifacts included in the release profile, then build the frontend:

```powershell
python scripts/download_data.py --dataset crop_recommendation_kaggle --dataset eurosat_rgb
python scripts/train_public.py
cd frontend
npm ci
npm run build
cd ..
docker compose -f docker-compose.public.yml up --build
```

Open http://127.0.0.1:8000. The compose file binds to loopback only. Stop it with Ctrl+C. The root `Dockerfile` builds the frontend and FastAPI API into one container and uses one same-origin URL. It packages only the crop and EuroSAT model files. Model `.npz` files are local build inputs and are excluded from Git by default.

## Data and model release choices

- Crop model: the Kaggle dataset page reports Apache 2.0. Keep its attribution and benchmark caveat with the model card; its high random-split score is not field validation.
- EuroSAT model: land-cover classification only. Retain dataset and Copernicus attribution/terms in the source inventory.
- Soil-photo model: excluded from the public container. Its source repository specifies CC BY-NC 4.0 and notes that source image rights may vary. Local research use must follow those terms; do not expose or redistribute its weights until rights are resolved.
- Yield and price-index models: excluded from the public container and endpoints. They do not beat persistence and use national aggregate data; their price target is an index, not INR/kg.
- Product catalog: all records are marked mock and must never be described as live retail prices.

The public profile returns this policy through `/health`. It blocks soil-photo classification and annual yield/price forecast routes even if their local weights happen to exist. Public builds use an in-memory cache for climate/soil API responses, so coordinate-bearing responses are not written to disk.

## Required before an internet-facing launch

1. Obtain the teacher's acceptance and choose the host, public URL, project owner, and maintainer contact.
2. Configure HTTPS at the host, exact CORS origins if UI and API are split, trusted proxy handling, and request limits for image analysis and third-party data routes. The application alone does not implement a distributed rate limiter.
3. Review provider logs/retention and adjust `frontend/public/privacy.html` to the actual host and applicable requirements. Coordinates are sent to Open-Meteo and SoilGrids for context; optional climate endpoints call NASA POWER or Open-Meteo archive. Uploaded images are processed by this service and are not sent to those data providers.
4. Review licenses/attribution in `DATA_SOURCES.md`, model cards, and the selected host's terms. Do not add API keys to the image or browser bundle. Store any future credentials in the hosting provider's secret manager.
5. Review the public crop guidance, plan regional limits, benchmark feature units, accessibility, mobile layout, load behavior, abuse controls, API outage behavior, and recovery procedure with the teacher.
6. Run the complete tests and frontend build from a clean checkout with trained artifacts, verify `/health`, check logs, and have an agricultural domain expert review all public-facing claims.

## Safety and scope

CropAdvisor is a research/education prototype. The crop classifier uses a dataset described as augmented; soil-photo classes are not soil tests; satellite labels are not soil labels; and economics depend on user inputs while retail data are mock. Do not present the software as a farm-validated agronomic or financial service. The project has no user accounts and no authentication; a public host must apply suitable network-level rate controls and monitor external API usage before exposure.
