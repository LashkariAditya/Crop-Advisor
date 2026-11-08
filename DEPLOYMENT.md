# Run and deployment guide

## Local development

1. Install Python 3.11 and Node.js 20 or newer.
2. From `D:\CropAdvisor`, create a virtual environment, install `requirements.txt`, and copy `config/.env.example` to `.env`.
3. Start the API with `uvicorn backend.app:app --host 127.0.0.1 --port 8000`.
4. In a second terminal, run `cd frontend`, `npm ci`, and `npm run dev`.
5. Open `http://localhost:5173`. Local API docs are at `http://127.0.0.1:8000/docs`.

The standard `docker-compose.yml` is for local development and binds both services to loopback. It mounts local data and model artifacts. Do not expose it directly to the internet.

## Production-profile preview on this computer

The root `Dockerfile` builds the React application and FastAPI API into one container. It serves the UI and API from the same origin, hides interactive API documentation, adds basic response security headers, applies an 8 MiB upload limit and a 20-megapixel image limit, and runs one API worker. The `docker-compose.public.yml` preview binds to `127.0.0.1:8000` only:

```powershell
docker compose -f docker-compose.public.yml up --build
```

It packages the crop and satellite models. It intentionally omits the non-commercial soil-photo model and the weak FAOSTAT aggregate forecast models. Public mode returns this policy from `/health`, disables soil-photo classification and yield/price forecast routes, and caches external climate/soil responses in memory rather than writing coordinate-bearing responses to disk. It does not contact a public host or expose the local machine to the network.

## Before a future public deployment

The project is intentionally not deployed until teacher acceptance. Read [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md). Select the host and URL, configure HTTPS, exact CORS origins for any split frontend/API, trusted proxy handling, host-level request rate limits and abuse monitoring, and review host log retention. The application does not implement distributed rate limiting; this matters because anonymous analysis requests call third-party weather and soil services. Store API credentials only in the host's secret manager. Update the privacy page to match the actual host and local requirements. Keep mock retail data and all model warnings visible.

The crop classifier was trained on a benchmark described as augmented and is not farm validated. The annual yield and price networks underperform persistence and are not included in public mode. The local soil-image model uses data with non-commercial terms and is not included in the public container. Public suitability of any future model or dataset must be checked separately.
