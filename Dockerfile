# Build the browser UI with a same-origin API base. The public image contains
# only model artifacts whose datasets have redistributable terms.
FROM node:22-alpine AS frontend-build
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CROPADVISOR_PUBLIC_RELEASE=1 \
    CROPADVISOR_ALLOWED_ORIGINS=
WORKDIR /app
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY backend ./backend
COPY ml ./ml
COPY data/crops ./data/crops
COPY config ./config
COPY licenses ./licenses
COPY THIRD_PARTY_NOTICES.md ./THIRD_PARTY_NOTICES.md
COPY models/crop_latest.npz models/crop_latest_card.json ./models/
COPY models/landcover_latest.npz models/landcover_latest_card.json ./models/
COPY DATA_SOURCES.md ./DATA_SOURCES.md
COPY --from=frontend-build /ui/dist ./frontend/dist
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health',timeout=3).read()"
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
