"""FastAPI endpoints for CropAdvisor with optional local model artifacts."""
from __future__ import annotations
import asyncio, io, json, os
import math
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError, ImageStat
from pydantic import BaseModel, Field
import numpy as np
from backend.adapters.products import mock_products, sort_by_unit_price
from backend.adapters.live_data import weather, weather_archive, soil, nasa_power, mandi_prices
from backend.services.pack_math import cheapest_pack_mix
from backend.services.economics import economics_report

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "data" / "crops"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
PUBLIC_RELEASE = os.getenv("CROPADVISOR_PUBLIC_RELEASE", "").lower() in {"1", "true", "yes"}
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("CROPADVISOR_ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if origin.strip()]
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="CropAdvisor", version="0.3.0", description="Crop-planning research prototype; estimates must be verified locally.", docs_url=None if PUBLIC_RELEASE else "/docs", redoc_url=None if PUBLIC_RELEASE else "/redoc", openapi_url=None if PUBLIC_RELEASE else "/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(self)")
    return response

class EconomicsRequest(BaseModel):
    crop: str
    area: float = Field(gt=0)
    unit: str = "acre"
    assumptions: dict[str, Any] = {}

class ForecastRequest(BaseModel):
    features: dict[str,float]

class ComparisonOption(BaseModel):
    crop: str
    assumptions: dict[str,Any]

class EconomicsCompareRequest(BaseModel):
    area: float = Field(gt=0)
    unit: str = "acre"
    crops: list[ComparisonOption]

@app.get("/health")
def health():
    return {"status": "ok", "public_release": PUBLIC_RELEASE, "crop_model_ready": (ROOT / "models" / "crop_latest.npz").exists(), "soil_model_ready": not PUBLIC_RELEASE and ((ROOT / "models" / "soil_transfer_latest.pt").exists() or (ROOT / "models" / "image_latest.npz").exists()), "soil_core_model_ready": (ROOT / "models" / "soil_core_latest.pt").exists(), "soil_model_note": "Disabled in public release pending commercial data rights; the source dataset is non-commercial." if PUBLIC_RELEASE else None, "landcover_model_ready": (ROOT / "models" / "landcover_latest.npz").exists(), "yield_model_ready": not PUBLIC_RELEASE and (ROOT / "models" / "yield_latest.npz").exists(), "price_model_ready": not PUBLIC_RELEASE and (ROOT / "models" / "price_latest.npz").exists(), "forecast_model_note": "National annual forecast models are omitted from the public build pending a separate data-rights and usefulness review." if PUBLIC_RELEASE else None, "note": "Research prototype. Predictions require local agronomic verification; no data are fabricated."}

@app.get("/climate/history")
def climate_history(latitude: float, longitude: float, start_date: str, end_date: str):
    return weather_archive(latitude,longitude,start_date,end_date)

@app.get("/climate/nasa-power")
def climate_power(latitude: float, longitude: float, start_date: str, end_date: str):
    return nasa_power(latitude,longitude,start_date,end_date)

@app.get("/market/mandi")
def market_mandi(state:str|None=None,district:str|None=None,commodity:str|None=None):
    filters={k:v for k,v in {"State":state,"District":district,"Commodity":commodity}.items() if v}
    return mandi_prices(filters=filters)

@app.get("/third-party-notices", include_in_schema=not PUBLIC_RELEASE)
def third_party_notices():
    return FileResponse(ROOT/"THIRD_PARTY_NOTICES.md",media_type="text/markdown")

@app.post("/forecast/{task}")
def forecast(task:str,req:ForecastRequest):
    if task not in {"yield","price"}: raise HTTPException(404,"Supported forecast tasks are yield and price.")
    if PUBLIC_RELEASE: raise HTTPException(404,"National annual forecast models are not included in this public build; use current local market and agricultural data.")
    path=ROOT/"models"/f"{task}_latest.npz"
    if not path.exists(): raise HTTPException(503,f"No trained {task} model is installed. Train it from reviewed local data first.")
    try:
        from backend.services.model_inference import regression_prediction
        result=regression_prediction(path,req.features)
        card=ROOT/"models"/f"{task}_latest_card.json"
        model_card=json.loads(card.read_text(encoding='utf-8')) if card.exists() else {}
        result["model_card"]=model_card or None
        result["target_units"]=model_card.get("target_units", "units not recorded")
        baseline_column=model_card.get("baseline_column")
        baseline_value=req.features.get(baseline_column) if baseline_column else None
        selected=str(model_card.get("validation_selected_predictor", "mlp")).lower()
        result["validation_selected_predictor"]=selected
        if baseline_value is not None and math.isfinite(baseline_value):
            result["persistence_baseline_prediction"]=baseline_value
            result["recommended_estimate"]=baseline_value if selected == "persistence" else result["prediction"]
        else:
            result["recommended_estimate"]=result["prediction"]
        if task == "price":
            result["warning"]="Experimental national FAOSTAT producer-price index only (2014-2016=100); not INR/kg or a local mandi quote. Verify with current market data."
        else:
            result["warning"]="Experimental India-level annual FAOSTAT yield trend in kg/ha; not a farm-level yield estimate."
        if selected == "persistence":
            result["warning"] += " Validation selected the last-observation persistence baseline over the MLP; recommended_estimate uses that baseline when supplied."
        return result
    except ValueError as exc: raise HTTPException(422,str(exc))

@app.post("/analyze")
async def analyze(image: UploadFile = File(...), latitude: float = Form(...,ge=-90,le=90), longitude: float = Form(...,ge=-180,le=180), area: float = Form(...), area_unit: str = Form("acre"), photo_type: str = Form("field"), nitrogen: float | None = Form(None), phosphorus: float | None = Form(None), potassium: float | None = Form(None), temperature: float | None = Form(None), humidity: float | None = Form(None), ph: float | None = Form(None), rainfall: float | None = Form(None)):
    if image.content_type not in ALLOWED_TYPES: raise HTTPException(415, "Upload a JPEG, PNG, or WebP field photo.")
    if not math.isfinite(area) or area <= 0 or area_unit not in {"acre", "hectare"}: raise HTTPException(422, "Enter a positive area in acres or hectares.")
    if photo_type not in {"field","satellite"}: raise HTTPException(422,"photo_type must be field or satellite")
    payload = await image.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES: raise HTTPException(413, "Image exceeds the 8 MiB upload limit.")
    try:
        with Image.open(io.BytesIO(payload)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS: raise HTTPException(413, "Image dimensions exceed the 20-megapixel limit.")
            img.verify()
        with Image.open(io.BytesIO(payload)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS: raise HTTPException(413, "Image dimensions exceed the 20-megapixel limit.")
            clean = img.convert("RGB")
            clean.thumbnail((1024, 1024))
    except Image.DecompressionBombError: raise HTTPException(413, "Image dimensions exceed the safe processing limit.")
    except (UnidentifiedImageError, OSError): raise HTTPException(400, "The uploaded file is not a supported image.")
    climate, soil_data = await asyncio.gather(
        asyncio.to_thread(weather, latitude, longitude),
        asyncio.to_thread(soil, latitude, longitude),
    )
    image_path=ROOT/"models"/("landcover_latest.npz" if photo_type=="satellite" else ("soil_transfer_latest.pt" if (ROOT/"models"/"soil_transfer_latest.pt").exists() else "image_latest.npz")); photo_prediction=None
    soil_photo_allowed = photo_type == "satellite" or not PUBLIC_RELEASE
    if soil_photo_allowed and image_path.exists():
        try:
            from backend.services.model_inference import image_model_size, predict_image
            image_size = image_model_size(image_path)
            arr=np.asarray(clean.resize((image_size,image_size)),dtype=np.float32)[None]/255
            photo_prediction=await asyncio.to_thread(predict_image,image_path,arr)
        except Exception: photo_prediction=None
    crop_path=ROOT/"models"/"crop_latest.npz"; crop_results=[]; prediction_ready=False
    supplied={"N":nitrogen,"P":phosphorus,"K":potassium,"temperature":temperature,"humidity":humidity,"ph":ph,"rainfall":rainfall}
    if crop_path.exists() and all(v is not None for v in supplied.values()):
        try:
            from backend.services.model_inference import predict_crop
            crop_results=await asyncio.to_thread(predict_crop,crop_path,supplied); prediction_ready=True
        except Exception: crop_results=[]
    blurry=ImageStat.Stat(clean.convert('L')).stddev[0]<3.0
    low_confidence=not prediction_ready or (crop_results[0]["confidence"]<.6 if crop_results else True) or (photo_prediction is not None and photo_prediction['confidence']<.6) or blurry
    warning=("Uploaded field image has very low contrast; try a clearer close-up photo. " if blurry else "")
    warning += ("No crop model is trained, or required soil/climate values are missing. Train from reviewed data and enter values matching its training units." if not prediction_ready else ("Top crop probability is below 60%; treat this as a low-confidence estimate." if crop_results and crop_results[0]["confidence"]<.6 else ""))
    if PUBLIC_RELEASE and prediction_ready: warning += " Crop rankings come from an augmented benchmark dataset and are not field-validated; verify with a local agricultural expert."
    if PUBLIC_RELEASE and photo_type == "field": warning += " Soil-photo classification is disabled in this public build because its training dataset has non-commercial terms."
    if photo_prediction is not None and photo_prediction['confidence']<.6: warning += " Image model confidence is below 60%."
    return {"location": {"latitude": latitude, "longitude": longitude}, "area": area, "area_unit": area_unit,
            "image_dimensions": list(clean.size), "metadata_stripped": True,
            "photo_prediction": photo_prediction, "photo_type": photo_type, "climate": climate, "soil_context": soil_data, "ranked_crops": crop_results, "prediction_ready": prediction_ready,
            "low_confidence": low_confidence, "warning": warning or None,
            "photo_guidance": "Soil-photo classification is disabled in the public build pending data rights." if PUBLIC_RELEASE and photo_type == "field" else "Upload a clear close-up soil/field photograph; satellite-scene images are not suitable for soil type."}

@app.post("/classify/soil-core")
async def classify_soil_core(image: UploadFile = File(...)):
    """Separate geotechnical soil-core task; its labels are not farm soil types."""
    if image.content_type not in ALLOWED_TYPES: raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
    path=ROOT/"models"/"soil_core_latest.pt"
    if not path.exists(): raise HTTPException(503, "Soil-core model is not trained yet.")
    payload=await image.read(MAX_IMAGE_BYTES+1)
    if len(payload)>MAX_IMAGE_BYTES: raise HTTPException(413,"Image exceeds the 8 MiB upload limit.")
    try:
        with Image.open(io.BytesIO(payload)) as img:
            if img.width*img.height>MAX_IMAGE_PIXELS: raise HTTPException(413,"Image dimensions exceed the 20-megapixel limit.")
            img.verify()
        with Image.open(io.BytesIO(payload)) as img:
            size=img.size; clean=img.convert('RGB'); clean.thumbnail((1024,1024))
    except Image.DecompressionBombError: raise HTTPException(413,"Image dimensions exceed the safe processing limit.")
    except (UnidentifiedImageError,OSError): raise HTTPException(400,"The uploaded file is not a supported image.")
    from backend.services.model_inference import image_model_size,predict_image
    n=image_model_size(path); arr=np.asarray(clean.resize((n,n)),dtype=np.float32)[None]/255
    pred=await asyncio.to_thread(predict_image,path,arr)
    return {"prediction":pred,"task":"geotechnical soil-core main fraction","image_dimensions":list(size),"warning":"This research model classifies soil-core geotechnical fractions. It is not the agricultural field-soil classifier and does not measure nutrients, pH, or moisture."}

@app.get("/plan/{crop}")
def plan(crop: str):
    path = PLANS / f"{crop.lower().replace(' ', '_')}.json"
    if not path.exists(): raise HTTPException(404, f"No sourced cultivation plan is installed for {crop}.")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/products/{crop}")
def products(crop: str):
    return {"crop": crop, "prices_are_live": False, "items": sort_by_unit_price(mock_products(crop)), "warning": "MOCK demonstration prices only; do not use for purchasing or economics."}

@app.post("/products/quantity")
def product_quantity(body: dict[str, Any]):
    try: return cheapest_pack_mix(float(body.get("required_quantity", 0)),body.get("packs",[]))
    except (ValueError,KeyError,TypeError) as exc: raise HTTPException(422,str(exc))

@app.post("/economics")
def economics(req: EconomicsRequest):
    try:
        assumptions=req.assumptions.copy()
        plan_path=PLANS/f"{req.crop.lower().replace(' ','_')}.json"
        if plan_path.exists(): assumptions.setdefault('rotation_suggestion',json.loads(plan_path.read_text(encoding='utf-8')).get('rotation_suggestion'))
        return economics_report(req.crop, req.area, req.unit, assumptions)
    except ValueError as exc: raise HTTPException(422, str(exc))

@app.post("/economics/compare")
def compare_economics(req:EconomicsCompareRequest):
    if req.unit not in {"acre","hectare"}: raise HTTPException(422,"unit must be acre or hectare")
    rows=[]
    for option in req.crops:
        try: rows.append(economics_report(option.crop,req.area,req.unit,option.assumptions))
        except ValueError as exc: raise HTTPException(422,f"{option.crop}: {exc}")
    rows.sort(key=lambda r:r['net_profit_inr'],reverse=True)
    return {"ranked_by":"estimated net profit INR","items":rows,"warning":"Comparisons are estimates from supplied assumptions, not guarantees."}

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def frontend_index():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/privacy.html", include_in_schema=False)
    def privacy_page():
        return FileResponse(FRONTEND_DIST / "privacy.html")
