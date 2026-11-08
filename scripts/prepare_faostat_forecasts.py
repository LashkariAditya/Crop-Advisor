"""Build leakage-safe annual one-step forecasting tables from FAOSTAT data."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

YIELD_ITEMS = [
    "Rice", "Wheat", "Maize (corn)", "Millet", "Sorghum", "Chick peas, dry",
    "Lentils, dry", "Soya beans", "Groundnuts, excluding shelled", "Potatoes",
    "Sugar cane", "Bananas", "Onions and shallots, dry (excluding dehydrated)", "Tomatoes",
]
PRICE_ITEMS = ["Rice", "Wheat", "Maize (corn)", "Sorghum", "Millet", "Barley", "Chick peas, dry", "Groundnuts, excluding shelled"]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def add_item_indicators(frame: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    for item in items:
        frame[f"item__{slug(item)}"] = (frame["Item"] == item).astype(int)
    return frame


def build_yield() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data/processed/faostat_india_crop_production.csv")
    source = source[source["Item"].isin(YIELD_ITEMS)].copy()
    source["yield_kg_per_ha"] = pd.to_numeric(source["yield_kg_per_ha"], errors="coerce")
    source = source.dropna(subset=["yield_kg_per_ha"]).sort_values(["Item", "Year"])
    for lag in (1, 2, 3):
        source[f"yield_lag_{lag}"] = source.groupby("Item")["yield_kg_per_ha"].shift(lag)
    source["yield_persistence"] = source["yield_lag_1"]
    source = source.dropna(subset=["yield_lag_1", "yield_lag_2", "yield_lag_3"]).copy()
    source["date"] = pd.to_datetime(source["Year"].astype(str) + "-12-31")
    source = add_item_indicators(source, YIELD_ITEMS)
    columns = ["date", "Year", "Item", "yield_kg_per_ha", "yield_lag_1", "yield_lag_2", "yield_lag_3", "yield_persistence"] + [f"item__{slug(i)}" for i in YIELD_ITEMS]
    return source[columns].sort_values(["date", "Item"]).reset_index(drop=True)


def build_price_index() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data/processed/faostat_india_producer_prices.csv")
    source = source[source["Item"].isin(PRICE_ITEMS)].copy()
    source["producer_price_index"] = pd.to_numeric(source["Producer Price Index (2014-2016 = 100)"], errors="coerce")
    source = source.dropna(subset=["producer_price_index"]).sort_values(["Item", "Year"])
    source = source[source["producer_price_index"] > 0].copy()
    source["date"] = pd.to_datetime(source["Year"].astype(str) + "-12-31")
    source = add_item_indicators(source, PRICE_ITEMS)
    columns = ["date", "Year", "Item", "producer_price_index"] + [f"item__{slug(i)}" for i in PRICE_ITEMS]
    return source[columns].sort_values(["date", "Item"]).reset_index(drop=True)


def main() -> None:
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    yield_frame = build_yield()
    price_frame = build_price_index()
    yield_frame.to_csv(out / "faostat_india_yield_forecast.csv", index=False)
    price_frame.to_csv(out / "faostat_india_price_index_forecast.csv", index=False)
    report = {
        "source": "FAOSTAT India annual production/yield and annual producer price index; see DATA_SOURCES.md",
        "forecast_protocol": "One-step-ahead annual prediction. Sort by calendar year and hold out whole years; each feature row uses only prior-year target values.",
        "yield": {"rows": len(yield_frame), "items": sorted(yield_frame.Item.unique().tolist()), "year_min": int(yield_frame.Year.min()), "year_max": int(yield_frame.Year.max()), "target": "yield_kg_per_ha", "features": ["Year", "yield_lag_1", "yield_lag_2", "yield_lag_3", "item one-hot indicators"], "baseline": "yield_lag_1 persistence"},
        "producer_price_index": {"rows": len(price_frame), "items": sorted(price_frame.Item.unique().tolist()), "year_min": int(price_frame.Year.min()), "year_max": int(price_frame.Year.max()), "target": "producer_price_index", "units": "index (2014-2016=100); not INR/kg and not mandi retail price", "baseline": "price_lag_1 persistence"},
        "cautions": ["These are national annual series and cannot be interpreted as field-level yield or local mandi price forecasts.", "Producer-price index is a relative index, not a currency amount.", "All predictions are one-step annual estimates using historical lagged values only."],
    }
    (out / "faostat_forecast_data_card.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"yield_rows": len(yield_frame), "yield_years": [int(yield_frame.Year.min()), int(yield_frame.Year.max())], "price_index_rows": len(price_frame), "price_index_years": [int(price_frame.Year.min()), int(price_frame.Year.max())]}, indent=2))


if __name__ == "__main__":
    main()
