"""Product adapter interface and clearly labeled demo catalog."""
from __future__ import annotations
from datetime import datetime, timezone


def mock_products(crop: str) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    # Demonstration-only prices. The source is explicitly labeled MOCK; never present as live market data.
    return [
        {"store": "MOCK demo catalog", "product": f"{crop} certified seed (demo 1 kg pack)", "price_inr": 120.0, "unit_price_inr": 120.0, "unit": "kg", "pack_size": 1, "url": "https://example.com/mock/seed", "last_updated": now, "mock": True},
        {"store": "MOCK demo catalog", "product": "NPK fertilizer (demo 1 kg pack)", "price_inr": 55.0, "unit_price_inr": 55.0, "unit": "kg", "pack_size": 1, "url": "https://example.com/mock/fertilizer", "last_updated": now, "mock": True},
    ]


def sort_by_unit_price(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: float(row["unit_price_inr"]))
