"""Transparent crop economics calculations with user supplied assumptions."""
from __future__ import annotations

from typing import Any
import math

ACRE_HECTARES = 0.40468564224


def economics_report(crop: str, area: float, unit: str, assumptions: dict[str, Any]) -> dict[str, Any]:
    if not math.isfinite(area) or area <= 0 or unit not in {"acre", "hectare"}:
        raise ValueError("area must be positive and unit must be acre or hectare")
    hectares = area * ACRE_HECTARES if unit == "acre" else area
    costs = assumptions.get("costs", {})
    cost_per_ha = {str(k): float(v) for k, v in costs.items()}
    if any(not math.isfinite(v) or v < 0 for v in cost_per_ha.values()):
        raise ValueError("cost inputs cannot be negative")
    total_per_ha = sum(cost_per_ha.values())
    total_cost = total_per_ha * hectares
    yield_kg_ha = float(assumptions.get("yield_kg_per_ha", 0))
    price_per_kg = float(assumptions.get("price_inr_per_kg", 0))
    yield_low = float(assumptions.get("yield_low_kg_per_ha", yield_kg_ha * .8))
    yield_high = float(assumptions.get("yield_high_kg_per_ha", yield_kg_ha * 1.2))
    price_floor = assumptions.get("floor_price_inr_per_kg")
    if price_floor is not None:
        price_floor = float(price_floor)
        if not math.isfinite(price_floor) or price_floor < 0: raise ValueError("floor price must be finite and nonnegative")
    if any(not math.isfinite(v) or v < 0 for v in (yield_kg_ha,price_per_kg,yield_low,yield_high)):
        raise ValueError("yield and price inputs cannot be negative")
    if yield_low>yield_high: raise ValueError("low yield cannot exceed high yield")
    revenue = yield_kg_ha * price_per_kg * hectares
    break_even_price = total_per_ha / yield_kg_ha if yield_kg_ha else None
    break_even_yield = total_per_ha / price_per_kg if price_per_kg else None
    duration = float(assumptions.get("duration_months", 1))
    if not math.isfinite(duration) or duration<=0: raise ValueError("duration_months must be positive")
    scenarios = {}
    for name, y_factor, p_factor, c_factor in (("pessimistic", .8, .8, 1.2), ("expected", 1, 1, 1), ("optimistic", 1.2, 1.2, .9)):
        scenario_revenue = yield_kg_ha * y_factor * price_per_kg * p_factor * hectares
        scenario_cost = total_cost * c_factor
        scenarios[name] = {"revenue_inr": scenario_revenue, "cost_inr": scenario_cost, "net_profit_inr": scenario_revenue - scenario_cost}
    sensitivity = []
    for factor in (.8, 1.0, 1.2):
        for variable in ("price", "yield", "input_cost"):
            r = revenue * (factor if variable == "price" else 1)
            r *= factor if variable == "yield" else 1
            c = total_cost * (factor if variable == "input_cost" else 1)
            sensitivity.append({"variable": variable, "change_pct": round((factor - 1) * 100), "net_profit_inr": r - c})
    sources = assumptions.get("sources", {})
    risk_inputs = assumptions.get("risk_inputs", {})
    risk_keys=("price_volatility_0_1","yield_variance_0_1","weather_risk_0_1")
    risk_score=None
    if all(k in risk_inputs and risk_inputs[k] not in (None,"") for k in risk_keys):
        price_volatility,yield_variance,weather_risk=(float(risk_inputs[k]) for k in risk_keys)
        if any(not 0 <= x <= 1 for x in (price_volatility,yield_variance,weather_risk)): raise ValueError("risk inputs must each be between 0 and 1")
        risk_score=round(100*(.35*price_volatility+.35*yield_variance+.30*weather_risk),1)
    return {
        "crop": crop, "area": area, "area_unit": unit, "area_hectares": hectares,
        "currency": "INR", "estimate": True,
        "costs_inr": {k: v * hectares for k, v in cost_per_ha.items()},
        "costs_per_hectare_inr": cost_per_ha,
        "costs_per_acre_inr": {k: v * ACRE_HECTARES for k, v in cost_per_ha.items()},
        "total_cost_inr": total_cost,
        "cost_per_kg_inr": total_per_ha / yield_kg_ha if yield_kg_ha else None,
        "cost_per_quintal_inr": (total_per_ha / yield_kg_ha * 100) if yield_kg_ha else None,
        "yield_kg": yield_kg_ha * hectares, "yield_range_kg": [yield_low * hectares, yield_high * hectares],
        "price_inr_per_kg": price_per_kg, "floor_price_inr_per_kg": price_floor,
        "revenue_inr": revenue, "gross_profit_inr": revenue - total_cost,
        "net_profit_inr": revenue - total_cost, "profit_per_acre_inr": (revenue - total_cost) / (hectares / ACRE_HECTARES),
        "profit_per_month_inr": (revenue - total_cost) / duration,
        "roi_pct": ((revenue - total_cost) / total_cost * 100) if total_cost else None,
        "benefit_cost_ratio": revenue / total_cost if total_cost else None,
        "break_even_yield_kg_per_ha": break_even_yield,
        "break_even_price_inr_per_kg": break_even_price,
        "scenarios": scenarios, "sensitivity": sensitivity,
        "risk_score": risk_score, "risk_method": "Heuristic weighted score (price volatility 35%, yield variance 35%, weather risk 30%); enter each 0–1 input. Score is omitted until all three are provided.", "risk_inputs": risk_inputs, "sources": sources,
        "provenance": {"area": "user input", "costs": sources.get("costs", "user input"), "yield": sources.get("yield", "user input"), "price": sources.get("price", "user input"), "floor_price": sources.get("floor_price", "user input"), "risk_score": "user risk inputs + heuristic calculation" if risk_score is not None else "not calculated; risk inputs missing"},
        "rotation_suggestion": assumptions.get("rotation_suggestion", "No sourced rotation suggestion available; verify crop rotation locally."),
        "note": "Estimates only; actual costs, yields and prices vary by location and season.",
    }
