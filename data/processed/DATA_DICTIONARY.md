# Processed data dictionary

`faostat_india_crop_production.csv` is derived from the downloaded FAOSTAT normalized production archive and filtered to India.

- `Area`: source country/area name (India).
- `Item`: source crop/item name; FAOSTAT item taxonomy retained.
- `Year`: FAOSTAT year.
- `area_ha`: harvested area converted to hectares from rows whose source unit is `ha`.
- `production_kg`: production converted from tonnes to kilograms (`t × 1,000`).
- `yield_kg_per_ha`: yield retained in kg/ha.
- `source_flags`: source observation flags retained where available.
- Rows with non-crop measures or units that cannot be converted from their source unit were excluded; unmatched metrics remain blank after pivoting. No imputation or learned preprocessing is applied. Source/license are recorded in DATA_SOURCES.md.


## `faostat_india_producer_prices.csv`

India annual producer-price records from the normalized FAOSTAT source; the original source units/currencies are retained and are not converted into mandi prices.

- `Area`, `Item`, `Year`: source keys.
- `Producer Price (LCU/tonne)`: local-currency units per tonne as reported.
- `Producer Price (SLC/tonne)`: standard local currency per tonne as reported.
- `Producer Price (USD/tonne)`: US dollars per tonne as reported.
- `Producer Price Index (2014-2016 = 100)`: source price-index value.
- `source_flags`: source observation flags when provided.


## Derived annual forecast feature tables (2026-11-08)

- `faostat_india_yield_forecast.csv`: 854 one-step-ahead observations from 14 India FAOSTAT crop series. `date`/`Year` identify the target year; `yield_lag_1..3` contain only previous observed annual yield in kg/ha; `yield_persistence` is the lag-1 baseline; `item__*` columns identify crop. The first three years per item are excluded.
- `faostat_india_price_index_forecast.csv`: 280 annual observations from eight FAOSTAT India producer-price item series. `producer_price_index` uses FAOSTAT's 2014-2016=100 index. It is a relative national producer-price index, not a currency price or mandi quote. The trainer creates three groupwise lags using previous years only.
- Both forecast tables are ordered by target date and evaluated with whole-year chronological partitions. Training means/standard deviations are fit on training rows only. `faostat_forecast_data_card.json` records row counts, years, items, units and limits.
