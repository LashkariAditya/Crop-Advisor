# Dataset sources

Archives were checked and downloaded on 2026-11-07 and 2026-11-08 (UTC timestamps and SHA-256 digests or pinned source commit are recorded in `data/raw/MANIFEST.json`). All downloaded data are retained under `data/raw/`; generated processed data are under `data/processed/`. The raw directory is git-ignored due to file size.

| Name | URL | License / terms | Download size and structure | Download date | Use |
|---|---|---|---|---|---|
| EuroSAT RGB | https://zenodo.org/records/7711810 | MIT per [EuroSAT project repository](https://github.com/phelber/EuroSAT); Copernicus Sentinel terms also apply | 94,658,721-byte ZIP; 27,000 RGB JPEGs; 10 land-cover classes (class counts are in `data/processed/exploration_report.md`) | 2026-11-07 12:55 UTC | Land-cover helper only. Never treat satellite-scene labels as soil type. |
| FAOSTAT crop production | https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip | CC BY 4.0 per FAO statistical database policy; FAOSTAT statistical database terms apply | 33,921,825-byte ZIP; normalized CSV has 4,209,110 records and columns: Area Code, Area Code (M49), Area, Item Code, Item Code (CPC), Item, Element Code, Element, Year Code, Year, Unit, Value, Flag, Note | 2026-11-07 12:55 UTC | Historical production, harvested area and yield. Cleaned India metrics in `data/processed/faostat_india_crop_production.csv`; convertible units standardized and source flags retained. |
| FAOSTAT producer prices | https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip | CC BY 4.0 per FAO statistical database policy; FAOSTAT statistical database terms apply | 11,684,078-byte ZIP; normalized CSV has 1,319,583 records and columns: Area Code, Area Code (M49), Area, Item Code, Item Code (CPC), Item, Element Code, Element, Year Code, Year, Months Code, Months, Unit, Value, Flag | 2026-11-07 12:56 UTC | Producer price history; source unit retained because currencies/units require conversion metadata before comparisons. |
| Soil Classification Dataset (original images) and CyAUG-Dataset | https://github.com/Phantom-fs/Soil-Classification-Dataset | CC BY-NC 4.0 per repository `LICENCE`; non-commercial research/education only; cite associated paper; underlying image rights/provenance may vary | 1,189 original photos (Alluvial 52, Arid 284, Black 255, Laterite 219, Mountain 201, Red 109, Yellow 69), pinned commit `c3108e9ea8277a2762b0aad587e90cc9722e63d4`; CyAUG archive audited locally, 3,909 non-exact generated images after exact copies were excluded from the experiment | 2026-11-08 | The currently selected app model uses original images only. A separate CyAUG training experiment appended generated files to training only and did not improve validation; its candidate is not selected. Generated samples have incomplete source-photo lineage and are not independent photographs. Neither output is a soil test or moisture measurement. Academic/non-commercial use only. |
| Soil core images | https://huggingface.co/datasets/grano1/core_sample_image_data | MIT per dataset card; cite Granitzer et al. (2025) | 3,604 image rows in Parquet train/validation/test shards (192,437,383 bytes total); each shard verified against SHA-256 in `data/raw/soil_core_hf/manifest.json` | 2026-11-08 | Separate geotechnical main-fraction classification task; grouped by borehole identifier before splitting to avoid repeated-borehole leakage. Labels are not Indian agricultural soil classes. |
| Crop Recommendation Dataset | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset | Apache 2.0 per Kaggle v1 dataset metadata, checked 2026-11-08 | 65,234-byte ZIP; `Crop_recommendation.csv` is 150,034 bytes, 2,200 rows, 8 columns, 22 classes with 100 rows each; SHA-256 in manifest | 2026-11-08 | Benchmark-only crop suitability classifier. The Kaggle description says the table was built by augmenting rainfall, climate and fertilizer datasets; treat the held-out score as same-benchmark performance, not field generalization. |
| Crop Production in India | https://www.kaggle.com/datasets/abhinand05/crop-production-in-india | Review exact license/terms on Kaggle dataset page before use | State, district, crop, season, area and production; exact archive size/schema not verified | Not downloaded | Candidate state/district production dataset; disabled until reviewed and manually downloaded. |
| PlantVillage (optional) | https://github.com/spMohanty/PlantVillage-Dataset | No clear data license found in linked GitHub repository; excluded pending confirmation | 54,306 images and 38 classes stated by repository description; archive size not verified | Not downloaded | Not used. |
| FAOSTAT annual forecast tables | Derived from the two FAOSTAT archives above | CC BY 4.0 source terms retained; generated feature tables are derived data | `data/processed/faostat_india_yield_forecast.csv` (854 observations, 14 crop series, 1964–2024); `faostat_india_price_index_forecast.csv` (280 observations, 8 item series, 1991–2025) | 2026-11-08 | One-step annual research benchmarks. Yield is India-level kg/ha; price target is producer-price index 2014–2016=100, not rupees/kg or mandi retail price. Time split holds out whole years. |

The FAO states that its statistical databases are governed by its open data licensing policy; FAOSTAT statistical catalogue pages specify CC BY 4.0. Review the current terms before redistribution or commercial use. The FAOSTAT provider may replace bulk archives in place.

## Runtime data APIs (not downloaded datasets)

- Open-Meteo forecast: https://open-meteo.com/en/docs
- Open-Meteo historical weather archive: https://archive-api.open-meteo.com/v1/archive
- NASA POWER: https://power.larc.nasa.gov/docs/services/api/
- ISRIC SoilGrids: https://rest.isric.org/soilgrids/v2.0/docs (beta REST API; fair-use limits apply)
- India mandi prices: https://www.data.gov.in/ (resource ID/API key required; local `.env` only)

API results are cached locally and identify their source where available. APIs can be unavailable or change their terms and formats.


Downloaded crop_recommendation_kaggle on 2026-11-08T06:24:23.585389+00:00; 2 files, 215268 bytes under `data\raw\crop_recommendation_kaggle`.

Downloaded soil_classification_nc on 2026-11-08T06:51:15.034215+00:00; 1191 files, 237849178 bytes under `data\raw\soil_classification_nc`.
