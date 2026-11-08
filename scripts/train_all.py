"""Run only model training jobs whose reviewed input datasets are present."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
jobs=[('crop suitability',ROOT/'data/raw/crop_recommendation_kaggle/Crop_recommendation.csv',[sys.executable,'scripts/train_tabular.py','--task','crop','--csv','data/raw/crop_recommendation_kaggle/Crop_recommendation.csv','--target','label','--output','models/crop_latest.npz','--epochs','200']),('soil image (research-only dataset)',ROOT/'data/raw/soil_classification_nc/Orignal-Dataset',[sys.executable,'scripts/train_image.py','--task','soil','--data','data/raw/soil_classification_nc/Orignal-Dataset','--epochs','30','--augmentation-copies','3','--out','models/image_latest.npz']),('EuroSAT land cover',ROOT/'data/raw/eurosat_rgb/EuroSAT_RGB',[sys.executable,'scripts/train_image.py','--task','landcover','--data','data/raw/eurosat_rgb/EuroSAT_RGB','--epochs','12','--augmentation-copies','2','--out','models/landcover_latest.npz'])]
missing=[]
for name,source,cmd in jobs:
    if source.is_file() if source.suffix else source.is_dir():
        print(f'\nTraining {name} from reviewed local data...'); result=subprocess.run(cmd,cwd=ROOT); 
        if result.returncode: raise SystemExit(result.returncode)
    else: missing.append((name,source))
if missing:
    print('\nSkipped training; required datasets are not present:')
    for name,path in missing: print(f'- {name}: {path}')
    print('Follow the exact source and license steps in README.md and DATA_SOURCES.md. No sample data is generated.')

if (ROOT/'data/processed/faostat_india_crop_production.csv').exists() and (ROOT/'data/processed/faostat_india_producer_prices.csv').exists():
    print('\nPreparing annual forecast features from cleaned FAOSTAT data...')
    subprocess.run([sys.executable,'scripts/prepare_faostat_forecasts.py'],cwd=ROOT,check=True)
    jobs += [
        ('national yield trend',ROOT/'data/processed/faostat_india_yield_forecast.csv',[sys.executable,'scripts/train_tabular.py','--task','yield','--csv','data/processed/faostat_india_yield_forecast.csv','--date-column','date','--target','yield_kg_per_ha','--output','models/yield_latest.npz','--baseline-column','yield_persistence','--target-units','FAOSTAT national annual yield, kg/ha','--epochs','200']),
        ('producer price index trend',ROOT/'data/processed/faostat_india_price_index_forecast.csv',[sys.executable,'scripts/train_tabular.py','--task','price','--csv','data/processed/faostat_india_price_index_forecast.csv','--date-column','date','--group-columns','Item','--lag-periods','3','--target','producer_price_index','--output','models/price_latest.npz','--baseline-column','price_lag_1','--target-units','FAOSTAT annual producer price index (2014-2016=100); not INR/kg','--epochs','200']),
    ]
    for name,source,cmd in jobs[-2:]:
        if source.is_file():
            print(f'\nTraining {name} with chronological validation/test splits...')
            result=subprocess.run(cmd,cwd=ROOT)
            if result.returncode: raise SystemExit(result.returncode)
