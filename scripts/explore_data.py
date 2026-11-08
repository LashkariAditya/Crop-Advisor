"""Create a factual data exploration report from downloaded and processed files."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'data/raw/MANIFEST.json'
report={'sources':[{k:v for k,v in row.items() if k!='sample_files'} for row in json.loads(manifest.read_text(encoding='utf-8'))] if manifest.exists() else [],'datasets':{}}
for name,rel in [('faostat_india_crop_production','data/processed/faostat_india_crop_production.csv'),('faostat_india_producer_prices','data/processed/faostat_india_producer_prices.csv')]:
    p=ROOT/rel
    if p.exists():
        d=pd.read_csv(p); item=d['Item'] if 'Item' in d else pd.Series([],dtype=str); report['datasets'][name]={'rows':len(d),'columns':list(d.columns),'missing_values':{k:int(v) for k,v in d.isna().sum().items()},'items':int(item.nunique()),'year_min':int(d.Year.min()),'year_max':int(d.Year.max()),'item_row_counts_top_30':{str(k):int(v) for k,v in item.value_counts().head(30).items()}}
euro=ROOT/'data/raw/eurosat_rgb/EuroSAT_RGB'
if euro.exists(): report['datasets']['eurosat_rgb']={'classes':{p.name:sum(1 for f in p.glob('*.jpg')) for p in euro.iterdir() if p.is_dir()}}
(ROOT/'data/processed/exploration_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Data exploration report','', 'Generated from downloaded, licensed source archives; figures are descriptive only. No dataset was used to train a model in this report.','']
for name,meta in report['datasets'].items():
    lines += [f'## {name}','',f'- Details: `{json.dumps(meta,ensure_ascii=False)}`','']
(ROOT/'data/processed/exploration_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(f"Wrote {ROOT/'data/processed/exploration_report.md'}")
