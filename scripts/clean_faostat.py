"""Standardize Indian FAOSTAT crop area, output, and yield units."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/raw/faostat_crop_production/Production_Crops_Livestock_E_All_Data_(Normalized).csv'
if not source.exists(): raise SystemExit(f"Dataset missing: {source}. Run make data first.")
df=pd.read_csv(source,encoding='latin-1',low_memory=False)
df=df[df['Area'].eq('India') & df['Element'].isin(['Area harvested','Production','Yield'])].copy()
# Normalize only compatible crop metrics. Unsupported animal counts and number units are omitted.
df['normalized_metric']=df['Element'].map({'Area harvested':'area_ha','Production':'production_kg','Yield':'yield_kg_per_ha'})
df['normalized_value']=pd.NA
df.loc[(df.Element=='Area harvested')&(df.Unit=='ha'),'normalized_value']=df.loc[(df.Element=='Area harvested')&(df.Unit=='ha'),'Value']
df.loc[(df.Element=='Production')&(df.Unit=='t'),'normalized_value']=df.loc[(df.Element=='Production')&(df.Unit=='t'),'Value']*1000
df.loc[(df.Element=='Yield')&(df.Unit=='kg/ha'),'normalized_value']=df.loc[(df.Element=='Yield')&(df.Unit=='kg/ha'),'Value']
df=df.dropna(subset=['normalized_value'])
keys=['Area','Item','Year']
wide=df.pivot_table(index=keys,columns='normalized_metric',values='normalized_value',aggfunc='first').reset_index()
flags=df.groupby(keys)['Flag'].agg(lambda x:';'.join(sorted(set(str(v) for v in x if pd.notna(v))))).reset_index(name='source_flags')
wide=wide.merge(flags,on=keys,how='left').sort_values(['Item','Year'])
out=ROOT/'data/processed/faostat_india_crop_production.csv'; out.parent.mkdir(parents=True,exist_ok=True); wide.to_csv(out,index=False)
(ROOT/'data/processed/DATA_DICTIONARY.md').write_text('''# Processed data dictionary\n\n`faostat_india_crop_production.csv` is derived from the downloaded FAOSTAT normalized production archive and filtered to India.\n\n- `Area`: source country/area name (India).\n- `Item`: source crop/item name; FAOSTAT item taxonomy retained.\n- `Year`: FAOSTAT year.\n- `area_ha`: harvested area converted to hectares from rows whose source unit is `ha`.\n- `production_kg`: production converted from tonnes to kilograms (`t × 1,000`).\n- `yield_kg_per_ha`: yield retained in kg/ha.\n- `source_flags`: source observation flags retained where available.\n- Rows with non-crop measures or units that cannot be converted from their source unit were excluded; unmatched metrics remain blank after pivoting. No imputation or learned preprocessing is applied. Source/license are recorded in DATA_SOURCES.md.\n''',encoding='utf-8')
print(f'Wrote {len(wide)} crop/year rows to {out}')
