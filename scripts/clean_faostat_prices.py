"""Clean India FAOSTAT annual producer prices without inventing FX conversions."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/raw/faostat_producer_prices/Prices_E_All_Data_(Normalized).csv'
if not source.exists(): raise SystemExit(f"Dataset missing: {source}. Run make data first.")
df=pd.read_csv(source,encoding='latin-1',low_memory=False)
df=df[df['Area'].eq('India')].copy()
df=df[df['Element'].isin(['Producer Price (LCU/tonne)','Producer Price (SLC/tonne)','Producer Price (USD/tonne)','Producer Price Index (2014-2016 = 100)'])]
keys=['Area','Item','Year']
wide=df.pivot_table(index=keys,columns='Element',values='Value',aggfunc='first').reset_index()
flags=df.groupby(keys)['Flag'].agg(lambda x:';'.join(sorted(set(str(v) for v in x if pd.notna(v))))).reset_index(name='source_flags')
wide=wide.merge(flags,on=keys,how='left').sort_values(['Item','Year'])
out=ROOT/'data/processed/faostat_india_producer_prices.csv'; out.parent.mkdir(parents=True,exist_ok=True); wide.to_csv(out,index=False)
with (ROOT/'data/processed/DATA_DICTIONARY.md').open('a',encoding='utf-8') as f:
 f.write('''\n\n## `faostat_india_producer_prices.csv`\n\nIndia annual producer-price records from the normalized FAOSTAT source; the original source units/currencies are retained and are not converted into mandi prices.\n\n- `Area`, `Item`, `Year`: source keys.\n- `Producer Price (LCU/tonne)`: local-currency units per tonne as reported.\n- `Producer Price (SLC/tonne)`: standard local currency per tonne as reported.\n- `Producer Price (USD/tonne)`: US dollars per tonne as reported.\n- `Producer Price Index (2014-2016 = 100)`: source price-index value.\n- `source_flags`: source observation flags when provided.\n''')
print(f'Wrote {len(wide)} India item/year price rows to {out}')
