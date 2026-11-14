"""Write a compact offline evaluation report from predictions and labels CSV."""
import argparse,json,html
from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ml.eval.metrics import classification_metrics,regression_metrics
ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--true'); ap.add_argument('--pred'); ap.add_argument('--task',choices=['classification','regression']); ap.add_argument('--out',default='reports/evaluation.json'); a=ap.parse_args(); p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
if a.csv:
    if not all((a.true,a.pred,a.task)): ap.error('--csv requires --true, --pred, and --task')
    df=pd.read_csv(a.csv); f=classification_metrics if a.task=='classification' else regression_metrics; result=f(df[a.true].to_numpy(),df[a.pred].to_numpy())
else:
    result={'model_cards':[]}
    current_cards=[Path('models')/name for name in ('crop_latest_card.json','image_latest_card.json','soil_transfer_latest_card.json','soil_core_latest_card.json','landcover_latest_card.json','yield_latest_card.json','price_latest_card.json')]
    for card in current_cards:
        try:
            raw=json.loads(card.read_text(encoding='utf-8'))
            def brief_metrics(metrics):
                if not isinstance(metrics,dict): return metrics
                keep={key:value for key,value in metrics.items() if key in {'accuracy','macro_f1','macro_precision','macro_recall','mae','rmse','mape_pct'}}
                if 'classes' in metrics and isinstance(metrics['classes'],list):
                    keep['per_class']=[{k:v for k,v in row.items() if k in {'precision','recall','f1'}} for row in metrics['classes']]
                if 'per_class' in metrics and isinstance(metrics['per_class'],dict): keep['per_class']=metrics['per_class']
                return keep
            result['model_cards'].append({
                'card':card.name,'name':raw.get('name'),'task':raw.get('task'),
                'data':raw.get('data',raw.get('dataset')),'date':raw.get('date'),
                'split':raw.get('split',raw.get('split_counts')),'features':raw.get('features'),
                'target_units':raw.get('target_units'),'validation_metrics':brief_metrics(raw.get('validation_metrics')),
                'test_metrics':brief_metrics(raw.get('test_metrics')),
                'persistence_baseline_test_metrics':brief_metrics(raw.get('persistence_baseline_test_metrics')),
                'validation_selected_predictor':raw.get('validation_selected_predictor'),
                'warning':raw.get('warning')
            })
        except (OSError,json.JSONDecodeError):pass
p.write_text(json.dumps(result,indent=2),encoding='utf-8')
rows=''.join(f'<tr><td>{html.escape(str(k))}</td><td><pre>{html.escape(json.dumps(v))}</pre></td></tr>' for k,v in result.items())
(p.with_suffix('.html')).write_text(f'<!doctype html><meta charset="utf-8"><title>CropAdvisor evaluation</title><h1>Evaluation report</h1><p>Model cards are source data and independently reported metrics. No new predictions are fabricated by this report.</p><table border="1"><tr><th>Metric</th><th>Value</th></tr>{rows}</table>',encoding='utf-8')
print(json.dumps(result,indent=2))
