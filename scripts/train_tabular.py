"""Train crop classification or yield/price regression MLPs from a reviewed CSV."""
from __future__ import annotations
import argparse, json
from datetime import date,datetime
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ml.crop.model import MLPClassifier
from importlib import import_module
MLPRegressor=import_module("ml.yield.model").MLPRegressor
from ml.eval.metrics import classification_metrics,regression_metrics

def split(n,seed=7):
    rng=np.random.default_rng(seed); order=rng.permutation(n); a=int(n*.7); b=int(n*.85); return order[:a],order[a:b],order[b:]

def stratified_split(y,seed=7):
    rng=np.random.default_rng(seed); parts=[[],[],[]]
    for label in np.unique(y):
        indices=rng.permutation(np.flatnonzero(np.asarray(y)==label)); n=len(indices); a=int(n*.7); b=int(n*.85)
        for dest,chunk in zip(parts,(indices[:a],indices[a:b],indices[b:])): dest.extend(chunk.tolist())
    return tuple(rng.permutation(np.asarray(part,dtype=int)) for part in parts)

def chronological_split(n):
    a=int(n*.7); b=int(n*.85); return np.arange(a),np.arange(a,b),np.arange(b,n)

def chronological_group_split(values):
    """Split whole timestamps so records from one year cannot straddle a split."""
    values=np.asarray(values); unique=np.unique(values); a=max(1,int(len(unique)*.70)); b=max(a+1,int(len(unique)*.85)); b=min(b,len(unique)-1)
    train=np.flatnonzero(np.isin(values,unique[:a])); val=np.flatnonzero(np.isin(values,unique[a:b])); test=np.flatnonzero(np.isin(values,unique[b:]))
    return train,val,test

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=["crop","yield","price"],required=True); ap.add_argument("--csv",required=True); ap.add_argument("--target",default=None); ap.add_argument("--output",default=None); ap.add_argument("--date-column",default="date"); ap.add_argument("--group-columns",default="crop,market"); ap.add_argument("--lag-periods",type=int,default=12); ap.add_argument("--baseline-column",default=None,help="optional persistence baseline column, excluded from model features"); ap.add_argument("--target-units",default=None); ap.add_argument("--epochs",type=int,default=100); args=ap.parse_args()
    if not args.output: args.output={"crop":"models/crop_latest.npz","yield":"models/yield_latest.npz","price":"models/price_latest.npz"}[args.task]
    df=pd.read_csv(args.csv); target=args.target or ("label" if args.task=="crop" else {"yield":"yield_kg_ha","price":"price_inr_per_kg"}[args.task])
    if target not in df: raise SystemExit(f"Missing target column {target}; inspect your source and set --target.")
    dates=None; price_features=[]
    if args.task=="price" or (args.task=="yield" and args.date_column in df.columns):
        if args.task=="price" and args.date_column not in df: raise SystemExit(f"Price training needs a date column named {args.date_column}; use --date-column to select it.")
        if args.date_column in df.columns:
            dates=pd.to_datetime(df[args.date_column],errors='coerce'); df=df.assign(__date=dates).dropna(subset=['__date']).sort_values('__date').reset_index(drop=True); dates=df['__date']
    if args.task=="price":
        if args.date_column not in df: raise SystemExit(f"Price training needs a date column named {args.date_column}; use --date-column to select it.")
        if "Year" in df.columns and "Year" != args.date_column: df=df.drop(columns=["Year"])
        df['month']=dates.dt.month; df['year']=dates.dt.year; df['month_sin']=np.sin(2*np.pi*df['month']/12); df['month_cos']=np.cos(2*np.pi*df['month']/12)
        groups=[c.strip() for c in args.group_columns.split(',') if c.strip() and c.strip() in df.columns]
        for lag in range(1,args.lag_periods+1):
            name=f'price_lag_{lag}'
            df[name]=df.groupby(groups,dropna=False)[target].shift(lag) if groups else df[target].shift(lag)
            price_features.append(name)
    yraw=df[target]; baseline_raw=pd.to_numeric(df[args.baseline_column],errors='coerce') if args.baseline_column else None
    Xdf=df.drop(columns=[target]+([args.baseline_column] if args.baseline_column else [])).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan); valid=yraw.notna() & Xdf.notna().all(axis=1)
    if baseline_raw is not None: valid &= baseline_raw.notna()
    time_values=df.loc[valid,'__date'].to_numpy() if '__date' in df else None
    if baseline_raw is not None: baseline_raw=baseline_raw.loc[valid].reset_index(drop=True)
    X=Xdf.loc[valid].to_numpy(float); yraw=yraw.loc[valid]
    if args.task!="crop":
        yraw=pd.to_numeric(yraw,errors="coerce"); finite=np.isfinite(yraw.to_numpy(float)); X=X[finite]; yraw=yraw.iloc[np.flatnonzero(finite)]
    # Split before computing any normalization values.
    if time_values is not None: time_values=time_values[np.isfinite(pd.to_numeric(yraw,errors='coerce').to_numpy(float))] if args.task!="crop" else time_values
    labels=yraw.astype(str).to_numpy() if args.task=="crop" else None
    if time_values is not None: it,iv,ie=chronological_group_split(time_values)
    else: it,iv,ie=(stratified_split(labels) if args.task=="crop" else split(len(X)))
    mean=X[it].mean(0); std=X[it].std(0); std[std==0]=1; X=(X-mean)/(std+1e-8)
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    if args.task=="crop":
        classes=sorted(yraw.astype(str).unique()); lookup={v:i for i,v in enumerate(classes)}; y=yraw.astype(str).map(lookup).to_numpy(int)
        model=MLPClassifier(X.shape[1],max(16,min(128,X.shape[1]*2)),classes); counts=np.bincount(y[it],minlength=len(classes)); weights=counts.sum()/np.maximum(counts,1); model.fit(X[it],y[it],epochs=args.epochs,class_weights=weights,validation_data=(X[iv],y[iv]) if len(iv) else None)
        def named_metrics(indices):
            raw=classification_metrics(y[indices],np.argmax(model.predict_proba(X[indices]),1),len(classes))
            per=raw.pop("classes")
            raw["per_class"]={name:per[i] for i,name in enumerate(classes)}
            raw["macro_precision"]=float(np.mean([m["precision"] for m in per]))
            raw["macro_recall"]=float(np.mean([m["recall"] for m in per]))
            raw["macro_f1"]=float(np.mean([m["f1"] for m in per]))
            return raw
        val_metrics=named_metrics(iv) if len(iv) else {}
        test_metrics=named_metrics(ie) if len(ie) else {}
        model.save(out,mean,std,Xdf.columns.tolist())
        extra={"classes":classes,"class_balance":{c:int((y==i).sum()) for c,i in lookup.items()},"class_balance_train":{c:int((y[it]==i).sum()) for c,i in lookup.items()}}
    else:
        y=yraw.to_numpy(float); ym=y[it].mean(); ys=y[it].std() or 1; yn=(y-ym)/ys
        model=MLPRegressor(X.shape[1]); model.fit(X[it],yn[it],epochs=args.epochs,validation_data=(X[iv],yn[iv]) if len(iv) else None); val_metrics=regression_metrics(y[iv],model.predict(X[iv])*ys+ym) if len(iv) else {}; test_metrics=regression_metrics(y[ie],model.predict(X[ie])*ys+ym) if len(ie) else {}
        np.savez(out,w1=model.layers[0].W,b1=model.layers[0].b,w2=model.layers[2].W,b2=model.layers[2].b,mean=mean,std=std,target_mean=ym,target_std=ys,features=np.array(Xdf.columns.tolist(),dtype=str),date_column=args.date_column if args.task=='price' else '',lag_periods=args.lag_periods if args.task=='price' else 0)
        extra={"target_units": args.target_units or "must be declared in input CSV and verified by user", "baseline_column": args.baseline_column}
        if baseline_raw is not None:
            if len(iv): extra["persistence_baseline_validation_metrics"]=regression_metrics(y[iv],baseline_raw.to_numpy(float)[iv])
            if len(ie): extra["persistence_baseline_test_metrics"]=regression_metrics(y[ie],baseline_raw.to_numpy(float)[ie])
            if len(iv): extra["validation_selected_predictor"]="MLP" if val_metrics["mae"] < extra["persistence_baseline_validation_metrics"]["mae"] else "persistence"
    train_curve=model.history if hasattr(model,'history') else []
    val_curve=model.validation_history
    overfit=bool(val_curve and val_curve[-1]>max(min(val_curve)*1.2,min(val_curve)+.1))
    split_method="chronological by date" if time_values is not None else ("stratified random" if args.task=='crop' else "seeded random")
    (out.parent/(out.stem+"_card.json")).write_text(json.dumps({"name":args.task,"trained_from_scratch":True,"data":str(Path(args.csv)),"target":target,"features":Xdf.columns.tolist(),"date":str(date.today()),"seed":7,"epochs":args.epochs,"epochs_trained":len(train_curve),"best_validation_epoch":model.best_validation_epoch,"early_stopping_patience":15,"split":{"train":len(it),"validation":len(iv),"test":len(ie),"method":split_method,"train_date_max":str(np.max(time_values[it])) if time_values is not None else None,"validation_date_min":str(np.min(time_values[iv])) if time_values is not None and len(iv) else None,"validation_date_max":str(np.max(time_values[iv])) if time_values is not None and len(iv) else None,"test_date_min":str(np.min(time_values[ie])) if time_values is not None and len(ie) else None},"validation_metrics":val_metrics,"test_metrics":test_metrics,"training_loss_curve":train_curve,"validation_loss_curve":val_curve,"overfit_flag":overfit,"preprocessing":"numeric columns only; train split statistics used for scaling","lag_periods":args.lag_periods if args.task=='price' else None,"warning":"Check the source, units, regional coverage, and dataset license before using predictions.",**extra},indent=2),encoding="utf-8")
    runs_dir=out.parent/'runs'; runs_dir.mkdir(parents=True,exist_ok=True)
    versioned=runs_dir/f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"; shutil.copy2(out,versioned); card=out.parent/(out.stem+"_card.json"); shutil.copy2(card,versioned.with_name(versioned.stem+"_card.json"))
    def concise(metrics):
        return {key:metrics[key] for key in ("accuracy","macro_f1","mae","rmse","mape_pct") if key in metrics}
    print(json.dumps({"artifact":str(out),"validation_metrics":concise(val_metrics),"test_metrics":concise(test_metrics),"persistence_baseline_validation_metrics":extra.get("persistence_baseline_validation_metrics"),"persistence_baseline_test_metrics":extra.get("persistence_baseline_test_metrics"),"validation_selected_predictor":extra.get("validation_selected_predictor"),"overfit_flag":overfit},indent=2))
if __name__=="__main__": main()
