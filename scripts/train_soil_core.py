"""Train a separate soil-core fraction model with borehole-group splits.

This dataset has geotechnical labels and must not replace field-soil labels.
Its image IDs include multiple views/intervals from the same borehole; all
records from a borehole are assigned to the same split.
"""
import json, random
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

ROOT=Path(__file__).resolve().parents[1]
LABELS=['fS','mS','U','gS']

class Rows(Dataset):
    def __init__(self, frame, lookup, train=False):
        self.frame=frame.reset_index(drop=True); self.lookup=lookup
        self.tf=transforms.Compose(([transforms.RandomResizedCrop(160,scale=(.75,1.0)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(15)] if train else [transforms.Resize(184),transforms.CenterCrop(160)])+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        image=Image.open(BytesIO(row.image['bytes'])).convert('RGB')
        return self.tf(image),self.lookup[row.HB]

def grouped_split(frame,seed=17):
    groups=sorted(frame.well.unique()); rng=np.random.default_rng(seed)
    counts=frame.groupby(['well','HB']).size().unstack(fill_value=0).reindex(index=groups,columns=LABELS,fill_value=0).to_numpy(float)
    total=counts.sum(0); best=None
    # Keep entire boreholes together. Search many 10/2/3 group assignments,
    # choosing validation/test proportions with useful coverage of each class.
    for _ in range(30000):
        order=rng.permutation(len(groups)); a,b=order[:10],order[10:12]; c=order[12:]
        parts=[counts[a].sum(0),counts[b].sum(0),counts[c].sum(0)]
        sizes=[x.sum() for x in parts]
        score=sum(((x/np.maximum(total,1)-frac)**2).sum() for x,frac in zip(parts,[.70,.15,.15]))
        score+=sum(1.0 for x in parts for v in x if v==0)
        score+=sum((s/frame.shape[0]-frac)**2 for s,frac in zip(sizes,[.70,.15,.15]))
        if best is None or score<best[0]: best=(score,set(groups[i] for i in a),set(groups[i] for i in b),set(groups[i] for i in c))
    _,train_g,val_g,test_g=best
    return tuple(frame[frame.well.isin(g)].copy() for g in (train_g,val_g,test_g)), {'train':sorted(train_g),'validation':sorted(val_g),'test':sorted(test_g)}

def epoch(model,loader,device,optimizer=None,criterion=None):
    train=optimizer is not None; model.train(train); total=correct=0; ys=[]; ps=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            z=model(x); loss=criterion(z,y)
            if train: loss.backward(); optimizer.step()
        p=z.argmax(1);total+=len(y);correct+=int((p==y).sum());ys.extend(y.cpu().tolist());ps.extend(p.cpu().tolist())
    return correct/max(total,1),np.array(ys),np.array(ps)

def score(y,p,n):
    cm=np.zeros((n,n),int);np.add.at(cm,(y,p),1);pr=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(n),where=cm.sum(0)>0);re=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(n),where=cm.sum(1)>0);f=np.divide(2*pr*re,pr+re,out=np.zeros(n),where=pr+re>0)
    return {'accuracy':float(cm.trace()/cm.sum()),'macro_f1':float(f.mean()),'per_class':{LABELS[i]:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f[i]),'support':int(cm[i].sum())} for i in range(n)}}

def main():
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(min(4,torch.get_num_threads()))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames=[pd.read_parquet(ROOT/f'data/raw/soil_core_hf/{s}-00000-of-00001.parquet') for s in ('train','val','test')]
    frame=pd.concat(frames,ignore_index=True); frame=frame[frame.HB.isin(LABELS)].copy()
    frame['well']=frame.Bildname.astype(str).str.extract(r'^(BKF_\d+[a-z]?_\d+)')[0]
    if frame.well.isna().any(): raise ValueError('Could not recover borehole group from every image identifier.')
    (tr,va,te),groups=grouped_split(frame)
    lookup={x:i for i,x in enumerate(LABELS)}
    train_dl=DataLoader(Rows(tr,lookup,True),batch_size=32,shuffle=True,num_workers=0)
    val_dl=DataLoader(Rows(va,lookup),batch_size=32,shuffle=False,num_workers=0)
    test_dl=DataLoader(Rows(te,lookup),batch_size=32,shuffle=False,num_workers=0)
    weights=ResNet18_Weights.DEFAULT; model=models.resnet18(weights=weights);model.fc=nn.Linear(model.fc.in_features,len(LABELS));model.to(device)
    for p in model.parameters():p.requires_grad=False
    for p in model.fc.parameters():p.requires_grad=True
    counts=np.array([int((tr.HB==c).sum()) for c in LABELS]); cw=torch.tensor(counts.sum()/(len(LABELS)*counts),dtype=torch.float32,device=device);criterion=nn.CrossEntropyLoss(weight=cw)
    optimizer=torch.optim.AdamW(model.fc.parameters(),lr=4e-4,weight_decay=1e-3)
    history=[];best=(-1.,-1.);best_state=None;best_epoch=0;wait=0
    for e in range(12):
        if e==3:
            for p in model.layer4.parameters():p.requires_grad=True
            optimizer=torch.optim.AdamW([{'params':model.layer4.parameters(),'lr':1e-5},{'params':model.fc.parameters(),'lr':1e-4}],weight_decay=1e-3)
        train_acc,_,_=epoch(model,train_dl,device,optimizer,criterion)
        _,y,p=epoch(model,val_dl,device,criterion=criterion); m=score(y,p,len(LABELS)); key=(m['macro_f1'],m['accuracy'])
        history.append({'epoch':e+1,'train_accuracy':train_acc,'validation':m});print(f"epoch {e+1:02d}/12 train={train_acc:.3f} val={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}",flush=True)
        if key>best:best=key;best_epoch=e+1;wait=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=5:break
    model.load_state_dict(best_state);_,y,p=epoch(model,test_dl,device,criterion=criterion);test=score(y,p,len(LABELS))
    artifact=ROOT/'models/soil_core_latest.pt';torch.save({'state_dict':best_state,'classes':LABELS,'architecture':'resnet18_imagenet_finetuned','input_size':160},artifact)
    card={'name':'soil_core_fraction_resnet18','task':'geotechnical soil-core main fraction','source':'Hugging Face grano1/core_sample_image_data','source_license':'MIT dataset card; cite Granitzer et al. (3FOMLIG 2025)','dataset_rows_after_label_filter':len(frame),'labels':LABELS,'excluded_labels':'Rare classes were excluded before splitting: fS+mS, fS+U, S, G, fG; those labels are not merged into pure fractions.','split_method':'Group by borehole identifier (BKF_nn_yy), 70/15/15 approximate, seed 17; all splits drawn from source official train/val/test then regrouped to prevent same-borehole leakage.','split_boreholes':groups,'split_counts':{'train':len(tr),'validation':len(va),'test':len(te)},'device':str(device),'pretrained':'torchvision ResNet18 ImageNet1K_V1','epochs_trained':len(history),'best_validation_epoch':best_epoch,'validation_metrics':history[best_epoch-1]['validation'],'test_metrics':test,'history':history,'warning':'This predicts geotechnical core fractions, not agricultural soil types; do not use as the CropAdvisor field-soil classifier.'}
    (ROOT/'models/soil_core_latest_card.json').write_text(json.dumps(card,indent=2),encoding='utf-8');print(json.dumps({'artifact':str(artifact),'validation':card['validation_metrics'],'test':test,'splits':card['split_counts'],'boreholes':groups},indent=2),flush=True)

if __name__=='__main__':main()
