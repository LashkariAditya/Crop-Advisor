"""Train the shallow NumPy CNN on labeled class folders from scratch."""
import argparse,json,shutil
import sys
from datetime import date,datetime
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ml.image.cnn import SmallCNN,augment_image
from ml.eval.metrics import classification_metrics


def stratified_split(labels,seed=7):
    rng=np.random.default_rng(seed); train=[]; val=[]; test=[]
    for label in np.unique(labels):
        ix=rng.permutation(np.flatnonzero(labels==label)); a=int(.7*len(ix)); b=int(.85*len(ix)); train.extend(ix[:a]); val.extend(ix[a:b]); test.extend(ix[b:])
    return tuple(rng.permutation(x) for x in (train,val,test))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',required=True,help='directory containing one folder per label'); parser.add_argument('--task',choices=['soil','landcover'],default='soil'); parser.add_argument('--epochs',type=int,default=10); parser.add_argument('--image-size',type=int,default=32); parser.add_argument('--augmentation-copies',type=int,default=1,help='independent train-only augmented views per training image'); parser.add_argument('--out',default=None); args=parser.parse_args()
    if args.augmentation_copies < 1: raise SystemExit('--augmentation-copies must be at least 1')
    root=Path(args.data); classes=sorted(p.name for p in root.iterdir() if p.is_dir())
    if args.task=='soil' and {'AnnualCrop','Forest','River','SeaLake'}.issubset(set(classes)):
        raise SystemExit('This appears to be EuroSAT satellite data. Use --task landcover; never label it as soil type.')
    if len(classes)<2: raise SystemExit('Need at least two labeled class folders.')
    rows=[]
    for label,name in enumerate(classes):
        for file in (root/name).rglob('*'):
            if file.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                try:
                    with Image.open(file) as im: rows.append((np.asarray(im.convert('RGB').resize((args.image_size,args.image_size)),dtype=np.float32)/255,label))
                except OSError: continue
    if len(rows)<100: raise SystemExit('Too few readable images for a useful training run; provide the reviewed full dataset.')
    x=np.stack([r[0] for r in rows]); y=np.array([r[1] for r in rows]); train,val,test=stratified_split(y); rng=np.random.default_rng(7)
    # Augment only the train partition; validation/test images are kept untouched.
    augmented=[np.stack([augment_image(x[i],rng) for i in train]) for _ in range(args.augmentation_copies)]
    train_x=np.concatenate([x[train],*augmented]); train_y=np.concatenate([y[train]]+[y[train] for _ in augmented])
    counts=np.bincount(y[train],minlength=len(classes)); weights=counts.sum()/np.maximum(counts,1); model=SmallCNN(classes,image_size=args.image_size)
    history=model.fit(train_x,train_y,epochs=args.epochs,validation_data=(x[val],y[val]),class_weights=weights)
    val_metrics=classification_metrics(y[val],model.predict_proba(x[val]).argmax(1),len(classes)); test_metrics=classification_metrics(y[test],model.predict_proba(x[test]).argmax(1),len(classes))
    out=Path(args.out or ('models/landcover_latest.npz' if args.task=='landcover' else 'models/image_latest.npz')); out.parent.mkdir(parents=True,exist_ok=True); model.save(out)
    card={'name':'eurosat_landcover_cnn' if args.task=='landcover' else 'soil_photo_cnn','task':args.task,'trained_from_scratch':True,'dataset':str(root),'source_license':'EuroSAT project MIT license; Copernicus Sentinel terms also apply. See DATA_SOURCES.md.' if args.task=='landcover' else 'CC BY-NC 4.0 repository terms; underlying photo rights/provenance may vary. See DATA_SOURCES.md.','public_distribution':'Attribution/terms review required.' if args.task=='landcover' else 'Excluded from the public image; research/education use only pending rights review.','classes':classes,'class_balance':{c:int((y==i).sum()) for i,c in enumerate(classes)},'image_size':[args.image_size,args.image_size],'epochs_requested':args.epochs,'epochs_trained':len(history),'best_validation_epoch':model.best_validation_epoch,'early_stopping_patience':15,'date':str(date.today()),'seed':7,'split_counts':{'train':len(train),'validation':len(val),'test':len(test)},'augmentation':'random flips, right-angle rotations, crop and brightness applied only to training partition','augmentation_copies':args.augmentation_copies,'validation_metrics':val_metrics,'test_metrics':test_metrics,'training_loss_curve':history,'validation_loss_curve':model.validation_history,'overfit_flag':bool(model.validation_history and model.validation_history[-1]>max(min(model.validation_history)*1.2,min(model.validation_history)+.1)),'warning':'EuroSAT outputs apply only to satellite land-cover imagery; they are not soil type, moisture or vegetation measurements.' if args.task=='landcover' else 'Verify the dataset license, image context, regional coverage and class labels before field use.'}
    card_path=out.with_name(out.stem+'_card.json'); card_path.write_text(json.dumps(card,indent=2),encoding='utf-8')
    runs_dir=out.parent/'runs'; runs_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); versioned=runs_dir/f'{out.stem.replace("_latest","")}_{stamp}.npz'; shutil.copy2(out,versioned); shutil.copy2(card_path,versioned.with_name(versioned.stem+'_card.json'))
    print(json.dumps({'artifact':str(out),'versioned_artifact':str(versioned),'validation_metrics':val_metrics,'test_metrics':test_metrics,'overfit_flag':card['overfit_flag']},indent=2))

if __name__=='__main__': main()
