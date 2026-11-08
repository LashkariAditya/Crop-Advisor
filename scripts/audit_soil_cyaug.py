"""Audit CyAUG near-duplicates against the seeded original-photo split."""
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ORIG=ROOT/'data/raw/soil_classification_nc/Orignal-Dataset'
AUG=ROOT/'data/raw/soil_classification_nc/cyaug_source'

def split(labels,seed=7):
    rng=np.random.default_rng(seed);parts=[[],[],[]]
    for label in np.unique(labels):
        ix=rng.permutation(np.flatnonzero(labels==label));n=len(ix);a=int(.7*n);b=int(.85*n)
        for dst,chunk in zip(parts,(ix[:a],ix[a:b],ix[b:])):dst.extend(chunk.tolist())
    return tuple(rng.permutation(x) for x in parts)

def phash(path):
    gray=np.asarray(Image.open(path).convert('L').resize((32,32)),dtype=np.float32)
    # Orthonormal 2-D DCT, then the low-frequency 8 x 8 signature.
    n=32;k=np.arange(n);freq=np.arange(n)[:,None]
    mat=np.cos(np.pi*(2*k+1)*freq/(2*n)).astype(np.float32)
    mat[0]*=1/np.sqrt(n);mat[1:]*=np.sqrt(2/n)
    low=(mat[:8]@gray@mat[:8].T).ravel()[1:]
    median=np.median(low)
    bits=low>median
    value=0
    for b in bits:value=(value<<1)|int(b)
    return value

def main():
    classes=sorted(p.name for p in ORIG.iterdir() if p.is_dir())
    rows=[]
    for c in classes:
        for f in (ORIG/c).rglob('*'):
            if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:rows.append((f,c))
    y=np.array([c for _,c in rows]);tr,va,te=split(y)
    role={rows[i][0]:'train' for i in tr}|{rows[i][0]:'validation' for i in va}|{rows[i][0]:'test' for i in te}
    report={'distance_threshold_bits':6,'per_class':{},'original_split':{'train':len(tr),'validation':len(va),'test':len(te)}}
    accepted=[]
    for c in classes:
        originals=[(p,role[p],phash(p),hashlib.sha256(p.read_bytes()).digest()) for p,cl in rows if cl==c]
        augfiles=[p for p in (AUG/c).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
        summary={'augmentation_images':len(augfiles),'exact_original_copies':0,'near_original_train':0,'near_original_validation':0,'near_original_test':0,'near_original_ambiguous':0,'unmatched_or_low_similarity':0,'kept_for_training':0}
        for p in augfiles:
            rawhash=hashlib.sha256(p.read_bytes()).digest()
            exact=next((r for r in originals if r[3]==rawhash),None)
            if exact is not None:summary['exact_original_copies']+=1;continue
            h=phash(p); ds=[((h^r[2]).bit_count(),r[1]) for r in originals]; md=min(d for d,_ in ds)
            nearest={split for d,split in ds if d==md}
            if md<=6 and len(nearest)>1:
                summary['near_original_ambiguous']+=1
            elif md<=6:
                r=next(iter(nearest));summary['near_original_'+r]+=1
                if r=='train':summary['kept_for_training']+=1;accepted.append(str(p.relative_to(ROOT)))
            else:
                summary['unmatched_or_low_similarity']+=1
                # Unknown provenance: omit from training for a conservative split.
        report['per_class'][c]=summary
    report['kept_images']=len(accepted);report['kept_paths']=accepted
    out=ROOT/'data/processed/soil_cyaug_audit.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'original_split':report['original_split'],'classes':report['per_class'],'kept_images':len(accepted),'report':str(out)},indent=2))

if __name__=='__main__':main()
