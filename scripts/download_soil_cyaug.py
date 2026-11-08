"""Fetch the repository's supplemental CyAUG soil-photo set."""
from __future__ import annotations
import hashlib, json, urllib.request, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/raw/soil_classification_nc/cyaug_source'
ARCHIVE=ROOT/'data/raw/soil_classification_nc/cyaug_source.zip'
URL='https://github.com/Phantom-fs/Soil-Classification-Dataset/archive/refs/heads/main.zip'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    if not ARCHIVE.exists():
        req=urllib.request.Request(URL,headers={'User-Agent':'CropAdvisor-educational-research'})
        digest=hashlib.sha256();total=0
        with urllib.request.urlopen(req,timeout=60) as response, ARCHIVE.open('wb') as out:
            while chunk:=response.read(1024*1024):
                out.write(chunk);digest.update(chunk);total+=len(chunk)
                if total%(25*1024*1024)<1024*1024:print(f'{total/1024/1024:.0f} MiB downloaded',flush=True)
        print(f'Archive: {total:,} bytes, SHA-256 {digest.hexdigest()}',flush=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        matches=[n for n in z.namelist() if '/CyAUG-Dataset/' in n and not n.endswith('/')]
        if len(matches)<4000:raise RuntimeError(f'Expected at least 4000 CyAUG images, found {len(matches)}')
        for name in matches:
            relative=name.split('/CyAUG-Dataset/',1)[1]
            target=DEST/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with z.open(name) as src,target.open('wb') as dst:dst.write(src.read())
        counts={p.name:sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}) for p in DEST.iterdir() if p.is_dir()}
    card={'source':'https://github.com/Phantom-fs/Soil-Classification-Dataset','commit':'c3108e9ea8277a2762b0aad587e90cc9722e63d4','license':'CC BY-NC 4.0 repository; underlying image rights/provenance may vary','archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),'image_counts':counts,'note':'CycleGAN-augmented images derived from the original collection; generated views are not additional independent soil observations.'}
    (DEST/'download_manifest.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    print(json.dumps(card,indent=2))

if __name__=='__main__':main()
