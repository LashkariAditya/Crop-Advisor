"""Download the MIT-licensed Hugging Face soil-core image dataset."""
import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/soil_core_hf'
FILES = {
    'train-00000-of-00001.parquet': (153_935_035, '8e9b82f82f1eeca0d42b515ef6bda400908c4a214ce11e0414e4fe40834a708f'),
    'val-00000-of-00001.parquet': (19_226_419, '42f77799bf05601aa029692ee5548e38675177a9394c7fbd2f2fa5e60e918d12'),
    'test-00000-of-00001.parquet': (19_274_929, '7a4a6e0c1a1bcddbad798e891992980357b51311fe9868db8a5607e31d8f3d38'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, expected in FILES.items():
        path = OUT / name
        if not path.exists():
            url = f'https://huggingface.co/datasets/grano1/core_sample_image_data/resolve/main/data/{name}?download=true'
            response = requests.get(url, stream=True, timeout=(30, 180))
            response.raise_for_status()
            digest, size = hashlib.sha256(), 0
            temp = path.with_suffix('.part')
            with temp.open('wb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk); digest.update(chunk); size += len(chunk)
                        if size % (20 * 1024 * 1024) < 1024 * 1024:
                            print(f'{name}: {size / 1024 / 1024:.0f} MiB', flush=True)
            if expected and (size != expected[0] or digest.hexdigest() != expected[1]):
                temp.unlink(missing_ok=True)
                raise RuntimeError(f'Checksum/size mismatch for {name}: {size} bytes, SHA-256 {digest.hexdigest()}')
            temp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[name] = {'bytes': path.stat().st_size, 'sha256': digest}
        print(f'Verified {name}: {path.stat().st_size:,} bytes SHA-256 {digest}', flush=True)
    import json
    (OUT / 'manifest.json').write_text(json.dumps({'source': 'https://huggingface.co/datasets/grano1/core_sample_image_data', 'license': 'MIT (dataset card)', 'files': manifest}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
