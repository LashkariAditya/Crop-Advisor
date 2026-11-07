"""Download configured public datasets with retries, resume, and a manifest."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "datasets.yaml"
MANIFEST = ROOT / "data" / "raw" / "MANIFEST.json"
SOURCES = ROOT / "DATA_SOURCES.md"
CHUNK = 1024 * 256


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def manual(name: str, reason: str) -> None:
    print(f"\n{ name }: {reason}")
    print("Manual steps: open config/datasets.yaml, confirm source and license, download the file from its listed URL, then place it in the dataset destination directory. Do not use it until DATA_SOURCES.md is updated.")


def download(entry: dict) -> dict | None:
    name, url = entry["name"], entry["url"]
    destination = ROOT / entry["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    filename = entry["expected_files"][0] if entry.get("expected_files") else Path(urlparse(url).path).name
    target = destination / filename
    partial = target.with_suffix(target.suffix + ".part")
    expected_checksum = entry.get("checksum")
    algorithm = entry.get("checksum_algorithm", "sha256")

    if target.exists():
        if expected_checksum and digest(target, algorithm).lower() != expected_checksum.lower():
            manual(name, "existing file checksum does not match; moved aside as .invalid")
            target.replace(target.with_suffix(target.suffix + ".invalid"))
        else:
            print(f"Skip {name}: file already exists ({target.stat().st_size} bytes)")
            return None

    last_error = "unknown error"
    for attempt in range(1, 4):
        try:
            headers = {"Range": f"bytes={partial.stat().st_size}-"} if partial.exists() else {}
            with requests.get(url, stream=True, allow_redirects=True, timeout=(20, 90), headers=headers) as response:
                response.raise_for_status()
                append = response.status_code == 206 and partial.exists()
                if partial.exists() and not append:
                    partial.unlink()
                total = int(response.headers.get("Content-Length", 0)) + (partial.stat().st_size if append else 0)
                mode = "ab" if append else "wb"
                with partial.open(mode) as out, tqdm(total=total or None, initial=partial.stat().st_size if append else 0, unit="B", unit_scale=True, desc=name) as bar:
                    for block in response.iter_content(CHUNK):
                        if block:
                            out.write(block)
                            bar.update(len(block))
            if expected_checksum and digest(partial, algorithm).lower() != expected_checksum.lower():
                raise ValueError(f"{algorithm} checksum mismatch")
            partial.replace(target)
            if entry.get("type") == "zip":
                with zipfile.ZipFile(target) as archive:
                    bad_member = archive.testzip()
                    if bad_member:
                        raise ValueError(f"corrupt zip member: {bad_member}")
                    archive.extractall(destination)
            files = [p for p in destination.rglob("*") if p.is_file() and p.name != ".DS_Store"]
            return {"name": name, "url": url, "license": entry.get("license", "unspecified"), "destination": str(destination.relative_to(ROOT)), "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files), "files": [str(p.relative_to(ROOT)) for p in files]}
        except Exception as exc:  # Report and continue to other configured sources.
            last_error = str(exc)
            print(f"Attempt {attempt}/3 failed for {name}: {last_error}")
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    manual(name, last_error)
    return None


def update_sources(records: list[dict]) -> None:
    text = SOURCES.read_text(encoding="utf-8")
    for record in records:
        text += f"\n\nDownloaded {record['name']} on {record['downloaded_at_utc']}; {record['file_count']} files, {record['total_bytes']} bytes under `{record['destination']}`."
    SOURCES.write_text(text, encoding="utf-8")


def main() -> int:
    with CONFIG.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    records = []
    prior = {item["name"]: item for item in json.loads(MANIFEST.read_text(encoding="utf-8"))} if MANIFEST.exists() else {}
    for entry in config.get("datasets", []):
        if not entry.get("enabled", False):
            continue
        result = download(entry)
        if result:
            prior[result["name"]] = result
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(list(prior.values()), indent=2), encoding="utf-8")
    records = [v for v in prior.values() if v.get("downloaded_at_utc")]
    if records:
        update_sources([r for r in records if r["name"] not in {x["name"] for x in json.loads(MANIFEST.read_text(encoding="utf-8")) if False}])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
