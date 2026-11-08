"""Download configured public datasets with retries, resume, and a manifest."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "datasets.yaml"
MANIFEST = ROOT / "data" / "raw" / "MANIFEST.json"
SOURCES = ROOT / "DATA_SOURCES.md"
CHUNK = 1024 * 256


def git_blob_digest(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download_github_tree(entry: dict) -> dict | None:
    """Fetch a pinned subset of files from a GitHub tree with per-file verification."""
    name = entry["name"]
    destination = ROOT / entry["destination"]
    prefix = entry["include_prefix"]
    try:
        response = requests.get(entry["tree_url"], timeout=(15, 45))
        response.raise_for_status()
        tree_data = response.json()
        if tree_data.get("truncated"):
            raise ValueError("GitHub returned a truncated repository tree")
        tree = tree_data.get("tree", [])
        selected = [item for item in tree if item.get("type") == "blob" and
                    (item["path"].startswith(prefix) or item["path"] in entry.get("include_extra_files", []))]
        if not selected:
            raise ValueError(f"no files found under pinned prefix {prefix}")
    except (requests.RequestException, ValueError, KeyError) as exc:
        manual(name, f"GitHub tree lookup failed: {exc}", entry.get("manual_steps", ""))
        return None

    destination.mkdir(parents=True, exist_ok=True)
    raw_base = entry["raw_url"].rstrip("/")
    errors = []

    def fetch(item: dict) -> tuple[bool, str]:
        rel = Path(item["path"])
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                data = target.read_bytes()
                if len(data) == item["size"] and git_blob_digest(data) == item["sha"]:
                    return True, str(target.relative_to(ROOT))
            except OSError:
                pass
        url = f"{raw_base}/{quote(item['path'], safe='/')}"
        last = "unknown error"
        for attempt in range(1, 4):
            try:
                with requests.get(url, stream=True, timeout=(20, 90)) as blob:
                    blob.raise_for_status()
                    temp = target.with_suffix(target.suffix + ".part")
                    digest = hashlib.sha1()
                    digest.update(f"blob {item['size']}\0".encode())
                    size = 0
                    with temp.open("wb") as out:
                        for block in blob.iter_content(CHUNK):
                            if block:
                                out.write(block)
                                digest.update(block)
                                size += len(block)
                if size != item["size"] or digest.hexdigest() != item["sha"]:
                    temp.unlink(missing_ok=True)
                    raise ValueError(f"Git blob verification failed for {item['path']}")
                temp.replace(target)
                return True, str(target.relative_to(ROOT))
            except Exception as exc:
                last = str(exc)
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        return False, f"{item['path']}: {last}"

    from tqdm import tqdm
    expected = entry.get("expected_image_count")
    image_count = sum(item["path"].lower().endswith((".jpg", ".jpeg", ".png", ".webp")) for item in selected)
    if expected is not None and image_count != expected:
        manual(name, f"expected {expected} original images in the pinned tree, found {image_count}", entry.get("manual_steps", ""))
        return None
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(fetch, item) for item in selected]
        for future in tqdm(as_completed(futures), total=len(futures), unit="file", desc=name):
            ok, message = future.result()
            if not ok:
                errors.append(message)
    if errors:
        manual(name, f"{len(errors)} of {len(selected)} files could not be verified; first error: {errors[0]}", entry.get("manual_steps", ""))
        return None

    files = [destination / Path(item["path"]) for item in selected]
    commit = urlparse(entry["tree_url"]).path.rsplit("/", 1)[-1]
    image_files = [item for item in selected if item["path"].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    classes: dict[str, int] = {}
    for item in image_files:
        cls = Path(item["path"]).parts[1]
        classes[cls] = classes.get(cls, 0) + 1
    return {
        "name": name, "url": entry["url"], "license": entry.get("license", "unspecified"),
        "destination": str(destination.relative_to(ROOT)),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files), "total_bytes": sum(path.stat().st_size for path in files),
        "source_commit": commit, "class_counts": classes,
        "sample_files": [str(path.relative_to(ROOT)) for path in files[:10]],
    }


def download_git_sparse(entry: dict) -> dict | None:
    """Sparse-check out only the original soil photos from a pinned Git commit."""
    name = entry["name"]
    destination = ROOT / entry["destination"]
    repo = destination / entry.get("checkout_path", "repo")
    commit = entry["commit"]
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if not (repo / ".git").is_dir():
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--no-checkout", entry["url"], str(repo)], cwd=ROOT, check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", commit], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "--cone", entry["include_path"]], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", commit], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        manual(name, f"pinned sparse Git checkout failed: {exc}", entry.get("manual_steps", ""))
        return None

    image_root = repo / entry["include_path"]
    files = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    expected = entry.get("expected_image_count")
    if not files or (expected is not None and len(files) != expected):
        manual(name, f"expected {expected} original images, found {len(files)}", entry.get("manual_steps", ""))
        return None
    classes: dict[str, int] = {}
    for path in files:
        classes[path.parent.name] = classes.get(path.parent.name, 0) + 1
    return {
        "name": name, "url": entry["url"], "license": entry.get("license", "unspecified"),
        "destination": str(destination.relative_to(ROOT)),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files),
        "source_commit": commit, "class_counts": classes,
        "sample_files": [str(path.relative_to(ROOT)) for path in files[:10]],
    }


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def manual(name: str, reason: str, steps: str = "") -> None:
    print(f"\n{ name }: {reason}")
    print("Manual steps: " + (steps or "open config/datasets.yaml, confirm source and license, download the file from its listed URL, then place it in the dataset destination directory."))
    print("Do not use it until DATA_SOURCES.md records the source and license.")


def secret(name: str) -> str:
    if os.getenv(name): return os.environ[name]
    env_path=ROOT/".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name+"="): return line.split("=",1)[1].strip().strip("\"'")
    return ""


def download(entry: dict) -> dict | None:
    name, url = entry["name"], entry["url"]
    if entry.get("type") == "git_sparse":
        return download_git_sparse(entry)
    if entry.get("type") == "github_tree":
        return download_github_tree(entry)
    if entry.get("type")=="kaggle" and not entry.get("license_verified",False):
        manual(name,"Kaggle license_verified is false. Review the dataset license and terms first, then explicitly set license_verified: true.",entry.get("manual_steps",""))
        return None
    auth=None
    if entry.get("type")=="kaggle":
        username,key=secret("KAGGLE_USERNAME"),secret("KAGGLE_KEY")
        if not username or not key:
            manual(name,"Kaggle credentials are missing. Set KAGGLE_USERNAME and KAGGLE_KEY in repository-root .env.",entry.get("manual_steps",""))
            return None
        auth=(username,key)
    destination = ROOT / entry["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    filename = entry.get("archive_name") or ("dataset.zip" if entry.get("type")=="kaggle" else (entry["expected_files"][0] if entry.get("expected_files") and entry.get("expected_files")[0].lower().endswith(".zip") else Path(urlparse(url).path).name or "dataset.zip"))
    if not Path(filename).suffix:
        filename += ".zip"
    target = destination / filename
    partial = target.with_suffix(target.suffix + ".part")
    expected_checksum = entry.get("checksum")
    algorithm = entry.get("checksum_algorithm", "sha256")

    remote_size = 0
    try:
        head = requests.head(url, allow_redirects=True, timeout=(15, 30),auth=auth)
        if head.ok:
            remote_size = int(head.headers.get("Content-Length", 0))
        else:
            # Some public archive endpoints support GET but return 404/405 to HEAD.
            with requests.get(url, stream=True, allow_redirects=True, timeout=(15, 30),auth=auth) as probe:
                probe.raise_for_status()
                remote_size = int(probe.headers.get("Content-Length", 0))
    except requests.RequestException as exc:
        manual(name, f"source HEAD/GET availability check failed: {exc}",entry.get("manual_steps",""))
        return None

    if target.exists():
        size_bad = remote_size > 0 and target.stat().st_size != remote_size
        checksum_bad = expected_checksum and digest(target, algorithm).lower() != expected_checksum.lower()
        if size_bad or checksum_bad:
            manual(name, "existing file size/checksum does not match; moved aside as .invalid",entry.get("manual_steps",""))
            target.replace(target.with_suffix(target.suffix + ".invalid"))
        else:
            print(f"Skip {name}: file already exists ({target.stat().st_size} bytes)")
            return None

    last_error = "unknown error"
    for attempt in range(1, 4):
        try:
            headers = {"Range": f"bytes={partial.stat().st_size}-"} if partial.exists() else {}
            with requests.get(url, stream=True, allow_redirects=True, timeout=(20, 90), headers=headers,auth=auth) as response:
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
            if entry.get("type") in {"zip","kaggle"}:
                with zipfile.ZipFile(target) as archive:
                    bad_member = archive.testzip()
                    if bad_member:
                        raise ValueError(f"corrupt zip member: {bad_member}")
                    root = destination.resolve()
                    for member in archive.infolist():
                        member_path = (destination / member.filename).resolve()
                        if root not in member_path.parents and member_path != root:
                            raise ValueError(f"unsafe archive path: {member.filename}")
                    archive.extractall(destination)
            files = [p for p in destination.rglob("*") if p.is_file() and p.name != ".DS_Store"]
            missing = [expected for expected in entry.get("expected_files", []) if not any(p.name == expected for p in files)]
            if missing:
                raise ValueError("archive did not contain expected file(s): " + ", ".join(missing))
            return {"name": name, "url": url, "license": entry.get("license", "unspecified"), "destination": str(destination.relative_to(ROOT)), "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files), "archive_size_bytes": target.stat().st_size, "archive_sha256": digest(target,"sha256"), "sample_files": [str(p.relative_to(ROOT)) for p in files[:10]]}
        except Exception as exc:  # Report and continue to other configured sources.
            last_error = str(exc)
            print(f"Attempt {attempt}/3 failed for {name}: {last_error}")
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    manual(name, last_error,entry.get("manual_steps",""))
    return None


def update_sources(records: list[dict]) -> None:
    text = SOURCES.read_text(encoding="utf-8")
    for record in records:
        text += f"\n\nDownloaded {record['name']} on {record['downloaded_at_utc']}; {record['file_count']} files, {record['total_bytes']} bytes under `{record['destination']}`."
    SOURCES.write_text(text, encoding="utf-8")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",action="append",dest="selected",help="download only this configured dataset name; repeat to select more than one")
    args=parser.parse_args()
    with CONFIG.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    entries=config.get("datasets",[])
    if args.selected:
        by_name={entry["name"]:entry for entry in entries}
        unknown=sorted(set(args.selected)-set(by_name))
        if unknown:
            parser.error("unknown dataset name(s): "+", ".join(unknown))
        disabled=[name for name in set(args.selected) if not by_name[name].get("enabled",False)]
        if disabled:
            parser.error("dataset(s) are disabled pending source/license review: "+", ".join(sorted(disabled)))
        entries=[by_name[name] for name in dict.fromkeys(args.selected)]
    downloaded_now = []
    prior = {item["name"]: item for item in json.loads(MANIFEST.read_text(encoding="utf-8"))} if MANIFEST.exists() else {}
    for entry in entries:
        if not entry.get("enabled", False):
            continue
        result = download(entry)
        if result:
            prior[result["name"]] = result
            downloaded_now.append(result)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(list(prior.values()), indent=2), encoding="utf-8")
    if downloaded_now:
        update_sources(downloaded_now)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

