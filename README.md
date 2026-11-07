# CropAdvisor

CropAdvisor is a local-first crop planning project. Models will be implemented and trained from scratch in NumPy; this repo does not download pretrained weights or call external AI prediction APIs.

## Milestone 1 (current)

This milestone establishes the repository, dataset registry, safe data downloader, and source log. Project defaults: India, major local crops, INR, hectares and acres, GPU available for training, and hosting undecided. See `config/project.yaml`. Dataset downloads are deliberate and data-only. Review dataset terms before any commercial use. Training and application milestones are not included yet.

## Setup

Requirements: Python 3.11, Git, and GNU Make (or run `python scripts/download_data.py` directly). Install the allowed data tooling:

```powershell
python -m pip install requests pyyaml tqdm
```

Copy `config/.env.example` to `.env` in the repository root and add credentials only when needed. `.env` is ignored by Git.

Run `make data` to download eligible public datasets. Failed or authentication-gated sources will be reported with manual instructions. Downloads go to `data/raw/<dataset_name>/`; a manifest and source log are updated after success.

## Data policy

- No pretrained models or downloaded weights.
- Never use a dataset until its source and license have been checked and recorded in `DATA_SOURCES.md`.
- EuroSAT is for land-cover assistance only; soil photo datasets are needed for soil type. Close-up field photos are the intended app input.
- Kaggle sources are not downloaded until credentials are configured and their terms are reviewed.
- Dataset availability, license terms, and prices can change. Check source pages before reuse.

## Progress

1. [x] Repository and dataset setup
2. [ ] Data cleaning and exploration (waiting for Milestone 1 approval)

