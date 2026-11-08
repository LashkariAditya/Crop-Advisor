.PHONY: data clean explore train train-public eval serve test

data:
	python scripts/download_data.py

clean:
	python scripts/clean_faostat.py
	python scripts/clean_faostat_prices.py
	python scripts/explore_data.py

explore:
	python scripts/explore_data.py

train:
	python scripts/train_all.py

train-public:
	python scripts/train_public.py

eval:
	python scripts/evaluate.py

serve:
	uvicorn backend.app:app --reload --port 8000

test:
	pytest -q
