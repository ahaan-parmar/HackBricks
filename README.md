# HackBricks — Drop(In)

Student-dropout prediction and intervention platform built on Databricks.
It trains classifiers on the *Predict Students' Dropout and Academic Success*
dataset, explains predictions with SHAP, audits them for fairness, groups
at-risk students into segments, and generates LLM-written intervention reports.
A FastAPI backend serves everything to a React dashboard.

## Repo layout

| Path | What it is |
|------|------------|
| `src/` | Databricks notebooks — bronze → silver → train → register → fairness/SHAP → gold |
| `local/` | Same pipeline as runnable local scripts (`run_pipeline.py` orchestrates) |
| `resources/` | Databricks Asset Bundle job definition (`ml_pipeline_job.yml`) |
| `backend/` | `main.py` — the Drop(In) FastAPI service (dashboard, students, segments, model eval, SHAP plots, fairness audit, LLM reports, fee calculator) |
| `frontend/` | Vite + React + TypeScript + shadcn/ui dashboard |
| `data/` | `models/` (trained `.pkl` + metadata JSON), `processed/`, and generated lake layers (git-ignored) |
| `dataset/` | Source CSV (git-ignored — download separately) |
| `tests/` | pytest suite (feature engineering) |
| `databricks.yml` | Asset Bundle config (dev / prod targets) |

## Prerequisites

- Python 3.11+
- Node 18+ and [Bun](https://bun.sh/) (or npm) for the frontend
- A Databricks workspace + the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) if you deploy the bundle

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset CSV at `dataset/students_dropout_academic_success.csv`.

## Run locally

```bash
# 1. Build bronze + silver parquet layers
python local/run_pipeline.py

# 2. Train models, cluster + SHAP, LLM recommendations
python local/03_train_models.py
python local/04_cluster_shap.py
python local/05_llm_recommendations.py

# 3. Start the API (from the project root)
python backend/main.py           # http://127.0.0.1:8000

# 4. Start the dashboard
cd frontend
bun install && bun run dev       # http://127.0.0.1:5173
```

## Deploy to Databricks

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run ml_pipeline_job -t dev
```

## Tests

```bash
pytest                    # backend / pipeline
cd frontend && bun run test
```

## Notes

- Trained model binaries and the raw dataset are git-ignored; regenerate them
  from the pipeline scripts above.
- `src/` (Databricks notebooks) and `local/` (local scripts) are kept in sync
  intentionally — edit both when changing pipeline logic.
