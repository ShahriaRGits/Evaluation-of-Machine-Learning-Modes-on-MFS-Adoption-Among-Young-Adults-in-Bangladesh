# AI Adoption Classification

> Predicting AI adoption intent (UTAUT) from survey data — multi-model
> benchmark with cross-validation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn 1.6](https://img.shields.io/badge/scikit--learn-1.6-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Author: **Shahriar Islam** · ID: 241000461

---

## What this project does

A survey-based classification analysis of **AI adoption intent** using
constructs from the **Unified Theory of Acceptance and Use of Technology
(UTAUT)**:

| Construct | Meaning                       |
|-----------|-------------------------------|
| `EE`      | Effort Expectancy             |
| `SI`      | Social Influence              |
| `FC`      | Facilitating Conditions       |
| `PV`      | Performance Value             |
| `PT`      | Performance Trust             |
| `PR`      | Perceived Risk                |
| `AI`      | **Adoption Intent** (target)  |

Plus demographics: `Gender`, `Age Group`, `Occupation`, `Education`,
`Monthly Income`.

The pipeline trains and benchmarks **six classifiers** on the same
train/test split, then re-evaluates with **k-fold cross-validation** to
check stability.

## Models

1. Logistic Regression
2. Random Forest
3. Support Vector Machine
4. K-Nearest Neighbors
5. Decision Tree
6. Naive Bayes

## Project layout

```
ai-adoption-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # drop survey CSV here; sample.csv is included
│   └── processed/        # reserved for cleaned/split datasets
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # original Colab notebook
├── src/
│   ├── __init__.py
│   ├── preprocessing.py  # load, clean, encode, split
│   ├── visualization.py  # EDA plots
│   ├── models.py         # training, evaluation, CV
│   └── main.py           # pipeline orchestrator
├── figures/              # generated PNGs
└── results/              # generated CSVs, JSON, and saved model
```

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/ai-adoption-classification.git
cd ai-adoption-classification
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

### 2. Add your data

Drop your survey CSV in `data/raw/`. It should match the schema in
[`data/raw/README.md`](data/raw/README.md). The repo includes a
50-row synthetic `sample.csv` so the pipeline runs out of the box.

### 3. Run the pipeline

```bash
python -m src.main
# or point at a different file:
python -m src.main --data data/raw/your-survey.csv
```

Outputs land in `figures/` (PNG) and `results/` (CSV, JSON, model
artifact).

### 4. (Optional) Explore in Jupyter

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Reproducing the results

The exact numbers depend on the dataset, but on the included sample CSV
the pipeline produces:

- **Holdout accuracy** — five of six models tie near the top, Naive Bayes
  trails (class-conditional independence assumption is a poor fit for
  Likert items)
- **Cross-validation** — 4-fold CV with imputation/scaling inside the
  pipeline (no leakage), standard deviation typically < 2%

The synthetic sample is heavily class-imbalanced toward `AI ∈ {4, 5}`,
which inflates accuracy for all models. The real survey is more balanced
and the spread between models widens.

## Design notes

- **Imputation + scaling happen inside the CV pipeline** (`sklearn.pipeline.Pipeline`)
  so cross-validation scores aren't optimistic from seeing test statistics.
- **Stratified train/test split** preserves the class distribution.
- **All plots save to `figures/`** instead of rendering inline — easier to
  track in git and embed in reports.
- **Best model is pickled** with its scaler and target encoder so it can
  be loaded for inference later.

## Data note

The original Colab notebook uploaded the dataset interactively with
`google.colab.files.upload()`. That flow is incompatible with a
reproducible repo, so the pipeline now reads from `data/raw/` directly.
If you want to keep the Colab upload workflow, see
`notebooks/01_eda_and_modeling.ipynb` for the original.

## License

MIT — see [LICENSE](LICENSE).
