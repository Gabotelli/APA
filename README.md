# APA: loan approval classification experiments

A collection of Python experiments for predicting the binary `Accept` field from tabular loan records. The repository contains input CSV files, several model scripts and saved submission CSV files. It documents exploratory academic work, not a deployed credit decision system or a validated financial model.

## What is implemented

The `codigo*.py` scripts explore cleaning currency and date fields, categorical encoding, engineered ratios and indicators, and classification with scikit-learn and XGBoost. The models include decision trees, logistic regression, random forests, gradient boosting and soft-voting ensembles. Each script is a separate experiment; filenames do not define a single production pipeline. `codigo6.py`, for example, trains individual baseline models and a voting ensemble, then writes `submission_v14_giants_ensemble.csv`.

## Files and execution

- `train.csv`: training records with `Accept` labels.
- `test_nolabel.csv`: records for generating predictions.
- `sample_submission.csv` and `submission_*.csv`: example and saved prediction files.
- `codigo.py` through `codigo6.py`: alternative experiments.
- `requirements.txt`: Python package requirements.

From the repository root, in a Python virtual environment:

```bash
python -m pip install -r requirements.txt
python codigo6.py
```

This reads the two input CSV files and overwrites the output file named in that script. Some models use many estimators and may take time and memory to fit. Other scripts may write differently named submission files.

## Evaluation limits

The repository includes internal validation calculations, but some preprocessing and target encoding are fitted before the train/validation split. This can leak information into the validation data and make reported scores optimistic. Saved submissions and score references in commit messages are records of experiments, not independently verified out-of-sample performance. A leakage-safe preprocessing pipeline and a held-out evaluation would be needed before drawing stronger conclusions.
