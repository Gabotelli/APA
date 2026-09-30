# Loan approval classification — academic ML competition

An academic competition to predict loan approval for **small and medium-sized enterprises (SMEs)** from tabular records. **Our team ranked 2nd out of 7 teams.**

We used **Python, NumPy, pandas, scikit-learn, logistic regression, random forest, XGBoost, cross-validation and ensemble modelling**. This is a team academic project; no individual authorship breakdown is claimed for every script.

## Final competition versions

The project owner confirmed `codigo3.py` (V21) and `codigo4.py` (V22) as the two best versions submitted in the final competition work. Their model logic is preserved; only file paths were adapted to the new layout. The ranking is not attributed to an individually verified score for either artifact.

| Source | Approach | Saved reference output |
| --- | --- | --- |
| [V21: pseudo-labeling ensemble](src/pseudo_labeling_ensemble.py), formerly `codigo3.py` | Decision-tree/logistic-regression baseline checks; soft-voting XGBoost + random forest + HistGradientBoosting (5:3:2); pseudo-labeling above 0.95 confidence. | [V21 CSV](results/reference_submissions/submission_v21_pseudo_labeling.csv) |
| [V22: constrained ensemble](src/monotonic_pseudo_labeling.py), formerly `codigo4.py` | Monotonic XGBoost + random forest (6:4); pseudo-labeling above 0.97 confidence; final threshold 0.45. | [V22 CSV](results/reference_submissions/submission_v22_fusion_final.csv) |

The other four scripts and 24 alternative submission CSVs were removed from the current branch. No Git history was rewritten.

## Repository layout

- `src/`: the two competition versions and a separate post-competition evaluation.
- `data/`: original `train.csv`, `test_nolabel.csv` and `sample_submission.csv`.
- `results/reference_submissions/`: the two preserved CSVs, each with 7,050 rows and `id,Accept` columns.
- `results/runs/`: outputs from new competition-script runs, ignored by Git.
- `results/evaluation/`: new evaluation reports, ignored by Git.
- `tests/`: checks for isolation of preprocessing and target-encoding cross fitting.
- `docs/`: evaluation design and cleanup record.

## Run

Install requirements in a virtual environment:

```bash
python -m pip install -r requirements.txt
python src/pseudo_labeling_ensemble.py
# Or:
python src/monotonic_pseudo_labeling.py
```

Input/output paths are relative to the repository location, so these scripts can be invoked from another working directory. New predictions go to `results/runs/`, preserving the reference submissions. Full retraining and equality with the reference CSVs have not been verified.

## Evaluation added after the competition

[The new evaluation](src/evaluate.py) reserves 20% of raw labeled rows as a holdout, obtains stratified out-of-fold predictions on the remaining 80%, and evaluates fixed baseline models and inductive counterparts of both ensembles. Learned category grouping, bank frequencies, imputation and ordinal mappings are fitted on training folds; target encoding uses internal cross fitting. Neither competition test rows nor holdout rows are used for pseudo-labeling.

```bash
python src/evaluate.py
# Faster functional check, not a benchmark:
python src/evaluate.py --quick --max-rows 1500 --folds 3
python -m unittest discover -s tests -v
```

The full command retains the historical ensemble weights, estimator counts and V22 monotonic constraint/threshold. The evaluation is a new workflow rather than an exact reproduction of the transductive submissions. It uses training-only category mappings, robust parsing, and keeps `State` as an ordinal feature instead of dropping it after inspecting all rows. The original team result remains **2nd out of 7 teams**; new local metrics do not measure or replace that ranking.

## Evaluation limitations

The historical competition sources retain their original limitations:

- V21 learns target encodings and median imputation before its baseline holdout split. Its baseline scores therefore contain validation leakage and do not evaluate the final ensemble.
- V22 has no active validation loop. Both original versions construct some features using concatenated train/test covariates and use predicted test labels for retraining.
- Cross-validation was part of the project record, but neither retained competition source implements an active cross-validation loop.

The new evaluation corrects label/preprocessing leakage across its folds and holdout, but uses **random row splits**. Repeated companies and chronology still need an audit before claiming entity-independent or forward-looking performance. Disbursement dates/amounts also need a feature-availability audit at the intended prediction date.

No score improvement is promised. Keep the holdout out of model/threshold selection and distinguish inductive evaluation from transductive competition experiments.

See [evaluation design](docs/evaluation.md) and [cleanup record](docs/cleanup-proposal.md). Validation completed: three isolation tests and a four-model smoke run on 1,500 rows with reduced estimator counts; full-size/full-estimator evaluation has not been run.
