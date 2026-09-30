# Loan approval classification — academic ML competition

An academic competition to predict loan approval for **small and medium-sized enterprises (SMEs)** from tabular records. **Our team ranked 2nd out of 7 teams.** This is the course competition result reported in my CV; it is not an independently measured deployment metric.

We used **Python, NumPy, pandas, scikit-learn, logistic regression, random forest, XGBoost, cross-validation and ensemble modelling**. The committed scripts retain experimental versions rather than a single finalized pipeline. The project record describes cross-validation; the active validation code in the six saved scripts is a single stratified holdout where present. Importing `StratifiedKFold` does not itself implement cross-validation.

## Experiments in this repository

| Script | Main experiment | Output written |
| --- | --- | --- |
| `codigo5.py` | V4: cleaned tabular features, unsmoothed target encoding and balanced XGBoost; holdout macro-F1. | `submission_v4_optimized.csv` |
| `codigo6.py` | V14: keyword features; separate decision-tree/logistic-regression checks; soft-voting XGBoost + random forest + HistGradientBoosting (5:3:2), with sample weights. | `submission_v14_giants_ensemble.csv` |
| `codigo.py` | V19 restoration: three-model weighted ensemble and separate academic baseline checks. | `submission_v19_restoration.csv` (not currently tracked) |
| `codigo2.py` | V20: company-size and bank-frequency features; monotonic XGBoost + random forest (6:4); prediction threshold 0.45. | `submission_v20_inductive_bias.csv` |
| `codigo3.py` | V21: three-model ensemble followed by pseudo-labeling of test records above 0.95 confidence. | `submission_v21_pseudo_labeling.csv` |
| `codigo4.py` | V22: constrained two-model ensemble, pseudo-labeling above 0.97 confidence, threshold 0.45. | `submission_v22_fusion_final.csv` |

**Suggested starting point: `codigo6.py`.** It covers the academic baseline comparison and the main three-model ensemble without the later pseudo-labeling extensions. This is a presentation recommendation, not proof that V14 was the submission responsible for the final ranking. That specific final artifact remains to be confirmed.

## Data and execution

- `train.csv`: labeled records with binary `Accept`.
- `test_nolabel.csv`: records used for competition predictions.
- `sample_submission.csv`: expected submission schema.
- 26 `submission_*.csv` files: saved experiment predictions, each with 7,050 rows and `id,Accept` columns. They contain no scores; even `submission_final_with_scores.csv` has only those two columns.
- `requirements.txt`: existing package requirements.

In a virtual environment, run from the repository root:

```bash
python -m pip install -r requirements.txt
python codigo6.py
```

This overwrites `submission_v14_giants_ensemble.csv`. Models use many estimators and can consume substantial time and memory. Syntax has been checked; full model retraining and equality with the saved submissions have not been verified.

## Evaluation limitations

- **Target leakage in validation:** target-encoding mappings use all training labels before the holdout split. Validation labels therefore influence their own features.
- **Preprocessing leakage:** the median imputer is fitted on all labeled rows before splitting. Rare-category grouping and some encodings use concatenated train/test covariates; this is transductive preprocessing, not a clean inductive evaluation.
- **No evaluated final ensemble in several scripts:** the holdout baseline scores do not measure the final full-data ensemble. Some versions have no active validation loop.
- **Pseudo-labeling:** V21/V22 train again on predicted test labels. This is a transductive competition experiment and is not evidence of performance on new unseen businesses.
- **Feature timing:** disbursement dates and amounts need an availability audit before interpreting this as prediction at the original loan-application date.
- Local scores, version names and score comments are not independently verified out-of-sample results. No numerical F1 claim is made here.

A leakage-safe evaluation can preserve the project's purpose: split raw rows first, fit preprocessing inside training folds, cross-fit target encoding, and reserve an untouched holdout for final assessment. That changes the evaluation/model workflow and is **proposed only**, pending approval.

## Proposed organization

See [the cleanup proposal](docs/cleanup-proposal.md) for the `src/`, `data/` and `results/` layout, the exact submission retention list and the evaluation proposal. Scripts, model logic and saved submissions remain unchanged pending approval.

This is a team academic project. The repository does not establish an individual authorship breakdown for every script.
