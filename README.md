# Loan approval classification — academic ML competition

An academic competition to predict loan approval for **small and medium-sized enterprises (SMEs)** from tabular records. **Our team ranked 2nd out of 7 teams.**

We used **Python, NumPy, pandas, scikit-learn, logistic regression, random forest, XGBoost, cross-validation and ensemble modelling**. This is a team academic project; no individual authorship breakdown is claimed for every script.

## Final competition versions

The project owner confirmed that **`codigo3.py` and `codigo4.py` were the two best versions submitted in the final competition work**. Both are retained without changes to their modelling logic. The team ranking is not attributed to an individually verified score for either file.

| Script | Approach | Output written |
| --- | --- | --- |
| [`codigo3.py`](codigo3.py) — V21 | Decision-tree and logistic-regression baseline checks; soft-voting XGBoost + random forest + HistGradientBoosting (5:3:2); pseudo-labeling above 0.95 confidence. | `submission_v21_pseudo_labeling.csv` |
| [`codigo4.py`](codigo4.py) — V22 | XGBoost with a monotonic constraint on disbursement amount + random forest (6:4); pseudo-labeling above 0.97 confidence; final threshold 0.45. | `submission_v22_fusion_final.csv` |

Start with `codigo3.py` to understand the baseline checks and three-model ensemble, then compare the constrained variant in `codigo4.py`. Earlier scripts have been removed from the current branch.

The project record includes cross-validation, but the retained code does not implement it: `codigo3.py` checks baseline models on one stratified holdout, and `codigo4.py` has no active validation loop.

## Data and execution

- `train.csv`: labeled records with binary `Accept`.
- `test_nolabel.csv`: competition prediction records.
- `sample_submission.csv`: expected submission schema.
- 26 `submission_*.csv` files: saved predictions with 7,050 rows and `id,Accept` columns. The two outputs listed above are the proposed retained final artifacts; deletion of the other 24 is pending approval.
- `requirements.txt`: existing package requirements.

In a virtual environment, run from the repository root:

```bash
python -m pip install -r requirements.txt
python codigo3.py
# Alternatively:
python codigo4.py
```

Each script overwrites its named submission file. Models use many estimators and can consume substantial time and memory. Full retraining and equality with the saved submissions have not been verified.

## Evaluation limitations

- **Target leakage in validation:** target-encoding mappings use all training labels before the holdout split in `codigo3.py`. Validation labels therefore influence their own features.
- **Preprocessing leakage:** its median imputer is fitted on all labeled rows before splitting. Both versions use concatenated train/test covariates for some feature construction, making preprocessing transductive.
- **Final ensemble assessment:** the baseline holdout scores in V21 do not evaluate its final full-data ensemble. V22 has no active validation loop.
- **Pseudo-labeling:** both versions train again on predicted test labels. This is a transductive competition experiment and is not evidence of performance on new unseen businesses.
- **Feature timing:** disbursement dates and amounts need an availability audit before interpreting this as prediction at the original loan-application date.
- Local scores and score comments are not independently verified out-of-sample results. No numerical F1 claim is made here.

A leakage-safe evaluation can preserve the project's purpose: split raw rows first, fit preprocessing inside training folds, cross-fit target encoding, and reserve an untouched holdout for final assessment. That changes the evaluation/model workflow and is **proposed only**, pending approval.

## Remaining organization proposal

See [the cleanup proposal](docs/cleanup-proposal.md) for a `src/`, `data/` and `results/` layout, the exact submission retention proposal and the evaluation design. Input data, saved submissions and both final scripts remain unchanged.
