# Post-competition evaluation

This evaluation was added during portfolio maintenance after the competition. It must not be presented as the workflow that produced the historical ranking.

## Design

1. Read labeled data only; separate `Accept` from raw features.
2. Reserve a fixed stratified 20% holdout before any learned preprocessing.
3. Use stratified outer cross-validation on the development rows. Each pipeline independently fits rare-bank/city grouping, V22 bank frequencies, median imputation and ordinal category mappings.
4. Use scikit-learn `TargetEncoder.fit_transform` within the pipeline, with five internal folds and smoothing 20. Validation/holdout rows use training-only mappings and priors.
5. Recompute balanced ensemble sample weights in each fit. Keep fixed V21/V22 weights, estimator settings and V22 monotonic disbursement constraint/0.45 threshold.
6. Report development out-of-fold macro-F1 and holdout macro-F1/confusion matrices for the decision tree, logistic regression and both ensembles. Do not choose models or thresholds from holdout metrics.

Pseudo-labeling is excluded from this inductive evaluation. The original scripts retain it for the historical competition workflow. The evaluation does not read `test_nolabel.csv` or any saved submission.

## Scope of changes

The historical sources retain their modelling logic; input/output path setup is the only code adaptation. Their ASTs were compared after normalizing the three I/O expressions.

The new evaluation changes preprocessing to eliminate validation leakage. It uses cross-fitted target encoding instead of full-label encodings, training-only frequencies/mappings, unknown-category handling, robust numeric/date parsing, and keeps `State` as a categorical feature. These choices are documented rather than claiming byte-identical predictions.

## Run and validation

```bash
python src/evaluate.py --folds 5
python -m unittest discover -s tests -v
# Smoke run used during maintenance:
python src/evaluate.py --quick --max-rows 1500 --folds 3
```

The smoke run completed with all four models, 1,200 development rows and 300 holdout rows, 20 estimators/iterations for ensemble components, and five internal target-encoding folds. Three tests passed: unseen validation categories leave training statistics fixed; unique target categories receive cross-fitted priors; holdout transformation leaves fitted imputer/target mappings unchanged.

The smoke check establishes execution and isolation, not portfolio performance. No full-size/full-estimator result is published. Reports are generated under `results/evaluation/` and ignored by Git.

## Remaining limits

- Random row splits do not prove independence across repeated businesses/banks or future periods.
- Disbursement information may be unavailable at the intended loan-application prediction date.
- Holdout results cease to be an independent assessment if used repeatedly for model selection.
- The new workflow evaluates inductive ensemble counterparts, not the historical pseudo-labeled competition submissions.

The feature-timing and temporal/entity audits remain separate work; preprocessing fixes do not establish deployment validity.

Reference: [scikit-learn TargetEncoder documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html), including the difference between `fit_transform` and `fit().transform()`.
