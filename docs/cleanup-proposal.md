# Loan approval cleanup proposal

**Pending approval.** No script, CSV, evaluation logic or repository name has changed.

## Which experiment to feature

Use `codigo6.py` (V14) as the readable project entry point: individual decision-tree/logistic-regression checks and an XGBoost/random-forest/HistGradientBoosting soft-voting ensemble. The latest numbered output represented by the saved scripts is V22 (`codigo4.py`), but its name does not prove it was the final competition submission. Confirm the actual submitted artifact before labeling any version as the model that placed second.

`codigo.py` writes `submission_v19_restoration.csv`, which is absent from the tree; `submission_v19.csv` is present, but there is no verified equivalence between them.

## Proposed layout

| Current files | Proposed location | Required handling |
| --- | --- | --- |
| `codigo6.py` | `src/competition_ensemble.py` | Preserve modelling; update only explicit input/output path handling with approval. |
| `codigo5.py` | `src/experiments/xgboost_baseline.py` | Preserve the V4 experiment. |
| `codigo.py` | `src/experiments/ensemble_restoration.py` | Preserve the V19 restoration. |
| `codigo2.py` | `src/experiments/monotonic_ensemble.py` | Preserve the V20 experiment. |
| `codigo3.py` | `src/experiments/pseudo_labeling_ensemble.py` | Preserve the V21 experiment. |
| `codigo4.py` | `src/experiments/monotonic_pseudo_labeling.py` | Preserve the V22 experiment. |
| `train.csv`, `test_nolabel.csv`, `sample_submission.csv` | `data/` | Keep inputs and schema; document provenance/redistribution permission before broader reuse. |
| Retained prediction CSVs | `results/reference_submissions/` | Read-only experiment artifacts; new runs should use a different output location. |
| Evaluation design and experiment index | `docs/` | Document assumptions, validation and artifact provenance. |

The current scripts expect input files in the process working directory. Moving the CSVs without updating paths would break them; the layout is not implemented yet.

## Submission retention proposal

All 26 files contain 7,050 prediction rows and `id,Accept` columns. No pair has the same Git blob SHA. Filenames and file order do not prove which prediction file was actually submitted or its score.

| Keep for now | Why |
| --- | --- |
| `submission_v4_optimized.csv` | Matches the output named in `codigo5.py`; baseline reference. |
| `submission_v14_giants_ensemble.csv` | Matches `codigo6.py`; representative ensemble. |
| `submission_v20_inductive_bias.csv` | Matches `codigo2.py`; distinct monotonic experiment. |
| `submission_v21_pseudo_labeling.csv` | Matches `codigo3.py`; distinct transductive experiment. |
| `submission_v22_fusion_final.csv` | Matches `codigo4.py`; latest represented fusion experiment. |
| `submission_final.csv` | Temporarily preserve until the final competition artifact is identified; name alone is insufficient. |
| `sample_submission.csv` | Required schema example; excluded from the 26 prediction artifacts. |

Matching the filename to a script establishes intended output provenance, not equality with a fresh execution. Retain the actual final submitted file once confirmed even if it is in the proposed deletion list below.

### Exact proposed deletion list

- `submission_balanced_final.csv`
- `submission_best_RandomForest_timesplit.csv`
- `submission_blend.csv`
- `submission_comparative.csv`
- `submission_ensemble_timesplit.csv`
- `submission_final_with_scores.csv`
- `submission_fixed.csv`
- `submission_mlp_pca.csv`
- `submission_v10_final_stacking.csv`
- `submission_v11_tfidf_voting.csv`
- `submission_v13_manual_sampling.csv`
- `submission_v15_final_requirements.csv`
- `submission_v17_fix_pipeline.csv`
- `submission_v18.csv`
- `submission_v19.csv`
- `submission_v5_nlp_ratios.csv`
- `submission_v6.csv`
- `submission_v7_ensemble.csv`
- `submission_v8_timesplit_nlp.csv`
- `submission_v9_stacking.csv`

These 20 files are older/alternative prediction artifacts whose generating source is not linked by the output names in the six surviving scripts. `submission_final_with_scores.csv` contains no score column. Delete only after confirming that none is the final ranked submission and after approval. Git history would retain recoverability; no history rewrite is proposed.

## Leakage-safe evaluation proposal

1. Reserve an untouched holdout from raw labeled rows; select stratified or time/group-aware splitting after checking the intended prediction setting and repeated entities.
2. Keep deterministic cleaning separate from learned preprocessing. Fit rare-category grouping, categorical mappings and median imputation only on training folds.
3. Cross-fit target encoding for each training fold; use mappings learned from that fold's training labels to transform validation rows. Handle unseen categories with a training-only prior.
4. Evaluate the same baseline models and the final ensemble with out-of-fold predictions and macro-F1; tune weights or thresholds only inside training validation.
5. Evaluate once on the untouched holdout. Treat pseudo-labeling as a separate transductive competition variant and exclude holdout records from its training.
6. Audit whether disbursement features exist at the prediction date. Report results without promising an improvement.

This can keep the binary-classification purpose while correcting validation leakage, but it changes preprocessing and evaluation. **Wait for approval before implementing.**
