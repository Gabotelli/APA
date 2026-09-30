# Loan approval cleanup and remaining proposals

## Completed cleanup

The project owner confirmed `codigo3.py` (V21) and `codigo4.py` (V22) as the two best versions submitted in the final competition work. These two sources remain byte-for-byte unchanged.

The earlier `codigo.py`, `codigo2.py`, `codigo5.py` and `codigo6.py` have been removed from the current branch. All input data and saved CSVs are still present. Git history has not been rewritten.

## Proposed layout — pending approval

| Current files | Proposed location | Required handling |
| --- | --- | --- |
| `codigo3.py` | `src/pseudo_labeling_ensemble.py` | Preserve modelling; update input/output path handling only after approval. |
| `codigo4.py` | `src/monotonic_pseudo_labeling.py` | Preserve modelling; update input/output path handling only after approval. |
| `train.csv`, `test_nolabel.csv`, `sample_submission.csv` | `data/` | Keep inputs and schema. |
| Two final prediction CSVs | `results/reference_submissions/` | Preserve saved artifacts; use a separate output location for new runs. |
| Evaluation design | `docs/` | Document assumptions, validation and artifact provenance. |

The scripts currently expect data in the process working directory. Moving files without updating their paths would break execution; this layout has not been implemented.

## Submission retention proposal — pending approval

| Keep | Why |
| --- | --- |
| `submission_v21_pseudo_labeling.csv` | Named output of the confirmed final version `codigo3.py`. |
| `submission_v22_fusion_final.csv` | Named output of the confirmed final version `codigo4.py`. |
| `sample_submission.csv` | Submission schema example; excluded from the 26 prediction artifacts. |

All 26 prediction files contain 7,050 rows and `id,Accept` columns. Matching a filename to a script does not verify equality with a fresh execution or an individual competition score.

The following 24 older/alternative prediction files are proposed for deletion from the current branch. **None has been deleted in this pass.**

- `submission_balanced_final.csv`
- `submission_best_RandomForest_timesplit.csv`
- `submission_blend.csv`
- `submission_comparative.csv`
- `submission_ensemble_timesplit.csv`
- `submission_final.csv`
- `submission_final_with_scores.csv`
- `submission_fixed.csv`
- `submission_mlp_pca.csv`
- `submission_v10_final_stacking.csv`
- `submission_v11_tfidf_voting.csv`
- `submission_v13_manual_sampling.csv`
- `submission_v14_giants_ensemble.csv`
- `submission_v15_final_requirements.csv`
- `submission_v17_fix_pipeline.csv`
- `submission_v18.csv`
- `submission_v19.csv`
- `submission_v20_inductive_bias.csv`
- `submission_v4_optimized.csv`
- `submission_v5_nlp_ratios.csv`
- `submission_v6.csv`
- `submission_v7_ensemble.csv`
- `submission_v8_timesplit_nlp.csv`
- `submission_v9_stacking.csv`

## Leakage-safe evaluation proposal — pending approval

1. Reserve an untouched holdout from raw labeled rows; choose stratified or time/group-aware splitting after checking the prediction setting and repeated entities.
2. Separate deterministic cleaning from learned preprocessing. Fit category grouping, mappings and median imputation only on training folds.
3. Cross-fit target encoding within training folds; transform validation rows with training-only mappings and priors.
4. Evaluate the baseline models and final ensemble with out-of-fold predictions and macro-F1; tune weights or thresholds only within training validation.
5. Assess once on the untouched holdout. Treat pseudo-labeling as a separate transductive competition variant, excluding holdout records from training.
6. Audit feature availability at the prediction date. Report measured results without promising an improvement.

This preserves the binary-classification purpose but changes preprocessing and evaluation. **Do not implement without approval.**
