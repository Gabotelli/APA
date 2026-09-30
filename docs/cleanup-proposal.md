# Completed loan approval cleanup

Approved by the project owner after confirming `codigo3.py`/V21 and `codigo4.py`/V22 as the two best final submitted versions.

- Retained the two sources under descriptive names in `src/`; only their input/output paths changed.
- Moved all three original input/schema CSVs to `data/` with identical Git blob SHAs.
- Retained V21/V22 CSVs under `results/reference_submissions/` with identical Git blob SHAs.
- Removed the other 24 alternative submission CSVs from the current branch. Earlier source cleanup removed `codigo.py`, `codigo2.py`, `codigo5.py` and `codigo6.py`.
- New competition runs write to ignored `results/runs/`, preserving references.
- Added a separate approved post-competition evaluation; see [its design and validation](evaluation.md).
- Removed unused matplotlib/imbalanced-learn requirements; the retained sources use NumPy, pandas, scikit-learn and XGBoost.
- Kept the historical team result and original evaluation limitations explicit.
- No repository rename, visibility change, archiving or Git-history rewrite was performed.

## Removed prediction artifacts

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
