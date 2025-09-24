# Plan for HEV Spare Parts Price Classification Notebook

## 1. Objective
Build a production-ready Jupyter notebook (`src/HEV-SpareParts-Price-Classification.ipynb`) that mirrors the demand-classification workflow but targets `data/Price.csv`, delivering reproducible preprocessing, model benchmarking, ANN training, and SHAP explanations for low/high price tiers.

## 2. Assumptions
- `data/Price.csv` is the authoritative dataset for price modeling and remains stable during development.
- The feature semantics align with those in the demand dataset (column names, data types).
- GPU acceleration is not required; CPU execution is sufficient.
- Required Python packages listed in `requirements.txt` (plus those installed ad-hoc in the demand notebook) are available or can be installed on the fly from within the notebook cells.

## 3. Constraints
- Drop exact duplicate rows prior to modeling.
- Exclude leak-prone feature `on line price` from the modeling feature set.
- Must reuse SEED = 42 for all randomized operations.
- Preprocessing (feature taxonomy, imputers, scalers, one-hot encoding, log transforms) should mirror the demand notebook unless data profiling suggests targeted adjustments.
- Notebook must execute from top to bottom without manual intervention and without writing model artifacts to disk (in-memory only, as per current repo policy).
- SHAP bee-swarm figures for low and high price categories must be saved under `figures/`.

## 4. Open Questions
1. Should any additional leak-prone features (beyond `on line price`) be excluded based on EDA findings?
2. Is there a requirement to persist evaluation tables (CSV/PNG) besides the SHAP figures?
3. Are runtime thresholds (execution time limits) a concern for stakeholders when running the notebook locally?

## 5. Acceptance Criteria
- Notebook exists at `src/HEV-SpareParts-Price-Classification.ipynb` with modular sections aligning with the demand workflow.
- All cells execute successfully via `nbconvert --execute` with `ExecutePreprocessor.timeout=0`.
- Preprocessing decisions are justified via inline commentary/EDA outputs.
- Model benchmarking covers the same classical models and ANN grid search as the demand notebook.
- SHAP summary bar plot plus class-specific beeswarm plots for low and high price tiers are generated and saved to `figures/`.
- Plan document kept up to date with checklist progress and notes.

## 6. Checklist (execute sequentially)
1. Profile `data/Price.csv`: inspect schema, dtypes, duplicates, missingness, and value distributions for potential preprocessing tweaks.
2. Decide and document feature taxonomy, leak-prone feature handling, and numeric transforms based on profiling results.
3. Scaffold the price notebook structure (markdown + empty code cells) mirroring the demand notebook lifecycle.
4. Implement data loading, cleaning, and preprocessing helper functions in the notebook.
5. Implement train/test grouped split, binning logic, and leakage checks.
6. Implement ANN hyperparameter tuning, final training, evaluation plots, and grouped CV.
7. Implement classical model benchmarking section with the specified model zoo.
8. Implement feature leakage audit section.
9. Implement SHAP explanation section, ensuring plots are saved under `figures/`.
10. Execute the notebook end-to-end via `nbconvert --execute` and resolve any execution issues.
11. Capture final metrics, ensure generated artifacts (plots) exist, and update plan progress log.

## 7. Minimal Test Plan
- Automated: `jupyter nbconvert --to notebook --execute --inplace src/HEV-SpareParts-Price-Classification.ipynb`
- Manual: Verify that `figures/` contains the SHAP summary and class-specific plots post-run.
- Sanity: Review key output cells (thresholds, confusion matrix, benchmark table) for plausibility.

## 8. Progress Log
- 2025-09-21T19:28:04: Completed Step 2 (feature taxonomy decisions). Documented duplicate removal and exclusion of 'on line price'.
- 2025-09-21T19:27:49: Completed Step 1 (data profiling). Identified 60 duplicates and confirmed 'on line price' as leak risk.
- 2025-09-21T19:43:36: Completed Step 3-9 – Implemented notebook scaffold, preprocessing helpers, model pipelines, benchmarks, leakage audit, and SHAP section..
- 2025-09-21T19:43:36: Completed Step 10 – Executed notebook via nbconvert without errors..
- 2025-09-21T19:43:36: Completed Step 11 – Verified SHAP figures in figures/ and captured key outputs..
- 2025-09-21T22:36:27: Iterated on price notebook to improve accuracy (expanded grids, optional random split, FP32 pipeline).
