# HEV Spare Parts ML

Reproducible Jupyter notebooks for classifying **demand** and **price tiers** of hybrid electric vehicle (HEV) spare parts. The project contains two fully instrumented workflows that share a common feature taxonomy, preprocessing logic, model zoo, and SHAP-based explainability.

- **Demand notebook**: `src/HEV-SpareParts-Demand-Classification.ipynb`
- **Price notebook**: `src/HEV-SpareParts-Price-Classification.ipynb`

Both notebooks now:
- Expect the lower-case datasets `data/demand.csv` and `data/price.csv`
- Offer a switch for grouped vs. stratified splits (`USE_GROUPED_SPLITS`)
- Tune a feed-forward ANN and several classical baselines (SVC, Logistic Regression, ExtraTrees, SGD, LDA)
- Display SHAP plots inline only (no figures are written to disk)
- Aggregate one-hot encodings back to the original feature names for interpretability

---

## Repository structure

```
hev-spareparts-ml/
+- data/
¦  +- demand.csv
¦  +- price.csv
+- src/
¦  +- HEV-SpareParts-Demand-Classification.ipynb
¦  +- HEV-SpareParts-Price-Classification.ipynb
¦  +- utils/feature_config.py
+- README.md
+- requirements.txt
+- CITATION.cff
+- LICENSE
+- plan.md
```

`utils/feature_config.py` centralises feature blocklists and metadata persistence for both notebooks.

---

## Environment setup

1. **Clone the repo**
   ```powershell
   git clone https://github.com/Oyaeesh/hev-spareparts-ml.git
   cd hev-spareparts-ml
   ```
2. **Create and activate a virtual environment (Windows example)**
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **(Optional) Register a Jupyter kernel**
   ```powershell
   python -m ipykernel install --user --name hev-spareparts-ml --display-name "Python (.venv) hev-spareparts-ml"
   ```
4. **Verify datasets**
   Ensure `data/demand.csv` and `data/price.csv` exist. Override locations via `DEMAND_DATA_PATH` or `PRICE_DATA_PATH` if needed.
5. **Open notebooks in VS Code / JupyterLab** and run all cells.

For automation you can use:
```powershell
python -m nbclient --execute src/HEV-SpareParts-Demand-Classification.ipynb --path src
python -m nbclient --execute src/HEV-SpareParts-Price-Classification.ipynb --path src
```

---

## Shared feature taxonomy

| Column                                   | Type        | Notes                                      |
|------------------------------------------|-------------|--------------------------------------------|
| `car type`, `made in`, `original/imitator`, `new\used`, `selling location`, `service location (repair shop/automotive company` | categorical | One-hot encoded (handle unknown gracefully) |
| `number of cars in jordan`, `car age`, `failure rate`, `price of the car`, `repair or replacement cost`, `car total maintenance cost average`, `critically`, `online price`, `price` | numeric | Skewed subset log-transformed before scaling |
| `part`                                   | categorical | Always retained as a feature in the price notebook; optional (`DROP_PART`) in demand; used for grouped splits |
| Targets                                  | numeric     | `demand`, `price` ? binned into {low, medium, high} |

The price notebook now loads the lower-case `price.csv`; both notebooks print the effective feature lists after sanitation.

---

## Model pipelines

- **Preprocessing** (`make_preprocessor`)
  - Log1p + StandardScaler on skewed numerics
  - StandardScaler on remaining numerics (tree-style pipelines omit the scaler)
  - OneHotEncoder (`handle_unknown='ignore'`) on categoricals
- **ANN grid** (`GRID`)
  - Layers: 1–3 | Units: 16 / 32
  - Dropout: 0.0 / 0.2 | L2: 0 / 1e-4
  - Learning rate: 1e-3 / 3e-3 / 1e-2
  - EarlyStopping and ReduceLROnPlateau now share repo-level constants (`EARLY_STOPPING_PATIENCE = 10`, `EARLY_STOPPING_MIN_DELTA = 1e-4`)
- **Classical grid** (`GRIDS_CLASSIC`)
  - `svc_rbf`, `logreg`, `sgd`, `lda`
  - **ExtraTrees** intentionally constrained (`n_estimators: 100–200`, `max_depth: 8–12`, `min_samples_leaf: 5–10` for price; `n_estimators: 200–300`, `max_depth: 14–18`, `min_samples_leaf: 4–6` for demand) to avoid unrealistically high scores from part-style leakage.

---

## Splitting strategy

Both notebooks expose `USE_GROUPED_SPLITS`:

- **Demand** default: `True` ? GroupShuffleSplit / GroupKFold on `part`
- **Price** default: `False` ? StratifiedShuffleSplit / StratifiedKFold on price-tier bins (suitable when you only score known part types)

Change the flag at the top of each notebook to switch behaviour. All downstream helpers (ANN tuning, classical model selection, CV, leakage audit) respect the flag.

---

## Interpretability workflow (SHAP)

- DeepExplainer preferred; automatically falls back to KernelExplainer if GPU/CUDA incompatibility occurs.
- We aggregate SHAP contributions back to the raw feature names—one-hot encoded columns are summed by source feature.
- Global summary bar charts and class-specific beeswarm plots are displayed inline only; no image files are written.
- The notebooks print:
  - Top aggregated absolute SHAP importances
  - Per-class top contributors (first 10 values)

This workflow reuses the in-memory ANN and preprocessor; ensure the training cells run before the SHAP section.

---

## Outputs

| Notebook | Metric highlights | SHAP output |
|----------|------------------|-------------|
| **Demand** | ANN accuracy ˜ 0.99 (grouped) | Inline summary + beeswarm, aggregated by raw feature |
| **Price**  | ANN accuracy ˜ 0.99 (stratified by price tier) | Same inline plots; no filesystem artifacts |

ExtraTrees models now report more conservative scores (˜0.90 macro F1 for price, ˜0.88–0.93 for demand depending on grouping) thanks to the updated hyperparameter grids.

---

## Reproducibility & troubleshooting

- Seeds (`SEED = 42`) applied to numpy and TensorFlow.
- Each notebook prints explicit missing-column messages if the CSV schema drifts.
- SHAP requires the datasets; if you store them elsewhere, set `PRICE_DATA_PATH` or `DEMAND_DATA_PATH` before executing the SHAP cells.
- Notebooks default to inline visualisation only; remove the final `plt.show()` blocks if you need headless execution.

---

## Citation

```
Abueed, O., Yaeesh, O., AlAlaween, W., Abueed, S., & Alalawin, A. (2025).
A Systematic AI-based Paradigm for Classifying Hybrid Electric Vehicle Spare Parts Using Their Price and Demand (v0.1.0) [Software].
https://github.com/Oyaeesh/hev-spareparts-ml
```

A machine-readable `CITATION.cff` is included; use the GitHub “Cite this repository” button for alternate formats.

---

## License & contact

Released under the [MIT License](LICENSE). For questions, open a GitHub issue or email Omar Abueed (Oabueed1@binghamton.edu).
