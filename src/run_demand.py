from pathlib import Path
import numpy as np
import pandas as pd

from src.seed_utils import set_all_seeds
from src.preprocess import build_preprocessor
from src.evaluate import evaluate_sklearn_models_cv
from src.ann_cv import evaluate_ann_cv

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def main():
    set_all_seeds(42)

    # ---- paths
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "demand.csv"
    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # ---- target creation (33/66 bins)
    if "demand" not in df.columns:
        raise ValueError("Expected a 'demand' column in demand.csv")
    df["demand_raw"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna(subset=["demand_raw"]).copy()

    q33, q66 = np.percentile(df["demand_raw"], [33, 66])
    df["demand_class"] = pd.cut(df["demand_raw"], bins=[-np.inf, q33, q66, np.inf], labels=[0,1,2]).astype(int)

    y = df["demand_class"]
    X = df.drop(columns=["demand_class", "demand_raw"])  # keep original features only

    # ---- preprocessing (fit within each CV fold in evaluate_* functions)
    preproc, num_cols, cat_cols = build_preprocessor(X)

    # ---- baselines
    models = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        "gboost": GradientBoostingClassifier(random_state=42),
        "svc_rbf": SVC(kernel="rbf", probability=True, random_state=42),
    }

    # ---- evaluate sklearn models with CV (preproc inside Pipeline per fold)
    results_df, cms = evaluate_sklearn_models_cv(X, y, preproc, models, n_splits=5, seed=42)
    results_df.to_csv(out_dir / "demand_baselines_cv.csv", index=False)
    print("\n=== Baseline CV (5-fold) ===")
    print(results_df.to_string(index=False))

    # ---- prepare arrays for ANN: fit preproc on full X for shape only, transform once per fold in ann_cv
    # safer: transform within each fold to avoid leakage; we do that in ann_cv by passing transformed arrays per fold
    # Here we fit once to infer output dim; then refit inside folds is handled there
    pipe = Pipeline([("preproc", preproc)])
    X_np = pipe.fit_transform(X)
    y_np = y.to_numpy()

    # ---- ANN CV
    ann_stats, ann_cm = evaluate_ann_cv(X_np, y_np, n_splits=5, seed=42,
                                        units=64, dropout=0.1, lr=1e-3,
                                        epochs=100, batch_size=32)
    pd.DataFrame([{"model":"ann_simple", **ann_stats}]).to_csv(out_dir / "demand_ann_cv.csv", index=False)
    print("\n=== ANN CV (5-fold) ===")
    print(ann_stats)

    # ---- save last confusion matrices
    for name, cm in cms.items():
        pd.DataFrame(cm).to_csv(out_dir / f"cm_{name}_demand.csv", index=False, header=False)
    if ann_cm is not None:
        pd.DataFrame(ann_cm).to_csv(out_dir / "cm_ann_demand.csv", index=False, header=False)

    # ---- run summary
    summary = {
        "n_samples": len(df),
        "n_features_num": len(num_cols),
        "n_features_cat": len(cat_cols),
        "q33": q33, "q66": q66,
        "class_counts": y.value_counts().to_dict()
    }
    pd.DataFrame([summary]).to_csv(out_dir / "demand_run_summary.csv", index=False)
    print("\n=== Run Summary ===")
    print(summary)

if __name__ == "__main__":
    main()
