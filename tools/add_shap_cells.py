import json
from pathlib import Path


def append_cells(nb_path: Path, new_cells):
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    nb['cells'].extend(new_cells)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding='utf-8')


def make_md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text + "\n"]
    }


def make_code_cell(lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l if l.endswith("\n") else l + "\n" for l in lines]
    }


def main():
    nb_path = Path('src') / 'HEV-SpareParts-Demand-Classification.ipynb'
    if not nb_path.exists():
        raise SystemExit(f"Notebook not found: {nb_path}")

    cells = []

    # 1) Setup & imports, ensure artifacts present
    cells.append(make_md_cell("(# SHAP Setup & Imports)"))
    cells.append(make_code_cell([
        "# Install and import SHAP; ensure artifacts/inputs are available",
        "import sys, subprocess, os, json",
        "try:\n    import shap\nexcept Exception:\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap', '-q'])\n    import shap",
        "print('shap version:', getattr(shap, '__version__', 'unknown'))",
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt",
        "import joblib",
        "from pathlib import Path",
        "import tensorflow as tf",
        "# Re-seed to keep explanations reproducible",
        "np.random.seed(SEED); tf.random.set_seed(SEED)",
        "\n# Ensure we have model, preprocessor, and processed matrices",
        "has_ann = 'model' in globals()",
        "has_pre = 'preprocessor' in globals()",
        "has_train_proc = 'X_train_proc' in globals()",
        "has_test_proc = 'X_test_proc' in globals()",
        "\nif not (has_ann and has_pre and has_train_proc and has_test_proc):",
        "    model_path = Path('demand_clf.keras')",
        "    preproc_path = Path('preprocessor.joblib')",
        "    meta_path = Path('label_metadata.json')",
        "    if not (model_path.exists() and preproc_path.exists() and meta_path.exists()):",
        "        raise FileNotFoundError('Missing saved artifacts. Run training cells first to create demand_clf.keras, preprocessor.joblib, label_metadata.json')",
        "    model = tf.keras.models.load_model(model_path)",
        "    preprocessor = joblib.load(preproc_path)",
        "    with open(meta_path, 'r') as f: meta = json.load(f)",
        "    cat_cols_effective = meta['cat_cols_effective']",
        "    num_all = meta['num_all']",
        "    skewed_cols = meta['skewed_cols']",
        "    other_num_cols = meta['other_num_cols']",
        "    # Locate dataset to build matrices consistent with earlier split",
        "    data_env = os.environ.get('DEMAND_DATA_PATH', '').strip()",
        "    candidates = [Path(p) for p in [data_env] if p] + [Path('data') / 'demand.csv', Path('..') / 'data' / 'demand.csv']",
        "    file_path = None",
        "    for p in candidates:\n        if p.exists():\n            file_path = p\n            break",
        "    if file_path is None:",
        "        raise FileNotFoundError('Could not find demand.csv for SHAP. Set DEMAND_DATA_PATH or place it in data/ or ../data/.')",
        "    df = pd.read_csv(file_path)",
        "    df.columns = df.columns.str.strip()",
        "    X_all = df[cat_cols_effective + num_all].copy()",
        "    groups_all = df['part'].astype(str)",
        "    from sklearn.model_selection import GroupShuffleSplit",
        "    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)",
        "    train_idx, test_idx = next(gss.split(X_all, None, groups=groups_all))",
        "    X_train_raw = X_all.iloc[train_idx]; X_test_raw = X_all.iloc[test_idx]",
        "    # Use the fitted preprocessor to transform",
        "    X_train_proc = preprocessor.transform(X_train_raw)",
        "    X_test_proc  = preprocessor.transform(X_test_raw)",
    ]))

    # 2) Feature names extraction
    cells.append(make_md_cell("(# SHAP Feature Names)"))
    cells.append(make_code_cell([
        "# Extract human-readable feature names from the fitted preprocessor",
        "def get_feature_names(preprocessor):",
        "    try:",
        "        names = preprocessor.get_feature_names_out()",
        "        # Clean prefixes from ColumnTransformer",
        "        return [str(n).replace('num_skew__','').replace('num__','').replace('cat__','') for n in names]",
        "    except Exception:",
        "        names = []",
        "        for name, trans, cols in preprocessor.transformers_:",
        "            if name == 'remainder' and trans == 'drop':",
        "                continue",
        "            if name in ('num_skew','num'):",
        "                names.extend(list(cols))",
        "            elif name == 'cat':",
        "                # Pipeline -> OneHotEncoder",
        "                try:",
        "                    ohe = trans.named_steps.get('ohe')",
        "                    categories = ohe.categories_",
        "                    for col, cats in zip(cols, categories):",
        "                        for cat in cats:",
        "                            names.append(f'{col}={cat}')",
        "                except Exception:",
        "                    names.extend(list(cols))",
        "            else:",
        "                if isinstance(cols, list): names.extend(list(cols))",
        "        return names",
        "\nfeature_names = get_feature_names(preprocessor)",
        "print('n_features:', len(feature_names))",
        "print('first 10:', feature_names[:10])",
    ]))

    # 3) Build SHAP explainer and compute values
    cells.append(make_md_cell("(# SHAP Explainer for ANN)"))
    cells.append(make_code_cell([
        "# Use DeepExplainer for Keras; fallback to KernelExplainer",
        "rng = np.random.RandomState(SEED)",
        "X_bg = X_train_proc",
        "X_eval = X_test_proc if 'X_test_proc' in globals() else X_train_proc",
        "bg_size = min(100, X_bg.shape[0])",
        "explain_size = min(200, X_eval.shape[0])",
        "bg_idx = rng.choice(X_bg.shape[0], size=bg_size, replace=False)",
        "explain_idx = rng.choice(X_eval.shape[0], size=explain_size, replace=False)",
        "background = X_bg[bg_idx]",
        "X_explain = X_eval[explain_idx]",
        "\ntry:",
        "    explainer = shap.DeepExplainer(model, background)",
        "    shap_values = explainer.shap_values(X_explain)",
        "    method = 'DeepExplainer'",
        "except Exception as e:",
        "    print('DeepExplainer failed, falling back to KernelExplainer:', e)",
        "    f = lambda x: model.predict(x, verbose=0)",
        "    explainer = shap.KernelExplainer(f, background)",
        "    shap_values = explainer.shap_values(X_explain, nsamples=100)",
        "    method = 'KernelExplainer'",
        "print('SHAP explainer:', method)",
        "# Persist for later cells",
        "_shap_values = shap_values; _X_explain = X_explain; _background = background",
    ]))

    # 4) Summary plots
    cells.append(make_md_cell("(# SHAP Summary Plots)"))
    cells.append(make_code_cell([
        "# Aggregate across classes for a global bar plot",
        "import matplotlib.pyplot as plt",
        "sv = _shap_values",
        "if isinstance(sv, list):",
        "    shap_abs = np.mean([np.abs(s) for s in sv], axis=0)",
        "else:",
        "    shap_abs = np.abs(sv)",
        "plt.figure(figsize=(10,6))",
        "shap.summary_plot(shap_abs, _X_explain, feature_names=feature_names, plot_type='bar', show=False, max_display=25)",
        "plt.tight_layout(); plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight'); plt.show()",
        "# Beeswarm for class 2 (High demand) if multiclass",
        "if isinstance(sv, list) and len(sv) >= 3:",
        "    plt.figure(figsize=(10,6))",
        "    shap.summary_plot(sv[2], _X_explain, feature_names=feature_names, show=False, max_display=25)",
        "    plt.tight_layout(); plt.savefig('shap_beeswarm_class2.png', dpi=300, bbox_inches='tight'); plt.show()",
        "# Export top-20 features",
        "vals = shap_abs.mean(axis=0)",
        "fi = pd.Series(vals, index=feature_names).sort_values(ascending=False)",
        "fi.head(20).to_csv('shap_feature_importance_top20.csv')",
        "print('Saved: shap_summary_bar.png, shap_beeswarm_class2.png, shap_feature_importance_top20.csv')",
    ]))

    append_cells(nb_path, cells)
    print('Appended SHAP cells:', len(cells))


if __name__ == '__main__':
    main()

