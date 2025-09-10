from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_sklearn_models_cv(X, y, preproc, models: Dict[str, object],
                               n_splits: int = 5, seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    cms = {}
    for name, est in models.items():
        fold_acc, fold_f1 = [], []
        last_cm = None
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            pipe = Pipeline([("preproc", preproc), ("clf", est)])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = pipe.predict(X.iloc[test_idx])
            acc = accuracy_score(y.iloc[test_idx], preds)
            f1m = f1_score(y.iloc[test_idx], preds, average="macro")
            fold_acc.append(acc); fold_f1.append(f1m)
            last_cm = confusion_matrix(y.iloc[test_idx], preds, labels=sorted(y.unique()))
        rows.append({"model": name,
                     "acc_mean": np.mean(fold_acc), "acc_std": np.std(fold_acc),
                     "f1_macro_mean": np.mean(fold_f1), "f1_macro_std": np.std(fold_f1)})
        cms[name] = last_cm
    return pd.DataFrame(rows).sort_values("acc_mean", ascending=False), cms

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=3))
