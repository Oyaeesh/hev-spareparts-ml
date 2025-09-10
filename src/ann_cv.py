from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def build_ann(input_dim: int, units: int = 64, dropout: float = 0.1, lr: float = 1e-3, num_classes: int = 3):
    model = Sequential([
        Dense(units, activation="relu", input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def evaluate_ann_cv(X_np, y_np, n_splits: int = 5, seed: int = 42,
                    units: int = 64, dropout: float = 0.1, lr: float = 1e-3,
                    epochs: int = 100, batch_size: int = 32) -> Tuple[dict, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []
    last_cm = None
    for train_idx, test_idx in skf.split(X_np, y_np):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        model = build_ann(input_dim=X_np.shape[1], units=units, dropout=dropout, lr=lr, num_classes=len(np.unique(y_np)))
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                  epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

        preds = model.predict(X_te, verbose=0)
        yhat = np.argmax(preds, axis=1)
        accs.append(accuracy_score(y_te, yhat))
        f1s.append(f1_score(y_te, yhat, average="macro"))
        last_cm = confusion_matrix(y_te, yhat, labels=sorted(np.unique(y_np)))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s))}, last_cm
