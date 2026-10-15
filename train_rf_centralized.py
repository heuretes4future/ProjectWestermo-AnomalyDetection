import json
import numpy as np
import pandas as pd
'''
80-20 split
entropy ist för gini

'''
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RANDOM_STATE = 42

DATA_PREP = Path("data_preprocessed")
FILES = {
    "Right":  DATA_PREP / "right_preprocessed.csv",
    "Bottom": DATA_PREP / "bottom_preprocessed.csv",
    "Left":   DATA_PREP / "left_preprocessed.csv",
}

BINARY_LABEL = "NST_B_Label"
MULTI_LABEL = "NST_M_Label"


def load_centralized_dataframe() -> pd.DataFrame:
    dfs = []
    for node, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}. Run preprocessing.py first.")
        df = pd.read_csv(path)
        df["node"] = node  # bra för felsökning/analys, tas bort från features
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def split_70_10_20(X, y):
    # 20% test först
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Av resterande 80% ska val vara 10% av totalen => 10/80 = 0.125
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=RANDOM_STATE, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def minmax_scale_train_only(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Låt binära 0/1-kolumner vara orörda (one-hot + ev. andra binära)
    binary_cols = []
    non_binary_cols = []
    for col in X_train.columns:
        vals = set(pd.unique(X_train[col].dropna()))
        if vals.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
        else:
            non_binary_cols.append(col)

    if non_binary_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train[non_binary_cols])
        X_train[non_binary_cols] = scaler.transform(X_train[non_binary_cols])
        X_val[non_binary_cols] = scaler.transform(X_val[non_binary_cols])
        X_test[non_binary_cols] = scaler.transform(X_test[non_binary_cols])

    return X_train, X_val, X_test


def evaluate(model, X, y, name="set"):
    y_pred = model.predict(X)
    return {
        f"{name}_accuracy": float(accuracy_score(y, y_pred)),
        f"{name}_precision": float(precision_score(y, y_pred, zero_division=0)),
        f"{name}_recall": float(recall_score(y, y_pred, zero_division=0)),
        f"{name}_f1": float(f1_score(y, y_pred, zero_division=0)),
        f"{name}_confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def main():
    df = load_centralized_dataframe()

    # Features: ta bort labels + node
    feature_cols = [c for c in df.columns if c not in [BINARY_LABEL, MULTI_LABEL, "node"]]
    X = df[feature_cols].copy()

    # Label: binär anomalidetektion
    y = df[BINARY_LABEL].copy()

    # Säkerställ numeric 0/1
    y = pd.to_numeric(y, errors="raise").astype(int)

    X_train, X_val, X_test, y_train, y_val, y_test = split_70_10_20(X, y)
    X_train, X_val, X_test = minmax_scale_train_only(X_train, X_val, X_test)

    # Enkel hyperparam-sök (val-F1)
    candidates = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 20,   "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 20,   "min_samples_leaf": 2},
    ]

    best = None
    best_val_f1 = -1.0

    for hp in candidates:
        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            **hp
        )
        model.fit(X_train, y_train)
        val_f1 = f1_score(y_val, model.predict(X_val), zero_division=0)

        print(f"HP={hp}  val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1    
            best = (hp, model)

    best_hp, best_model = best
    print("\nBest hyperparams:", best_hp)
    print("Best val F1:", best_val_f1)

    # Utvärdera (val + test)
    results = {
        "best_hyperparams": best_hp,
        "splits": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "metrics": {}
    }

    results["metrics"].update(evaluate(best_model, X_val, y_val, name="val"))
    results["metrics"].update(evaluate(best_model, X_test, y_test, name="test"))

    print("\nClassification report (TEST):")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

    # Spara resultat
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rf_centralized_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
