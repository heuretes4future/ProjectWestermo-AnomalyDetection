import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold
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
        df["node"] = node  # endast för analys/loggning
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def minmax_scale_fit_on_train_only(X_train: pd.DataFrame, X_other_list: list[pd.DataFrame]):
    """
    Fit MinMaxScaler on X_train (non-binary cols), transform X_train and all X_other.
    Binary 0/1 columns left unchanged.
    """
    X_train = X_train.copy()
    others = [x.copy() for x in X_other_list]

    non_binary_cols = []
    for col in X_train.columns:
        vals = set(pd.unique(X_train[col].dropna()))
        if not vals.issubset({0, 1, 0.0, 1.0}):
            non_binary_cols.append(col)

    if non_binary_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train[non_binary_cols])
        X_train[non_binary_cols] = scaler.transform(X_train[non_binary_cols])
        for i in range(len(others)):
            others[i][non_binary_cols] = scaler.transform(others[i][non_binary_cols])

    return X_train, others


def main():
    # 1) Load centralized data
    df = load_centralized_dataframe()

    # 2) Build X, y
    feature_cols = [c for c in df.columns if c not in [BINARY_LABEL, MULTI_LABEL, "node"]]
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[BINARY_LABEL], errors="raise").astype(int)

    # 3) 80/20 hold-out test (test rörs aldrig under CV)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Total rows: {len(X)}")
    print(f"Trainval (80%): {len(X_trainval)}")
    print(f"Test (20%): {len(X_test)}\n")

    # 4) 5-fold CV on trainval
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Baseline hyperparams (du kan ändra här senare)
    rf_params = {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "criterion": "entropy",  # paper-lik
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_trainval, y_trainval), start=1):
        X_tr = X_trainval.iloc[tr_idx].copy()
        y_tr = y_trainval.iloc[tr_idx].copy()
        X_va = X_trainval.iloc[va_idx].copy()
        y_va = y_trainval.iloc[va_idx].copy()

        # scaling: fit on fold-train only
        X_tr_scaled, [X_va_scaled] = minmax_scale_fit_on_train_only(X_tr, [X_va])

        model = RandomForestClassifier(**rf_params)
        model.fit(X_tr_scaled, y_tr)

        y_va_pred = model.predict(X_va_scaled)

        prec = precision_score(y_va, y_va_pred, zero_division=0)
        rec = recall_score(y_va, y_va_pred, zero_division=0)
        f1 = f1_score(y_va, y_va_pred, zero_division=0)

        fold_results.append({"fold": fold, "precision": float(prec), "recall": float(rec), "f1": float(f1)})
        print(f"Fold {fold}:  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    f1s = np.array([r["f1"] for r in fold_results], dtype=float)
    ps = np.array([r["precision"] for r in fold_results], dtype=float)
    rs = np.array([r["recall"] for r in fold_results], dtype=float)

    print("\nCV (5-fold) summary on trainval (80%):")
    print(f"Precision mean±std: {ps.mean():.4f} ± {ps.std():.4f}")
    print(f"Recall    mean±std: {rs.mean():.4f} ± {rs.std():.4f}")
    print(f"F1        mean±std: {f1s.mean():.4f} ± {f1s.std():.4f}")

    # 5) Train final model on ALL trainval (80%), then evaluate on held-out test (20%)
    X_trainval_scaled, [X_test_scaled] = minmax_scale_fit_on_train_only(X_trainval, [X_test])

    final_model = RandomForestClassifier(**rf_params)
    final_model.fit(X_trainval_scaled, y_trainval)

    y_test_pred = final_model.predict(X_test_scaled)

    print("\nTEST (held-out 20%):")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    test_cm = confusion_matrix(y_test, y_test_pred).tolist()
    test_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))

    # 6) Save results
    results = {
        "random_state": RANDOM_STATE,
        "mode": "centralized",
        "split": {"trainval": 0.80, "test": 0.20, "cv_folds": 5},
        "rf_params": rf_params,
        "cv": {
            "folds": fold_results,
            "precision_mean": float(ps.mean()),
            "precision_std": float(ps.std()),
            "recall_mean": float(rs.mean()),
            "recall_std": float(rs.std()),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std()),
        },
        "test": {
            "precision": test_prec,
            "recall": test_rec,
            "f1": test_f1,
            "confusion_matrix": test_cm,
        },
        "rows": {"total": int(len(X)), "trainval": int(len(X_trainval)), "test": int(len(X_test))},
        "feature_count": int(X.shape[1]),
    }

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rf_centralized_cv5_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()