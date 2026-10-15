import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt

from preprocessing import load_and_preprocess, normalize
from config_westermo import WESTERMO_CONFIG

RANDOM_STATE = 42
N_FOLDS = 5

# Base RF params
RF_BASE_PARAMS = {
    "criterion": "entropy",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Small tuning grid (validation tuning inside each fold)
RF_TUNE_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced"],
}

# Threshold search space for binary AD
THRESHOLDS = np.linspace(0.05, 0.95, 91)  # 0.05..0.95 step 0.01


def load_data():
    datasets, binary_col, multi_col = load_and_preprocess(WESTERMO_CONFIG)
    return datasets, binary_col, multi_col


def dict_product(grid: dict):
    import itertools
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def score_metrics(y_true, y_pred, average="weighted"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
    }


def tune_rf_on_validation(X_train, y_train, X_val, y_val, task_name):
    """
    Tune RF hyperparameters on validation set.
    - For binary (AD): optimize F1 for anomaly class (pos_label=1) if labels are {0,1}
    - For multiclass (AC): optimize weighted F1
    """
    X_train_n, X_val_n = normalize(X_train, X_val)
    labels = sorted(np.unique(y_train).tolist())
    is_binary_01 = (labels == [0, 1])

    best_params = None
    best_score = -1.0

    for params in dict_product(RF_TUNE_GRID):
        model = RandomForestClassifier(**RF_BASE_PARAMS, **params)
        model.fit(X_train_n, y_train)
        y_val_pred = model.predict(X_val_n)

        if is_binary_01:
            score = f1_score(y_val, y_val_pred, average="binary", pos_label=1, zero_division=0)
        else:
            score = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

        if score > best_score:
            best_score = score
            best_params = params

    print(f"Best val score ({task_name}): {best_score:.4f} with params: {best_params}")
    return best_params


def find_best_threshold(y_true, y_proba_pos, thresholds=THRESHOLDS):
    """
    Finds threshold that maximizes F1 for the positive class (1).
    Returns (best_threshold, best_f1).
    """
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred_t = (y_proba_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t, average="binary", pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, float(best_f1)


def train_eval_with_valsplit(
    X_train_val, y_train_val,
    X_test, y_test,
    all_classes, task_name,
    use_threshold_tuning_for_binary_ad=True
):
    """
    Inside each CV fold:
      - Fold test = 20% (given by StratifiedKFold)
      - Split remaining 80% into Train/Val as 70/10 of total:
          Val = 12.5% of train_val (=> 10% of total)
          Train = 87.5% of train_val (=> 70% of total)
      - Tune RF params on Val
      - Retrain RF on Train+Val
      - Evaluate on Test
      - For binary AD: ALSO tune threshold on Val probabilities and apply on Test
    """
    # 70/10/20 within fold
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.125,
        stratify=y_train_val,
        random_state=RANDOM_STATE
    )

    best_params = tune_rf_on_validation(X_train, y_train, X_val, y_val, task_name)

    # Retrain on Train+Val
    X_tr_full = pd.concat([X_train, X_val], axis=0)
    y_tr_full = pd.concat([y_train, y_val], axis=0)

    X_tr_full_n, X_test_n = normalize(X_tr_full, X_test)

    model = RandomForestClassifier(**RF_BASE_PARAMS, **best_params)
    model.fit(X_tr_full_n, y_tr_full)

    labels = sorted(np.unique(y_tr_full).tolist())
    is_binary_01 = (labels == [0, 1])

    chosen_threshold = None
    threshold_val_best_f1 = None

    # Default prediction
    y_pred = model.predict(X_test_n)

    # Threshold tuning ONLY for binary AD with labels {0,1}
    if use_threshold_tuning_for_binary_ad and is_binary_01 and "Anomaly Detection" in task_name:
        # Need normalized val too (fit scaler on Train only!). Our normalize() fits on first arg.
        X_train_n, X_val_n = normalize(X_train, X_val)

        # Train model on TRAIN only for threshold tuning on VAL
        model_for_thr = RandomForestClassifier(**RF_BASE_PARAMS, **best_params)
        model_for_thr.fit(X_train_n, y_train)

        y_val_proba = model_for_thr.predict_proba(X_val_n)[:, 1]
        chosen_threshold, threshold_val_best_f1 = find_best_threshold(y_val, y_val_proba)

        # Apply threshold on TEST using the final model trained on Train+Val
        y_test_proba = model.predict_proba(X_test_n)[:, 1]
        y_pred = (y_test_proba >= chosen_threshold).astype(int)

        print(f"Chosen threshold (from VAL): {chosen_threshold:.2f} | Val best F1(pos): {threshold_val_best_f1:.4f}")

    # Metrics (weighted overall, like your earlier code + paper style)
    metrics = score_metrics(y_test, y_pred, average="weighted")

    # Confusion matrix with fixed class order
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)

    # Per-class report
    report = classification_report(
        y_test, y_pred,
        labels=all_classes,
        target_names=[str(c) for c in all_classes],
        zero_division=0
    )

    extra = {
        "best_params": best_params,
        "threshold_used_if_any": chosen_threshold,
        "val_best_f1_at_threshold": threshold_val_best_f1,
    }

    return metrics, cm, report, extra


def run_cross_validation(datasets, label_col, all_label_cols, task_name):
    print(f"\nTask: {task_name}")

    all_data = pd.concat(datasets.values(), ignore_index=True)

    feature_cols = [c for c in all_data.columns if c not in all_label_cols]
    X = all_data[feature_cols]
    y = all_data[label_col]

    all_classes = sorted(y.unique().tolist())
    print(f"Classes: {all_classes}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_fold_metrics = []
    total_cm = np.zeros((len(all_classes), len(all_classes)), dtype=int)

    extras_per_fold = []
    reports_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{N_FOLDS}")

        X_train_val, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"Train+Val: {len(X_train_val)} rows, Test: {len(X_test)} rows")

        metrics, cm, report, extra = train_eval_with_valsplit(
            X_train_val, y_train_val,
            X_test, y_test,
            all_classes, task_name,
            use_threshold_tuning_for_binary_ad=True
        )

        all_fold_metrics.append(metrics)
        total_cm += cm
        extras_per_fold.append(extra)
        reports_per_fold.append(report)

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1 (w):    {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")

        print("\nPer-class classification report (this fold):")
        print(report)

    # Average metrics over folds
    avg_metrics = {}
    for key in all_fold_metrics[0]:
        vals = [m[key] for m in all_fold_metrics]
        avg_metrics[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_fold": [float(v) for v in vals],
        }

    print(f"\nAverage over {N_FOLDS} folds:")
    for key, val in avg_metrics.items():
        print(f"{key}: {val['mean']:.4f} (± {val['std']:.4f})")

    # Confusion matrix summed over folds
    print("\nConfusion Matrix (summed over all folds):")
    print(total_cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=all_classes)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (sum over folds) - {task_name}")
    plt.tight_layout()
    plt.show()

    # For binary AD, print anomaly recall from summed CM (assumes class order [0,1])
    if len(all_classes) == 2 and all_classes == [0, 1] and "Anomaly Detection" in task_name:
        TN, FP, FN, TP = total_cm.ravel()
        anomaly_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        anomaly_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        anomaly_f1 = (2 * anomaly_precision * anomaly_recall / (anomaly_precision + anomaly_recall)) if (anomaly_precision + anomaly_recall) > 0 else 0.0
        print(f"\nFrom summed CM -> Precision(pos): {anomaly_precision:.4f} | Recall(pos): {anomaly_recall:.4f} | F1(pos): {anomaly_f1:.4f}")

    extra_out = {
        "classes": [str(c) for c in all_classes],
        "confusion_matrix_sum": total_cm.tolist(),
        "per_fold_extras": extras_per_fold,
    }

    return avg_metrics, extra_out


def main():
    datasets, binary_col, multi_col = load_data()
    all_label_cols = [binary_col, multi_col]

    # AD (binary)
    ad_metrics, ad_extra = run_cross_validation(
        datasets,
        label_col=binary_col,
        all_label_cols=all_label_cols,
        task_name="Anomaly Detection (AD)"
    )

    # AC (multiclass)
    ac_metrics, ac_extra = run_cross_validation(
        datasets,
        label_col=multi_col,
        all_label_cols=all_label_cols,
        task_name="Anomaly Classification (AC)"
    )

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    results = {
        "settings": {
            "n_folds": N_FOLDS,
            "rf_base_params": {k: v for k, v in RF_BASE_PARAMS.items() if k != "n_jobs"},
            "rf_tune_grid": RF_TUNE_GRID,
            "threshold_grid": [float(t) for t in THRESHOLDS],
            "random_state": RANDOM_STATE,
            "note": (
                "Within each fold: test=20%, then train/val split of remaining 80% into 87.5%/12.5% "
                "(=> 70/10 of total). RF tuned on VAL. For binary AD, decision threshold is tuned on VAL "
                "using predict_proba to maximize F1(pos_label=1), then applied on TEST."
            ),
        },
        "anomaly_detection": {"metrics": ad_metrics, "extra": ad_extra},
        "anomaly_classification": {"metrics": ac_metrics, "extra": ac_extra},
    }

    out_path = out_dir / "rf_centralized_val_tuning_plus_threshold.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()