import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from preprocessing import load_and_preprocess, normalize
from config_westermo import WESTERMO_CONFIG


RANDOM_STATE = 42
N_FOLDS = 5

# Hyperparameter search grid
PARAM_GRID = {
    "n_estimators": [10, 50, 100],
    "criterion": ["gini", "entropy"],
}


def load_data():
    datasets, binary_col, multi_col = load_and_preprocess(WESTERMO_CONFIG)
    return datasets, binary_col, multi_col


def plot_confusion_matrix(cm, class_labels, title, save_path):
    plt.figure(figsize=(max(6, len(class_labels)), max(5, len(class_labels) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_attack_recall(recall_per_type, title, save_path):
    attack_types = list(recall_per_type.keys())
    recalls = list(recall_per_type.values())

    plt.figure(figsize=(8, max(4, len(attack_types) * 0.6)))
    bars = plt.barh(attack_types, recalls, color="steelblue", edgecolor="black")

    for bar, val in zip(bars, recalls):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2%}", va="center", fontsize=10)

    plt.xlim(0, 1.15)
    plt.xlabel("Recall")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def tune_hyperparameters(X_train, X_val, y_train, y_val, average):
    """
    Find the best RF hyperparameters using validation set.
    Tries all combinations of n_estimators and criterion.
    Selects the combination with the highest F1-score on validation.
    """
    best_score = -1
    best_params = None

    for n_est in PARAM_GRID["n_estimators"]:
        for crit in PARAM_GRID["criterion"]:
            rf = RandomForestClassifier(
                n_estimators=n_est,
                criterion=crit,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            score = f1_score(y_val, y_pred, average=average, zero_division=0)

            if score > best_score:
                best_score = score
                best_params = {"n_estimators": n_est, "criterion": crit}

    return best_params, best_score


def evaluate_model(model, X_test, y_test, average, binary_task=False):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
    }

    if binary_task:
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_negatives"] = int(tn)

    return metrics, y_pred


def run_cross_validation(datasets, label_col, all_label_cols, task_name, average, multi_col=None):
    """
    For each fold:
      1. Each node splits its data into trainval 80%, test 20%.
      2. Trainval is split into train ~70%, val ~10% for tuning.
      3. Best hyperparameters are found on the validation set.
      4. Final model is retrained on all trainval data with best params.
      5. Each model is tested on ALL nodes' test sets + combined "All".

    Normalization: the training node's scaler is used for all test sets.
    normalize(X_trainval, X_test) fits on X_trainval, applies to both.
    """
    print(f"\nTask: {task_name}")

    node_names = list(datasets.keys())
    test_nodes = node_names + ["All"]
    feature_cols = [c for c in datasets[node_names[0]].columns if c not in all_label_cols]

    # Create stratified fold indices per node
    fold_indices = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for node_name, df in datasets.items():
        fold_indices[node_name] = list(skf.split(df, df[label_col]))
        print(f"{node_name}: {len(df)} rows, {N_FOLDS} folds")

    # Collect per-fold metrics
    fold_results = {
        train_node: {test_node: [] for test_node in test_nodes}
        for train_node in node_names
    }

    # Per attack type recall tracking ("All" test set)
    attack_detected = {train_node: {} for train_node in node_names}
    attack_total = {train_node: {} for train_node in node_names}

    for fold in range(N_FOLDS):
        print(f"\nFold {fold + 1}/{N_FOLDS}")

        # Split each node's data into train-val, test
        node_splits = {}
        for node_name, df in datasets.items():
            trainval_idx, test_idx = fold_indices[node_name][fold]
            node_splits[node_name] = {
                "X_trainval": df.iloc[trainval_idx][feature_cols],
                "X_test": df.iloc[test_idx][feature_cols],
                "y_trainval": df.iloc[trainval_idx][label_col],
                "y_test": df.iloc[test_idx][label_col],
                "y_multi_test": df.iloc[test_idx][multi_col] if multi_col else None,
            }

        # Tune, train, and evaluate each node's model
        for train_node in node_names:
            X_trainval = node_splits[train_node]["X_trainval"]
            y_trainval = node_splits[train_node]["y_trainval"]

            # Split train-val
            val_splitter = StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE)
            train_idx, val_idx = next(val_splitter.split(X_trainval, y_trainval))

            X_train_raw = X_trainval.iloc[train_idx]
            X_val_raw = X_trainval.iloc[val_idx]
            y_train = y_trainval.iloc[train_idx]
            y_val = y_trainval.iloc[val_idx]

            # Normalize train/val (fit on train only, not entire train-val)
            X_train_norm, X_val_norm = normalize(X_train_raw, X_val_raw)

            # Tune hyperparameters on validation set
            best_params, val_score = tune_hyperparameters(
                X_train_norm, X_val_norm, y_train, y_val, average
            )

            # Retrain on all train-val data with best params
            # Normalize train-val (fit on train-val)
            X_trainval_norm, _ = normalize(X_trainval, X_trainval)

            model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_trainval_norm, y_trainval)

            print(f"{train_node}: params={best_params}, val F1={val_score:.4f}")

            # Test on each node's test set, normalized with training node's scaler
            all_y_true = []
            all_y_pred = []
            all_y_multi = []

            for test_node in node_names:
                X_test = node_splits[test_node]["X_test"]
                y_test = node_splits[test_node]["y_test"]

                # Normalize test data using training node's scaler
                _, X_test_norm = normalize(X_trainval, X_test)

                metrics, y_pred = evaluate_model(model, X_test_norm, y_test, average, binary_task=(multi_col is not None))
                fold_results[train_node][test_node].append(metrics)

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)

                # Collect multiclass labels for per attack analysis
                if node_splits[test_node]["y_multi_test"] is not None:
                    all_y_multi.extend(node_splits[test_node]["y_multi_test"])

            # "All" = combined predictions across all test nodes
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)

            metrics_all = {
                "accuracy": accuracy_score(all_y_true, all_y_pred),
                "f1": f1_score(all_y_true, all_y_pred, average=average, zero_division=0),
                "precision": precision_score(all_y_true, all_y_pred, average=average, zero_division=0),
                "recall": recall_score(all_y_true, all_y_pred, average=average, zero_division=0),
            }

            if multi_col is not None:
                cm = confusion_matrix(all_y_true, all_y_pred)
                tn, fp, fn, tp = cm.ravel()
                metrics_all["true_positives"] = int(tp)
                metrics_all["false_positives"] = int(fp)
                metrics_all["false_negatives"] = int(fn)
                metrics_all["true_negatives"] = int(tn)

            fold_results[train_node]["All"].append(metrics_all)

            # Per attack type recall (across all test nodes)
            if multi_col is not None and all_y_multi:
                all_y_multi = np.array(all_y_multi)
                for attack_type in np.unique(all_y_multi):
                    if attack_type == "Normal":
                        continue
                    mask = all_y_multi == attack_type
                    total = mask.sum()
                    detected = (all_y_pred[mask] == 1).sum()
                    attack_detected[train_node][attack_type] = \
                        attack_detected[train_node].get(attack_type, 0) + detected
                    attack_total[train_node][attack_type] = \
                        attack_total[train_node].get(attack_type, 0) + total

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    if multi_col is not None:
        for train_node in node_names:
            # Sum TP/FP/FN/TN across folds for this train node's "All" test
            sum_tp = sum(m["true_positives"] for m in fold_results[train_node]["All"])
            sum_fp = sum(m["false_positives"] for m in fold_results[train_node]["All"])
            sum_fn = sum(m["false_negatives"] for m in fold_results[train_node]["All"])
            sum_tn = sum(m["true_negatives"] for m in fold_results[train_node]["All"])
            sum_cm = np.array([[sum_tn, sum_fp], [sum_fn, sum_tp]])

            plot_confusion_matrix(
                sum_cm, ["Normal", "Anomaly"],
                title=f"RF Local: {train_node}",
                save_path=out_dir / f"rf_local_cm_AD_{train_node}.png"
            )

    per_attack_results = {}
    if multi_col is not None:
        for train_node in node_names:
            if attack_total[train_node]:
                per_attack_recall = {}
                for attack_type in sorted(attack_total[train_node].keys()):
                    det = attack_detected[train_node][attack_type]
                    tot = attack_total[train_node][attack_type]
                    r = det / tot
                    per_attack_recall[attack_type] = r

                per_attack_results[train_node] = per_attack_recall

                plot_per_attack_recall(
                    per_attack_recall,
                    title=f"RF Local: {train_node} \nRecall per Attack Type",
                    save_path=out_dir / f"rf_local_recall_per_attack_{train_node}.png"
                )

    avg_results = {}
    for train_node in node_names:
        avg_results[train_node] = {}
        for test_node in test_nodes:
            fold_metrics_list = fold_results[train_node][test_node]
            avg_metrics = {}
            for key in fold_metrics_list[0]:
                values = [m[key] for m in fold_metrics_list]
                avg_metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "per_fold": [float(v) for v in values],
                }
            avg_results[train_node][test_node] = avg_metrics

        if train_node in per_attack_results:
            avg_results[train_node]["per_attack_recall"] = per_attack_results[train_node]

    print(f"\nF1-score (mean over {N_FOLDS} folds)")

    label = "Train \\ Test"
    header = f"{label:<14}"
    for test_node in test_nodes:
        header += f"{test_node:>10}"
    print(header)

    for train_node in node_names:
        row = f"{train_node:<14}"
        for test_node in test_nodes:
            mean_f1 = avg_results[train_node][test_node]["f1"]["mean"]
            row += f"{mean_f1:>10.3f}"
        print(row)

    return avg_results


def main():
    print("\nStep 1: Load and preprocess data")
    datasets, binary_col, multi_col = load_data()
    all_label_cols = [binary_col, multi_col]

    print("\nStep 2: Anomaly Detection (AD)")
    ad_results = run_cross_validation(
        datasets, label_col=binary_col,
        all_label_cols=all_label_cols,
        task_name="Anomaly Detection (AD)",
        average="weighted",
        multi_col=multi_col,
    )

    print("\nStep 3: Anomaly Classification (AC)")
    ac_results = run_cross_validation(
        datasets, label_col=multi_col,
        all_label_cols=all_label_cols,
        task_name="Anomaly Classification (AC)",
        average="weighted",
    )

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    results = {
        "settings": {
            "n_folds": N_FOLDS,
            "param_grid": PARAM_GRID,
            "random_state": RANDOM_STATE,
        },
        "anomaly_detection": ad_results,
        "anomaly_classification": ac_results,
    }

    out_path = out_dir / "rf_local_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
