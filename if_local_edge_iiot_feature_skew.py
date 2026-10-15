import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.cluster import KMeans

from preprocessing import load_and_preprocess, normalize
from Edgeiiot_config import EDGEIIOT_CONFIG


RANDOM_STATE = 42
N_FOLDS = 5


# max_samples=256 is the recommended default from the original IF paper (Liu et al., 2008)
IF_PARAMS = {
    "n_estimators": 100,
    "max_samples": 256,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_data():
    datasets, binary_col, multi_col = load_and_preprocess(EDGEIIOT_CONFIG)
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


def train_and_evaluate(model, X_test, y_test):
    anomaly_scores = model.decision_function(X_test)

    # Adaptive threshold via KMeans
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(anomaly_scores.reshape(-1, 1))
    threshold = np.mean(kmeans.cluster_centers_)
    y_pred = np.array([1 if score < threshold else 0 for score in anomaly_scores])

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }

    return metrics, y_pred


def run_cross_validation(datasets, label_col, all_label_cols, task_name, multi_col):
    """
    For each fold:
      1. Each node splits its data into train-val 80%, test 20%.
      2. IF is trained on train-val.
      3. Each model is tested on all nodes' test sets + combined "All".

    Normalization: the training node's scaler is used for all test sets.
    """

    node_names = list(datasets.keys())
    test_nodes = node_names + ["All"]
    feature_cols = [c for c in datasets[node_names[0]].columns if c not in all_label_cols]

    # Create stratified fold indices per node
    fold_indices = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for node_name, df in datasets.items():
        fold_indices[node_name] = list(skf.split(df, df[label_col]))
        print(f"{node_name}: {len(df)} rows, {N_FOLDS} folds")

    # Collect per fold metrics
    fold_results = {
        train_node: {test_node: [] for test_node in test_nodes}
        for train_node in node_names
    }

    # Per attack type recall tracking
    attack_detected = {train_node: {} for train_node in node_names}
    attack_total = {train_node: {} for train_node in node_names}

    for fold in range(N_FOLDS):
        print(f"\nFold {fold + 1}/{N_FOLDS}")

        # Split each node's data into train-val/test
        node_splits = {}
        for node_name, df in datasets.items():
            trainval_idx, test_idx = fold_indices[node_name][fold]
            node_splits[node_name] = {
                "X_trainval": df.iloc[trainval_idx][feature_cols],
                "X_test": df.iloc[test_idx][feature_cols],
                "y_trainval": df.iloc[trainval_idx][label_col],
                "y_test": df.iloc[test_idx][label_col],
                "y_multi_test": df.iloc[test_idx][multi_col],
            }

        # Train and evaluate each node's model
        for train_node in node_names:
            X_trainval = node_splits[train_node]["X_trainval"]
            y_trainval = node_splits[train_node]["y_trainval"]

            mask_normal = (y_trainval == 0)
            X_trainval_normal = X_trainval[mask_normal]

            # Normalize train-val (fit only on the normal train-val data)
            X_trainval_norm, _ = normalize(X_trainval_normal, X_trainval_normal)

            model = IsolationForest(**IF_PARAMS)
            model.fit(X_trainval_norm)

            print(f"{train_node}: trained IF on {len(X_trainval_normal)} NORMAL rows")

            # Test on each node's test set, normalized with training node's scaler
            all_y_true = []
            all_y_pred = []
            all_y_multi = []

            for test_node in node_names:
                X_test = node_splits[test_node]["X_test"]
                y_test = node_splits[test_node]["y_test"]

                # Normalize test data using the scaler fitted on the normal data
                _, X_test_norm = normalize(X_trainval_normal, X_test)

                metrics, y_pred = train_and_evaluate(model, X_test_norm, y_test)
                fold_results[train_node][test_node].append(metrics)

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                all_y_multi.extend(node_splits[test_node]["y_multi_test"])

            # "All" = combined predictions across all test nodes
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)

            cm = confusion_matrix(all_y_true, all_y_pred)
            tn, fp, fn, tp = cm.ravel()

            metrics_all = {
                "accuracy": accuracy_score(all_y_true, all_y_pred),
                "f1": f1_score(all_y_true, all_y_pred, average="binary", zero_division=0),
                "precision": precision_score(all_y_true, all_y_pred, average="binary", zero_division=0),
                "recall": recall_score(all_y_true, all_y_pred, average="binary", zero_division=0),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }

            fold_results[train_node]["All"].append(metrics_all)

            # Per attack type recall
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

    # Confusion matrices per train node on "All" test
    for train_node in node_names:
        sum_tp = sum(m["true_positives"] for m in fold_results[train_node]["All"])
        sum_fp = sum(m["false_positives"] for m in fold_results[train_node]["All"])
        sum_fn = sum(m["false_negatives"] for m in fold_results[train_node]["All"])
        sum_tn = sum(m["true_negatives"] for m in fold_results[train_node]["All"])
        sum_cm = np.array([[sum_tn, sum_fp], [sum_fn, sum_tp]])

        plot_confusion_matrix(
            sum_cm, ["Normal", "Anomaly"],
            title=f"IF Local: {train_node}",
            save_path=out_dir / f"if_local_cm_AD_edge_iiot_{train_node}.png"
        )

    per_attack_results = {}
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
                title=f"IF Local: {train_node}\nRecall per attack type",
                save_path=out_dir / f"if_local_recall_per_attack_edge_iiot_{train_node}.png"
            )

    # Compute averages over all folds
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

    print(f"\nF1-score matrix")

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
        multi_col=multi_col,
    )

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    results = {
        "settings": {
            "n_folds": N_FOLDS,
            "if_params": {k: v for k, v in IF_PARAMS.items() if k != "n_jobs"},
            "random_state": RANDOM_STATE,
        },
        "anomaly_detection": ad_results,
    }

    out_path = out_dir / "if_local_results_edge_iiot.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()