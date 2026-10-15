import json
import pickle
import numpy as np
import pandas as pd
import flwr as fl

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def load_config(path="client_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def parameters_to_models(parameters):
    if parameters is None:
        return []
    ndarrays = fl.common.parameters_to_ndarrays(parameters)
    if not ndarrays:
        return []
    model_bytes = ndarrays[0].tobytes()
    return pickle.loads(model_bytes)


class FederatedTreeClient(fl.client.NumPyClient):
    def __init__(self):
        self.cfg = load_config()
        self.global_models = []
        self.model_type = self.cfg["model_type"]

        df = pd.read_csv(self.cfg["data_file"])

        drop_cols = self.cfg.get("drop_columns", [])
        existing = [c for c in drop_cols if c in df.columns]

        X = df.drop(existing, axis=1).values

        if self.model_type == "random_forest":
            self.label_column = self.cfg["label_column_rf"]
        elif self.model_type == "isolation_forest":
            self.label_column = self.cfg["label_column_if"]
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        y = df[self.label_column].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.cfg.get("test_size", 0.2),
            random_state=self.cfg.get("random_state", 42),
            stratify=y if self.model_type == "random_forest" else None,
        )

        self.model = None

    def _build_model(self):
        if self.model_type == "random_forest":
            rf_cfg = self.cfg["rf"]
            return RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 100),
                criterion=rf_cfg.get("criterion", "entropy"),
                random_state=self.cfg.get("random_state", 42),
                n_jobs=rf_cfg.get("n_jobs", -1),
            )

        if self.model_type == "isolation_forest":
            if_cfg = self.cfg["iforest"]
            return IsolationForest(
                n_estimators=if_cfg.get("n_estimators", 100),
                contamination=if_cfg.get("contamination", "auto"),
                random_state=if_cfg.get("random_state", 42),
            )

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _get_confusion_labels(self, y_true, preds):
        if self.model_type == "random_forest":
            return sorted(set(np.asarray(y_true).astype(str).tolist()) |
                        set(np.asarray(preds).astype(str).tolist()))

        elif self.model_type == "isolation_forest":
            return [-1, 1]

        return sorted(set(np.asarray(y_true).tolist()) |    
                    set(np.asarray(preds).tolist()))

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        try:
            if parameters:
                self.global_models = parameters_to_models(
                    fl.common.ndarrays_to_parameters(parameters)
                )
                print(
                    f"[{self.cfg['client_name']}] Received global ensemble with "
                    f"{len(self.global_models)} models"
                )
        except Exception as e:
            print(f"[{self.cfg['client_name']}] Could not load global ensemble in fit: {e}")

        self.model = self._build_model()

        if self.model_type == "random_forest":
            self.model.fit(self.X_train, self.y_train)

        elif self.model_type == "isolation_forest":
            unique_labels = set(np.unique(self.y_train))
            if unique_labels.issubset({0, 1}):
                normal_mask = self.y_train == 1
                X_train_normal = self.X_train[normal_mask]
                if len(X_train_normal) == 0:
                    X_train_normal = self.X_train
                self.model.fit(X_train_normal)
            else:
                self.model.fit(self.X_train)

        return [], len(self.X_train), {
            "model": pickle.dumps(self.model),
            "model_type": self.model_type,
        }

    def evaluate(self, parameters, config):
        try:
            if parameters:
                self.global_models = parameters_to_models(
                    fl.common.ndarrays_to_parameters(parameters)
                )
                print(
                    f"[{self.cfg['client_name']}] Evaluating global ensemble with "
                    f"{len(self.global_models)} models"
                )
        except Exception as e:
            print(f"[{self.cfg['client_name']}] Could not load global ensemble in evaluate: {e}")

        if self.model_type == "random_forest":
            if self.global_models:
                probas = [model.predict_proba(self.X_test) for model in self.global_models]
                avg_proba = np.mean(probas, axis=0)
                classes = self.global_models[0].classes_
                preds = classes[np.argmax(avg_proba, axis=1)]
            else:
                preds = self.model.predict(self.X_test)

            # Make all RF labels strings
            y_true = self.y_test.astype(str)
            preds = preds.astype(str)

        elif self.model_type == "isolation_forest":
            if self.global_models:
                ensemble_preds = np.array([model.predict(self.X_test) for model in self.global_models])
                vote_sum = np.sum(ensemble_preds, axis=0)
                preds = np.where(vote_sum >= 0, 1, -1)
            else:
                preds = self.model.predict(self.X_test)

            # 0=normal, 1=anomaly  ->  1=normal, -1=anomaly
            y_true = np.where(self.y_test == 1, 1, -1)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        labels = self._get_confusion_labels(y_true, preds)
        print(f"[{self.cfg['client_name']}] unique y_true: {sorted(set(np.asarray(y_true).tolist()))}")
        print(f"[{self.cfg['client_name']}] unique preds: {sorted(set(np.asarray(preds).tolist()))}")
        print(f"[{self.cfg['client_name']}] labels used: {labels}")
        cm = confusion_matrix(y_true, preds, labels=labels)

        acc = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, average="weighted", zero_division=0)
        recall = recall_score(y_true, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_true, preds, average="weighted", zero_division=0)

        print(
            f"[{self.cfg['client_name']}] "
            f"accuracy: {acc:.4f}, precision: {precision:.4f}, "
            f"recall: {recall:.4f}, f1: {f1:.4f}"
        )
        print(f"[{self.cfg['client_name']}] Confusion matrix labels: {labels}")
        print(f"[{self.cfg['client_name']}] Confusion matrix:\n{cm}")

        return 0.0, len(self.X_test), {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": pickle.dumps(cm),
        }


if __name__ == "__main__":
    cfg = load_config()
    fl.client.start_numpy_client(
        server_address=cfg["server_address"],
        client=FederatedTreeClient(),
    )