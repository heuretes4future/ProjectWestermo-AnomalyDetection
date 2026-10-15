import flwr as fl
import pickle
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def load_config(path="server_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def models_to_parameters(models):
    model_bytes = pickle.dumps(models)
    ndarray = np.frombuffer(model_bytes, dtype=np.uint8)
    return fl.common.ndarrays_to_parameters([ndarray])


class EnsembleStrategy(fl.server.strategy.Strategy):
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_models = []
        self.model_type = cfg["model_type"]
        self.num_clients = cfg["num_clients"]
        self.max_ensemble_size = cfg.get("max_ensemble_size", 10)
        self.save_dir = cfg.get("save_dir", "saved_result")

    def initialize_parameters(self, client_manager):
        return models_to_parameters(self.global_models)

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=self.num_clients)
        fit_config = {"model_type": self.model_type}
        return [(c, fl.common.FitIns(parameters, fit_config)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return models_to_parameters(self.global_models), {}

        models = []
        for _, fit_res in results:
            model = pickle.loads(fit_res.metrics["model"])
            models.append(model)

        self.global_models.extend(models)

        if len(self.global_models) > self.max_ensemble_size:
            self.global_models = self.global_models[-self.max_ensemble_size:]

        os.makedirs(self.save_dir, exist_ok=True)
        with open(
            f"{self.save_dir}/{self.model_type}_ensemble_round_{server_round}.pkl",
            "wb",
        ) as f:
            pickle.dump(self.global_models, f)

        print(
            f"[SERVER] Round {server_round}: saved {len(models)} new models, "
            f"ensemble size={len(self.global_models)}, model_type={self.model_type}"
        )

        return models_to_parameters(self.global_models), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=self.num_clients)
        eval_config = {"model_type": self.model_type}
        return [(c, fl.common.EvaluateIns(parameters, eval_config)) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_examples = sum(eval_res.num_examples for _, eval_res in results)

        weighted_accuracy = sum(
            eval_res.metrics["accuracy"] * eval_res.num_examples
            for _, eval_res in results
        ) / total_examples

        weighted_precision = sum(
            eval_res.metrics["precision"] * eval_res.num_examples
            for _, eval_res in results
        ) / total_examples

        weighted_recall = sum(
            eval_res.metrics["recall"] * eval_res.num_examples
            for _, eval_res in results
        ) / total_examples

        weighted_f1 = sum(
            eval_res.metrics["f1"] * eval_res.num_examples
            for _, eval_res in results
        ) / total_examples

        global_cm = None
        for _, eval_res in results:
            cm = pickle.loads(eval_res.metrics["confusion_matrix"])
            global_cm = cm if global_cm is None else global_cm + cm

        print(
            f"[SERVER] Round {server_round} ({self.model_type}) -> "
            f"accuracy: {weighted_accuracy:.4f}, "
            f"precision: {weighted_precision:.4f}, "
            f"recall: {weighted_recall:.4f}, "
            f"f1: {weighted_f1:.4f}"
        )

        os.makedirs(self.save_dir, exist_ok=True)

        if global_cm.shape == (7, 7):
            labels = [
                "BAD-MISCONF",
                "BAD-MISCONF-DUPLICATION",
                "BAD-MITM",
                "BAD-PORTSCAN1",
                "BAD-PORTSCAN2",
                "BAD-SSH",
                "Normal",
            ]
        else:
            labels = ["Anomaly", "Normal"]

        cm_norm = global_cm.astype(float) / global_cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, cmap="Blues")

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{self.model_type} Confusion Matrix - Round {server_round}")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{self.model_type}_round_{server_round}_confusion_matrix.png"
        )
        plt.close()

        return 0.0, {
            "accuracy": float(weighted_accuracy),
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
        }

    def evaluate(self, server_round, parameters):
        return None


if __name__ == "__main__":
    cfg = load_config()
    fl.server.start_server(
        server_address=cfg["server_address"],
        strategy=EnsembleStrategy(cfg),
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
    )