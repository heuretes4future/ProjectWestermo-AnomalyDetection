import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("runs/if_local_results_edge_iiot_feature_skew.json") as f:
    data = json.load(f)

attacks = data["anomaly_detection"]["client_0"]["per_attack_recall"].keys()

matrix = []

for client in ["client_0","client_1","client_2"]:
    recalls = data["anomaly_detection"][client]["per_attack_recall"]
    matrix.append([recalls[a]*100 for a in attacks])

df = pd.DataFrame(matrix, columns=attacks,
                  index=["Train: Node1","Train: Node2","Train: Node3"]).T

plt.figure(figsize=(8,10))
sns.heatmap(df, annot=True, cmap="RdYlGn", vmin=0, vmax=100)

plt.title("IF Local: Edge-IIoTset\nPer-Attack Recall (%) Feature Skew")
plt.ylabel("")
plt.tight_layout()
plt.show()