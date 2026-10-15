import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
INPUT_FILE = BASE_DIR / "ML-EdgeIIoT-dataset.csv"

ATTACK_CAP = 750
N_NODES = 3
RANDOM_SEED = 42

OUTPUT_FILES = {
    "Node1": DATA_FOLDER / "output_node1.csv",
    "Node2": DATA_FOLDER / "output_node2.csv",
    "Node3": DATA_FOLDER / "output_node3.csv",
}

COLUMNS_TO_REMOVE = [
    "ip.src_host",
    "ip.dst_host",
    "frame.time",
    "frame.time_epoch",
    "http.file_data",
    "http.request.full_uri",
    "http.request.uri.query",
    "http.request.method",
    "http.referer",
    "http.request.version",
    "tcp.payload",
    "tcp.options",
    "mqtt.msg",
    "mqtt.protoname",
    "mqtt.topic",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
]

FORCE_NUMERIC = [
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "dns.qry.name.len",
    "mqtt.conack.flags",
]


def subsample_attacks(df, cap, seed):
    normal = df[df["Attack_type"] == "Normal"].copy()
    attacks = df[df["Attack_type"] != "Normal"].copy()

    subsampled_attacks = []
    print(f"\nSubsampling attack classes (cap={cap}):")
    print(f"{'Attack type':<30} {'Original':>10} {'After':>10}")

    for attack_type in sorted(attacks["Attack_type"].unique()):
        subset = attacks[attacks["Attack_type"] == attack_type].copy()
        original_count = len(subset)

        if original_count > cap:
            subset = subset.sample(n=cap, random_state=seed)

        print(f"{attack_type:<30} {original_count:>10,} {len(subset):>10,}")
        subsampled_attacks.append(subset)

    attacks_combined = pd.concat(subsampled_attacks, ignore_index=True)

    print(f"\n{'Normal':<30} {len(normal):>10,} {len(normal):>10,}")
    print(f"{'Total':<30} {len(df):>10,} {len(normal) + len(attacks_combined):>10,}")

    total_after = len(normal) + len(attacks_combined)
    print(
        f"Balance after subsampling: "
        f"{len(normal) / total_after * 100:.1f}% normal / "
        f"{len(attacks_combined) / total_after * 100:.1f}% attack"
    )

    return pd.concat([normal, attacks_combined], ignore_index=True)


def prepare_features_for_partition(df):
    df = df.copy()

    existing_remove = [c for c in COLUMNS_TO_REMOVE if c in df.columns]
    df = df.drop(columns=existing_remove, errors="ignore")

    for col in FORCE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = [c for c in df.columns if c not in ["Attack_label", "Attack_type"]]
    X = df[feature_cols].copy()

    remaining_cat = X.select_dtypes(include=["object"]).columns.tolist()
    if remaining_cat:
        X = pd.get_dummies(X, columns=remaining_cat)

    X = X.fillna(0)

    return df, X


def create_feature_skew_nodes(df, seed=42, n_nodes=3):
    """
    Skapar fasta Node1, Node2, Node3 med feature skew:
    1. Klustrar endast normal trafik i feature-rummet.
    2. Tilldelar attacker till närmaste normala kluster.
    """
    if n_nodes != 3:
        raise ValueError("Den här versionen är byggd för exakt 3 noder.")

    df_clean, X_all = prepare_features_for_partition(df)

    normal_mask = df_clean["Attack_type"] == "Normal"
    attack_mask = ~normal_mask

    X_normal = X_all.loc[normal_mask].copy()
    X_attack = X_all.loc[attack_mask].copy()

    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_attack_scaled = scaler.transform(X_attack)

    kmeans = KMeans(n_clusters=n_nodes, random_state=seed, n_init=10)
    normal_clusters = kmeans.fit_predict(X_normal_scaled)

    cluster_to_node = {
        0: "Node1",
        1: "Node2",
        2: "Node3",
    }

    nodes = {name: [] for name in cluster_to_node.values()}

    # Lägg normal trafik i respektive nod
    normal_df = df_clean.loc[normal_mask].copy().reset_index(drop=True)
    for cluster_id, node_name in cluster_to_node.items():
        idx = np.where(normal_clusters == cluster_id)[0]
        cluster_rows = normal_df.iloc[idx].copy()
        nodes[node_name].append(cluster_rows)

    # Tilldela attacker till närmaste normala kluster
    attack_df = df_clean.loc[attack_mask].copy().reset_index(drop=True)
    attack_distances = kmeans.transform(X_attack_scaled)
    attack_clusters = np.argmin(attack_distances, axis=1)

    for cluster_id, node_name in cluster_to_node.items():
        idx = np.where(attack_clusters == cluster_id)[0]
        cluster_rows = attack_df.iloc[idx].copy()
        nodes[node_name].append(cluster_rows)

    final_nodes = {}
    for node_name, frames in nodes.items():
        final_nodes[node_name] = (
            pd.concat(frames, ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    return final_nodes


def print_summary(nodes):
    print("\nPartition Summary")
    print(f"{'Node':<10} {'Rows':>10} {'Normal':>10} {'Attack':>10} {'Anomaly%':>10}")

    for name, df in nodes.items():
        n_normal = (df["Attack_label"] == 0).sum()
        n_attack = (df["Attack_label"] == 1).sum()
        ratio = n_attack / len(df) * 100 if len(df) > 0 else 0
        print(f"{name:<10} {len(df):>10,} {n_normal:>10,} {n_attack:>10,} {ratio:>9.1f}%")

    print("\nAttack type distribution per node:")
    all_types = sorted(set().union(*[set(df["Attack_type"].unique()) for df in nodes.values()]))

    header = f"{'Attack type':<30} {'Node1':>8} {'Node2':>8} {'Node3':>8} {'Total':>8}"
    print(header)

    for attack in all_types:
        counts = []
        total = 0
        for node in ["Node1", "Node2", "Node3"]:
            count = int((nodes[node]["Attack_type"] == attack).sum())
            counts.append(count)
            total += count
        print(f"{attack:<30} {counts[0]:>8,} {counts[1]:>8,} {counts[2]:>8,} {total:>8,}")


def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Kunde inte hitta datasetfilen:\n{INPUT_FILE}\n\n"
            f"Lägg filen 'ML-EdgeIIoT-dataset.csv' i projektmappen:\n{BASE_DIR}"
        )


def main():
    check_input_file()

    print(f"Reading dataset from:\n{INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"\nLoaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    required_cols = {"Attack_label", "Attack_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset saknar obligatoriska kolumner: {missing}")

    df_balanced = subsample_attacks(df, ATTACK_CAP, RANDOM_SEED)

    nodes = create_feature_skew_nodes(df_balanced, seed=RANDOM_SEED, n_nodes=N_NODES)

    print_summary(nodes)

    total_rows = sum(len(x) for x in nodes.values())
    print(f"\nTotal rows across all nodes: {total_rows:,}")

    DATA_FOLDER.mkdir(exist_ok=True)

    print("\nSaving partitions:")
    for name, node_df in nodes.items():
        path = OUTPUT_FILES[name]
        node_df.to_csv(path, index=False)
        print(f"{name}: {path} ({len(node_df):,} rows)")

    print("\nDone. Created files:")
    for name, path in OUTPUT_FILES.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()