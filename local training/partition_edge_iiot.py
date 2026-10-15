import pandas as pd
import numpy as np
import os


DATA_FOLDER = "data_edge_IIoT"
INPUT_FILE = f"{DATA_FOLDER}/ML-EdgeIIoT-dataset.csv"

# Attack cap, target approx. 70/30 balance with 24,301 normal
# 24,301 = 70% -> total approx. 34,716 -> attack approx. 10,415 -> 10,415/14 approx. 744 (round to 750)
ATTACK_CAP = 750

# Node assignments, attack types are dominant on each node
NODE_ASSIGNMENTS = {
    "Node1": ["DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP"],
    "Node2": ["Backdoor", "Password", "Ransomware", "MITM"],
    "Node3": ["XSS", "SQL_injection", "Uploading", "Port_Scanning",
              "Fingerprinting", "Vulnerability_scanner"],
}

# Distribution (dominant node gets 80%, each other node gets 10%)
DOMINANT_RATIO = 0.80
SPILLOVER_RATIO = 0.10

RANDOM_SEED = 42


def subsample_attacks(df, cap, seed):
    normal = df[df["Attack_type"] == "Normal"].copy()
    attacks = df[df["Attack_type"] != "Normal"].copy()

    subsampled_attacks = []
    print(f"\nSubsampling attack classes (cap={cap}):")
    print(f"{'Attack type':<30} {'Original':>10} {'After':>10}")

    for attack_type in sorted(attacks["Attack_type"].unique()):
        subset = attacks[attacks["Attack_type"] == attack_type]
        original_count = len(subset)

        if original_count > cap:
            subset = subset.sample(n=cap, random_state=seed)

        print(f"{attack_type:<30} {original_count:>10,} {len(subset):>10,}")
        subsampled_attacks.append(subset)

    attacks_combined = pd.concat(subsampled_attacks, ignore_index=True)

    print(f"\n{'Normal':<30} {len(normal):>10,} {len(normal):>10,}")
    print(f"{'Total':<30} {len(df):>10,} {len(normal) + len(attacks_combined):>10,}")
    print(f"Balance: {len(normal)/(len(normal)+len(attacks_combined))*100:.1f}% normal / "
          f"{len(attacks_combined)/(len(normal)+len(attacks_combined))*100:.1f}% attack")

    return pd.concat([normal, attacks_combined], ignore_index=True)


def partition_nodes(df, node_assignments, dominant_ratio, spillover_ratio, seed):
    # Split normal traffic evenly across nodes
    normal = df[df["Attack_type"] == "Normal"].copy()
    normal_shuffled = normal.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(normal_shuffled)
    third = n // 3
    normal_splits = {
        "Node1": normal_shuffled.iloc[:third],
        "Node2": normal_shuffled.iloc[third:2*third],
        "Node3": normal_shuffled.iloc[2*third:],
    }

    dominant_node = {}
    for node, attacks in node_assignments.items():
        for attack in attacks:
            dominant_node[attack] = node

    # Split each attack class
    node_frames = {node: [normal_splits[node]] for node in node_assignments}

    attacks = df[df["Attack_type"] != "Normal"]
    for attack_type in sorted(attacks["Attack_type"].unique()):
        subset = attacks[attacks["Attack_type"] == attack_type].copy()
        subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)

        total = len(subset)
        dom_node = dominant_node[attack_type]

        # Calculate split sizes
        dom_count = int(total * dominant_ratio)
        spill_count = int(total * spillover_ratio)

        # Get the two non-dominant nodes
        other_nodes = [n for n in node_assignments if n != dom_node]

        # Assign rows
        idx = 0
        node_frames[dom_node].append(subset.iloc[idx:idx + dom_count])
        idx += dom_count

        for other in other_nodes:
            node_frames[other].append(subset.iloc[idx:idx + spill_count])
            idx += spill_count

        # Any remainder goes to dominant node
        if idx < total:
            node_frames[dom_node].append(subset.iloc[idx:])

    # Concatenate frames per node
    nodes = {}
    for node_name, frames in node_frames.items():
        nodes[node_name] = pd.concat(frames, ignore_index=True)

    return nodes


def print_summary(nodes, df_original):
    all_attacks = sorted([a for a in df_original["Attack_type"].unique() if a != "Normal"])

    print(f"\nPartition Summary")

    print(f"Node sizes:")
    total = 0
    for name, df in nodes.items():
        dominant_info = ", ".join(NODE_ASSIGNMENTS[name])
        print(f"{name} ({dominant_info}): {len(df):,} rows")
        total += len(df)
    print(f"{'Total':<50} {total:,} rows")

    # Attack distribution table
    print(f"\nAttack type distribution per node:")
    print(f"{'Attack type':<30} {'Node1':>8} {'Node2':>8} {'Node3':>8} {'Total':>8}")

    grand_total = {name: 0 for name in nodes}
    for attack in ["Normal"] + all_attacks:
        counts = {}
        row_total = 0
        for name, df in nodes.items():
            count = (df["Attack_type"] == attack).sum()
            counts[name] = count
            grand_total[name] += count
            row_total += count
        print(f"{attack:<30} {counts['Node1']:>8,} {counts['Node2']:>8,} {counts['Node3']:>8,} {row_total:>8,}")

    print(f"{'Total':<30} {grand_total['Node1']:>8,} {grand_total['Node2']:>8,} {grand_total['Node3']:>8,} "
          f"{sum(grand_total.values()):>8,}")

    # Binary balance per node
    print(f"\nBinary label (Normal vs Attack) per node:")
    print(f"{'Node':<10} {'Normal':>8} {'Attack':>8} {'Anomaly%':>10}")
    for name, df in nodes.items():
        n_normal = (df["Attack_label"] == 0).sum()
        n_attack = (df["Attack_label"] == 1).sum()
        ratio = n_attack / len(df) * 100
        print(f"{name:<10} {n_normal:>8,} {n_attack:>8,} {ratio:>9.1f}%")


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"\nLoaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Verify expected classes
    expected_classes = (
        ["Normal"] +
        NODE_ASSIGNMENTS["Node1"] +
        NODE_ASSIGNMENTS["Node2"] +
        NODE_ASSIGNMENTS["Node3"]
    )
    actual_classes = sorted(df["Attack_type"].unique())
    assert set(expected_classes) == set(actual_classes), \
        f"Class mismatch!\nExpected: {sorted(expected_classes)}\nActual: {actual_classes}"
    print(f"All {len(expected_classes)} expected classes found.")

    df_balanced = subsample_attacks(df, ATTACK_CAP, RANDOM_SEED)

    nodes = partition_nodes(df_balanced, NODE_ASSIGNMENTS, DOMINANT_RATIO, SPILLOVER_RATIO, RANDOM_SEED)

    print_summary(nodes, df_balanced)

    total_rows = sum(len(df) for df in nodes.values())
    print(f"\nData integrity check:")
    print(f"Total rows across all nodes: {total_rows:,}")

    print(f"\nSaving partitions:")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    for name, node_df in nodes.items():
        path = f"{DATA_FOLDER}/output_{name.lower()}.csv"
        node_df.to_csv(path, index=False)
        print(f"{name}: {path} ({len(node_df):,} rows)")


if __name__ == "__main__":
    main()
