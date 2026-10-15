import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# STEP 1: Remove unnecessary columns
def remove_unnecessary_columns(df, config):
    columns_to_remove = config["columns_to_remove"]
    existing = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    print(f"  Deleted {len(existing)} columns: {existing}")
    return df


# STEP 2: Select labels and merge specified classes with Normal
def prepare_labels(df, config):
    binary_col = config["binary_label"]
    multi_col = config["multiclass_label"]

    # Merge specified classes to Normal
    merged_count = 0
    for class_name in config["merge_to_normal"]:
        count = (df[multi_col] == class_name).sum()
        merged_count += count
        df[multi_col] = df[multi_col].replace(class_name, "Normal")

    if merged_count > 0:
        print(f"  Changed {merged_count} instances to 'Normal'")
        # Update binary label for merged rows
        if df[binary_col].dtype in [int, np.int64, float, np.float64]:
            df.loc[df[multi_col] == "Normal", binary_col] = 0
        else:
            df.loc[df[multi_col] == "Normal", binary_col] = "Normal"

    # Remove unused label columns
    labels_to_drop = [col for col in config["labels_to_drop"] if col in df.columns]
    df = df.drop(columns=labels_to_drop, errors="ignore")

    return df, binary_col, multi_col


# STEP 3: One-hot encoding of categorical columns
def one_hot_encode(df, label_columns, config):
    feature_cols = [col for col in df.columns if col not in label_columns]
    labels_df = df[label_columns].copy()
    features_df = df[feature_cols].copy()

    categorical_config = config["categorical_columns"]
    encoded_cols = []

    for col, categories in categorical_config.items():
        if col in features_df.columns:
            encoded_cols.append(col)
            for cat in categories:
                features_df[f"{col}_{cat}"] = (features_df[col] == cat).astype(float)
            features_df = features_df.drop(columns=[col])
        # Force columns that should be numeric to numeric type
    force_numeric = [
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "dns.qry.name.len",
        "mqtt.conack.flags",
    ]

    for col in force_numeric:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    # Handle any remaining categorical columns not in config
    remaining_cat = features_df.select_dtypes(include=["object"]).columns.tolist()
    if remaining_cat:
        encoded_cols.extend(remaining_cat)
        features_df = pd.get_dummies(features_df, columns=remaining_cat).astype(float)

    print(f"  One-hot encoding: {encoded_cols}")

    df_encoded = pd.concat([features_df, labels_df], axis=1)
    return df_encoded


# STEP 4: Fill in missing values with 0
def fill_missing_values(df, label_columns):
    feature_cols = [col for col in df.columns if col not in label_columns]
    missing = df[feature_cols].isnull().sum().sum()
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"  Filled {missing} missing values with 0")
    return df


# Run all preprocessing on a single node's dataset
def preprocess_single(df, node_name, config):
    print(f"\n  Preprocesses {node_name} ({df.shape[0]} rows, {df.shape[1]} columns)")

    df = remove_unnecessary_columns(df, config)
    df, binary_col, multi_col = prepare_labels(df, config)
    label_columns = [binary_col, multi_col]
    df = one_hot_encode(df, label_columns, config)
    df = fill_missing_values(df, label_columns)

    print(f"  Result: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, binary_col, multi_col


def load_and_preprocess(config):
    datasets_raw = {}
    binary_col = None
    multi_col = None

    #  Läs och gör basic cleaning per nod (utan one-hot)
    for node_name, file_path in config["files"].items():
        df = pd.read_csv(file_path, low_memory=False)
        df = remove_unnecessary_columns(df, config)
        df, binary_col, multi_col = prepare_labels(df, config)
        datasets_raw[node_name] = df

    # Concat alla noder
    all_data = pd.concat(datasets_raw.values(), ignore_index=True)

    label_columns = [binary_col, multi_col]

    # One-hot encoding EN gång på hela datan
    all_data = one_hot_encode(all_data, label_columns, config)

    # Fill missing
    all_data = fill_missing_values(all_data, label_columns)

    # Split tillbaka per nod
    datasets = {}
    start = 0

    for node_name, df_raw in datasets_raw.items():
        length = len(df_raw)
        datasets[node_name] = all_data.iloc[start:start+length].copy()
        start += length

    return datasets, binary_col, multi_col


# Normalization: Min-Max normalization to [0, 1], calculates min/max only from training data
# Applies to both training and testing. Binary columns (0/1) are left unchanged
def normalize(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    binary_cols = []
    non_binary_cols = []

    for col in X_train.columns:
        unique_vals = X_train[col].dropna().unique()
        if set(unique_vals).issubset({0.0, 1.0, 0, 1}):
            binary_cols.append(col)
        else:
            non_binary_cols.append(col)

    scaler = MinMaxScaler(feature_range=(0, 1))

    if non_binary_cols:
        scaler.fit(X_train[non_binary_cols])
        X_train[non_binary_cols] = scaler.transform(X_train[non_binary_cols])
        X_test[non_binary_cols] = scaler.transform(X_test[non_binary_cols])

    return X_train, X_test


def main():
    import os
    from config_westermo import WESTERMO_CONFIG

    datasets, binary_col, multi_col = load_and_preprocess(WESTERMO_CONFIG)
    label_columns = [binary_col, multi_col]

    for node_name, df in datasets.items():
        feature_count = df.shape[1] - len(label_columns)

        print(f"\n{'-'*60}")
        print(f"  Summary: {node_name}")
        print(f"{'-' * 60}")
        print(f"  Number of rows:    {df.shape[0]}")
        print(f"  Number columns: {df.shape[1]}")
        print(f"  Number of features: {feature_count}")
        print(f"  Number of labels:   {len(label_columns)}")

        print(f"\n  Class division ({multi_col}):")
        for klass, antal in df[multi_col].value_counts().items():
            pct = antal / len(df) * 100
            print(f"    {klass:<30} {antal:>6} ({pct:.1f}%)")

    all_data = pd.concat(datasets.values(), ignore_index=True)
    all_data = all_data.fillna(0)

    print(f"\n{'-'*60}")
    print(f"  Merged (all nodes)")
    print(f"{'-' * 60}")
    print(f"  Total number of rows: {all_data.shape[0]}")
    print(f"\n  Class division ({multi_col}):")
    for klass, antal in all_data[multi_col].value_counts().items():
        pct = antal / len(all_data) * 100
        print(f"    {klass:<30} {antal:>6} ({pct:.1f}%)")

    # Save preprocessed files
    output_folder = "data_preprocessed"
    os.makedirs(output_folder, exist_ok=True)

    for node_name, df in datasets.items():
        output_path = f"{output_folder}/{node_name.lower()}_preprocessed.csv"
        df.to_csv(output_path, index=False)
        print(f"\n  Saved: {output_path}")

    print("\nPreprocessing finished!")


if __name__ == "__main__":
    main()
