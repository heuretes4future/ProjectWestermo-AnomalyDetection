import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config_westermo import WESTERMO_CONFIG
#from config_edge_iiot import EDGE_IIOT_CONFIG


# STEP 1: Remove unnecessary columns
def remove_unnecessary_columns(df, config):
    columns_to_remove = config["columns_to_remove"]
    existing = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    print(f"Deleted {len(existing)} columns: {existing}")
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
        print(f"Changed {merged_count} instances to 'Normal'")
        # Update binary label for merged rows
        if df[binary_col].dtype in [int, np.int64, float, np.float64]:
            df.loc[df[multi_col] == "Normal", binary_col] = 0
        else:
            df.loc[df[multi_col] == "Normal", binary_col] = "Normal"

    # Remove unused label columns
    labels_to_drop = [col for col in config["labels_to_drop"] if col in df.columns]
    df = df.drop(columns=labels_to_drop, errors="ignore")

    return df, binary_col, multi_col


# STEP 3: Force mixed-type columns to numeric
def force_numeric_columns(df, config):
    force_numeric = config.get("force_numeric", [])
    converted = []

    for col in force_numeric:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted.append(col)

    return df


# STEP 4: One-hot encoding of categorical columns
def one_hot_encode(df, label_columns, config):
    feature_cols = [col for col in df.columns if col not in label_columns]
    labels_df = df[label_columns].copy()
    features_df = df[feature_cols].copy()

    categorical_config = config["categorical_columns"]
    encoded_cols = []

    # Explicit encoding for columns with known categories
    for col, categories in categorical_config.items():
        if col in features_df.columns:
            encoded_cols.append(col)
            for cat in categories:
                features_df[f"{col}_{cat}"] = (features_df[col] == cat).astype(float)
            features_df = features_df.drop(columns=[col])

    # Handle any remaining categorical columns not in config
    remaining_cat = features_df.select_dtypes(include=["object"]).columns.tolist()
    if remaining_cat:
        encoded_cols.extend(remaining_cat)
        features_df = pd.get_dummies(features_df, columns=remaining_cat).astype(float)

    print(f"One-hot encoding: {encoded_cols}")

    df_encoded = pd.concat([features_df, labels_df], axis=1)
    return df_encoded


# STEP 5: Fill in missing values with 0
def fill_missing_values(df, label_columns):
    feature_cols = [col for col in df.columns if col not in label_columns]
    missing = df[feature_cols].isnull().sum().sum()
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"Filled {missing} missing values with 0")
    return df


def load_and_preprocess(config):
    datasets_raw = {}
    binary_col = None
    multi_col = None

    # Step 1-3: Load each node, remove columns, prepare labels, force numeric
    for node_name, file_path in config["files"].items():
        print(f"\nPreprocesses {node_name} ({file_path})")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

        df = remove_unnecessary_columns(df, config)
        df, binary_col, multi_col = prepare_labels(df, config)
        df = force_numeric_columns(df, config)

        datasets_raw[node_name] = df

    label_columns = [binary_col, multi_col]

    # Step 4: Concatenate all nodes, one-hot encode once globally
    all_data = pd.concat(datasets_raw.values(), ignore_index=True)
    print(f"\nGlobal one-hot encoding on {len(all_data):,} rows")
    all_data = one_hot_encode(all_data, label_columns, config)

    # Step 5: Fill missing values
    all_data = fill_missing_values(all_data, label_columns)

    # Split back to individual nodes
    datasets = {}
    start = 0
    for node_name, df_raw in datasets_raw.items():
        length = len(df_raw)
        datasets[node_name] = all_data.iloc[start:start + length].copy()
        start += length

    for node_name, df in datasets.items():
        print(f"\nResult {node_name}: {df.shape[0]} rows, {df.shape[1]} columns")

    return datasets, binary_col, multi_col


# Normalization: Min-Max normalization to [0, 1]
# Calculates min/max only from training data, applies to both training and testing
# Binary columns (0/1) are left unchanged
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
    datasets, binary_col, multi_col = load_and_preprocess(WESTERMO_CONFIG)
    #datasets, binary_col, multi_col = load_and_preprocess(EDGE_IIOT_CONFIG)
    label_columns = [binary_col, multi_col]

    for node_name, df in datasets.items():
        feature_count = df.shape[1] - len(label_columns)

        print(f"\nSummary: {node_name}")
        print(f"Number of rows: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        print(f"Number of features: {feature_count}")
        print(f"Number of labels: {len(label_columns)}")

        print(f"\nClass distribution ({multi_col}):")
        for cls, count in df[multi_col].value_counts().items():
            pct = count / len(df) * 100
            print(f"{cls:<30} {count:>6} ({pct:.1f}%)")

        n_normal = (df[binary_col] == 0).sum()
        n_attack = (df[binary_col] == 1).sum()
        print(f"\nBinary: Normal={n_normal:,}, Attack={n_attack:,} "
              f"(anomaly ratio: {n_attack/len(df)*100:.1f}%)")

    all_data = pd.concat(datasets.values(), ignore_index=True)
    all_data = all_data.fillna(0)

    print(f"\nMerged (all nodes)")
    print(f"Total rows: {all_data.shape[0]:,}")
    print(f"Total columns: {all_data.shape[1]}")
    print(f"\nClass distribution ({multi_col}):")
    for cls, count in all_data[multi_col].value_counts().items():
        pct = count / len(all_data) * 100
        print(f"{cls:<30} {count:>6} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
