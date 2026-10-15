import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
class preprocessing:

    @staticmethod
    def resolve_dataset(dataset_arg, config):
        if config and dataset_arg in config.get("files", {}):
            return config["files"][dataset_arg]
        return dataset_arg
    
    @staticmethod
    def drop_label_columns(df, config):
        if not config:
            return df
        return df.drop(columns=config.get("labels_to_drop", []), errors="ignore")
    
    @staticmethod
    def merge_normal_classes(df, config):
        if not config:
            return df

        label_col = config.get("multiclass_label")
        merge_vals = config.get("merge_to_normal", [])

        if label_col in df.columns:
            df[label_col] = df[label_col].replace(merge_vals, "Normal")

        return df
    
    @staticmethod

    def find_file(filename):
        if filename is None:
            return None

        path = Path(filename)
        if path.exists():
            return str(path)

        cwd = Path.cwd()
        local_path = cwd / filename

        if local_path.exists():
            return str(local_path)

        return None

    @staticmethod
    def load_config(config_path):
        if config_path is None:
            return None

        import importlib.util

        spec = importlib.util.spec_from_file_location("config_module", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.config

    @staticmethod
    def drop_columns(dataframe, columns_removed): # fusk features
        cols_to_drop = [
            c for c in dataframe.columns
            if any(k.lower() in c.lower() for k in columns_removed)
        ]
        return dataframe.drop(columns=cols_to_drop, errors="ignore")

    @staticmethod
    def fillna(dataframe, value=0):
        return dataframe.fillna(value)
    
    @staticmethod
    def non_binary_normalization(dataframe, config=None):
        numeric_cols = dataframe.select_dtypes(include="number").columns.tolist()
 
        if config:
            for k in ("binary_label", "multiclass_label"):
                col = config.get(k)
                if col in numeric_cols:
                    numeric_cols.remove(col)

        # keep binary numeric columns unscaled
        binary_cols = [
            c for c in numeric_cols
            if set(dataframe[c].dropna().unique()) <= {0, 1}
        ]
        real_cols = [c for c in numeric_cols if c not in binary_cols]

        return preprocessing.normalize(real_cols, dataframe)
        
    @staticmethod
    def normalize(cols,train_df,test_df): # skalan
        scaler = MinMaxScaler()
        scaler.fit(train_df[cols])
        train_df[cols] = scaler.transform(train_df[cols])
        test_df[cols] = scaler.transform(test_df[cols])
        return train_df, test_df

    @staticmethod
    def encode_config(df, config): # labels
        if not config:
            return df

        cat_cols = config.get("categorical_columns", {})

        for col, categories in cat_cols.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=categories)
                df = pd.get_dummies(df, columns=[col], prefix=col)

        return df
    
    @staticmethod
    def merge_datasets(dataframes):
        if not dataframes:
            raise ValueError("No datasets provided")

        return pd.concat(dataframes, ignore_index=True)
    
    @staticmethod
    def zon1(df, config, pr):
        if not config:
            return df
        
        target = config.get("label_column")  
        to_remove = config.get("columns_to_remove", [])

        df = pr.drop_columns(df, to_remove)
    
    # Vi sparar bara kolumner som är numeriska eller vår target.
        df = df[[col for col in df.columns if df[col].dtype != 'object' or col == target]]
        
        df = pr.merge_normal_classes(df, config)
    
        df = pr.encode_config(df, config)


    # Om det kolumner med text kvar som inte är target, kasta den.
    # stoppar ValueError i Execute_IF.
        final_text_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target]
        if final_text_cols:
            df = df.drop(columns=final_text_cols)
        
        return df
    
    @staticmethod
    def zon2 (train_df, test_df, config, pr): 
        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)
        
        
        all_labels = config.get("all_labels_list", ["NST_B_Label", "Attack_Type", "Target"]) 

        cols = train_df.select_dtypes(include=['float64', 'int64']).columns
        cols_to_scale = [c for c in cols if c not in all_labels]

        
        train_df, test_df = pr.normalize(cols_to_scale, train_df, test_df)
        
        return train_df, test_df
   
def Execute_IF_kmeans(X_train, X_test):
    algorithm = IsolationForest(max_samples=256, n_estimators=100, random_state=42)
    algorithm.fit(X_train)
    raw_result = algorithm.predict(X_test)
    anomaly_scores = algorithm.decision_function(X_test)
    convert_to_RF = [1 if x == -1 else 0 for x in raw_result]

   
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(anomaly_scores.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_
    kmeans_threshold = np.mean(cluster_centers)
    predictions = [1 if score < kmeans_threshold else 0 for score in anomaly_scores]

    return convert_to_RF, anomaly_scores, predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    pr = preprocessing()
    args = sys.argv[1:]

    if args[-1].endswith(".py"):
        config = args[-1]
        datasets = args[:-1]
    else:
        config = None
        datasets = args

    if config:
        config_path = pr.find_file(config)
        config = pr.load_config(config_path) if config_path else None

    dataframes = []
    for ds in datasets:
        resolved = pr.resolve_dataset(ds, config)
        path = pr.find_file(resolved)

        if path is None:
            print("Dataset not found:", ds)
            sys.exit(1)

        dataframes.append(pd.read_csv(path))

    df = pr.merge_datasets(dataframes)
    if config:
        df = pr.zon1(df, config, pr)
        target_col = config.get("label_column", "NST_B_Label")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for current_fold, (train_index, test_index) in enumerate(kf.split(df), 1):
            print(f"\n--- Kör Fold {current_fold} ---")
            
            train_df = df.iloc[train_index].copy()
            test_df = df.iloc[test_index].copy()

            train_df, test_df = pr.zon2(train_df, test_df, config, pr)

            X_train = train_df.drop(columns=[target_col])
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            unique_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
            unnecessary_cols = list(set(unique_cols))

            ready_X_train = X_train.drop(columns=[c for c in unnecessary_cols if c in X_train.columns])
            ready_X_test = X_test.drop(columns=[c for c in unnecessary_cols if c in X_test.columns])

            predictions, scores, kmeans_predictions = Execute_IF_kmeans(ready_X_train, ready_X_test)
            
            print(f"Totalt antal gissade attacker (K-means): {sum(kmeans_predictions)}")
            print(f"Totalt antal faktiska attacker: {sum(y_test)}")
            
            accuracy = accuracy_score(y_test, kmeans_predictions)
            print(f"Modellens noggrannhet för denna fold: {accuracy:.2%}")
            print(classification_report(y_test, kmeans_predictions))