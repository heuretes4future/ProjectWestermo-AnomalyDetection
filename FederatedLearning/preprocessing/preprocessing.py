import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path
import os

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
    def drop_columns(dataframe, columns_removed):
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
    def normalize(real_cols, dataframe):
        scaler = MinMaxScaler()
        dataframe[real_cols] = scaler.fit_transform(dataframe[real_cols])
        return dataframe

    @staticmethod
    def encode_config(df, config):
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
    def clean_from_config(df, config, pr):
        if not config:
            return df
        
        df = pr.drop_columns(df, config.get("columns_to_remove", []))
        df = pr.drop_label_columns(df, config)
        df = pr.merge_normal_classes(df, config)
        df = pr.encode_config(df, config)
        df = pr.fillna(df, 0)

        #REFERENCED IN THE MODEL (RANDOM FOREST/ISOLATION FOREST)
        #df = pr.non_binary_normalization(df, config)

        return df
    


    @staticmethod
    def get_next_filename(basename):
        i = 1
        while True:
            filename = f"{basename}_{i}.csv"
            if not os.path.exists(filename):
                return filename
            i += 1



#STEPS 
#-----------------------------------------------------------------------------------------------------------------------------
#1. dataframe cleaning 
#2. hot encoding 
#3. missing values = 0 
#4. normalization  
#5. ettiquette strategy 
#6. Steg 6: Good-SSH transform -> normal    
#-----------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit(1)

    pr = preprocessing()
    args = sys.argv[1:]

    # last argument is config if .py
    if args[-1].endswith(".py"):
        config = args[-1]
        datasets = args[:-1]
    else:
        config = None
        datasets = args

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


# Code that preprocesses the code 
#--------------------------------------------------------------------------------
    if config:
        df = pr.clean_from_config(df, config, pr)

#--------------------------------------------------------------------------------

#Writing to output 
#-------------------------------------------------------------------------------- 
    output_path = preprocessing.get_next_filename("output")
    df.to_csv(output_path, index=False)


    print("Saved preprocessed file to:", output_path)
#--------------------------------------------------------------------------------