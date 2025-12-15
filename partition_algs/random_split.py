import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def random_split(file_name, SEEDS, test_size, keep_size = False):
    """
        keep_size: (default: False) -> set to True to keep the big-sized data, >1M samples
    """
    # Create directory if not exist
    output_dir = f"../data/splitted/{file_name}/Random_Split"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    data_path = f"../data/raw/{file_name}/{file_name}.parquet"
    df = pd.read_parquet(data_path)

    if df.shape[0] > 1000000 and not keep_size:
        df = df.sample(n = 800000, random_state = 42).reset_index(drop=True)
        print("Remove some samples due to extensive size")
        print(f"New Data: {df.shape[0]} samples, {df.shape[1]} features")
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    for idx, seed in enumerate(SEEDS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = test_size,
            random_state = seed,
            shuffle = True
        )

        df_train = pd.concat([X_train, y_train], axis = 1)
        df_test = pd.concat([X_test, y_test], axis = 1)

        # Save files using the idx
        path = os.path.join(output_dir, f"train_{idx}.parquet")
        df_train.to_parquet(path, index = False)
        path = os.path.join(output_dir, f"test_{idx}.parquet")
        df_test.to_parquet(path, index = False)