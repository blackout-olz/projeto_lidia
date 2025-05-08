import pandas as pd

def load_dataframes(train_path='data/train.parquet', test_path='data/test.parquet'):
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    #print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test
