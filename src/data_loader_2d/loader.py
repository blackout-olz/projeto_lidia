import pandas as pd

def load_dataframes(train_path='data/2d/train.parquet', test_path='data/2d/test.parquet'):
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    return df_train, df_test
