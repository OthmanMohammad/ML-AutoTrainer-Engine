import pandas as pd

def read_csv(file):
    return pd.read_csv(file)

def drop_columns(data, columns_to_drop):
    return data.drop(columns=columns_to_drop)
