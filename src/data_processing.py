import pandas as pd

def read_csv(file):
    return pd.read_csv(file)

def drop_columns(data, columns):
    return data.drop(columns=columns)

def handle_missing_values(data, strategy="drop"):
    if strategy == "drop":
        return data.dropna()
    elif strategy == "mean":
        return data.fillna(data.mean())
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
