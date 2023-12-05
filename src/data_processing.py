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
    elif strategy == "median":
        return data.fillna(data.median())
    elif strategy == "mode":
        # Assuming mode to be the first one
        return data.fillna(data.mode().iloc[0])
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

def convert_categorical_to_numerical(data, columns_to_convert):
    for col in columns_to_convert:
        unique_values = data[col].unique()
        if len(unique_values) == 2:
            # Map two unique values to 0 and 1 for binary categorical columns
            data[col] = data[col].map({unique_values[0]: 0, unique_values[1]: 1})
        else:
            # For columns with more than two unique values, use one-hot encoding
            one_hot = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, one_hot], axis=1)
            data.drop(col, axis=1, inplace=True)
    return data
