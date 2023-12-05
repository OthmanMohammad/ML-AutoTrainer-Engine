import pandas as pd
import joblib
from .feature_extraction import (apply_pca, apply_ica, apply_lda,
                                 apply_feature_agglomeration, select_k_best,
                                 apply_variance_threshold)

class DataProcessingPipeline:

    def __init__(self):
        self.pipeline = []

    def read_csv(self, file):
        data = pd.read_csv(file)
        self.pipeline.append(('read_csv', {'file': file}))
        return data

    def drop_columns(self, data, columns):
        data = data.drop(columns=columns)
        self.pipeline.append(('drop_columns', {'columns': columns}))
        return data

    def handle_missing_values(self, data, strategy="drop"):
        if strategy == "drop":
            data = data.dropna()
        elif strategy == "mean":
            data = data.fillna(data.mean())
        elif strategy == "median":
            data = data.fillna(data.median())
        elif strategy == "mode":
            data = data.fillna(data.mode().iloc[0])
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.pipeline.append(('handle_missing_values', {'strategy': strategy}))
        return data

    def convert_categorical_to_numerical(self, data, columns=None):
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            unique_values = data[col].unique()
            if len(unique_values) == 2:
                data[col] = data[col].map({unique_values[0]: 0, unique_values[1]: 1})
            else:
                one_hot = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, one_hot], axis=1)
                data.drop(col, axis=1, inplace=True)
        self.pipeline.append(('convert_categorical_to_numerical', {'columns': columns}))
        return data

    def feature_extraction(self, data, method, n_components):
        if method == 'PCA':
            data_transformed = apply_pca(data, n_components=n_components)
        elif method == 'ICA':
            data_transformed = apply_ica(data, n_components=n_components)
        # Note: Other methods need to be implemented
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")

        self.pipeline.append(('feature_extraction', {'n_components': n_components, 'method': method}))
        return pd.DataFrame(data_transformed)

    def save_pipeline(self, file_name):
        joblib.dump(self.pipeline, file_name)

    @staticmethod
    def load_pipeline(file_name):
        pipeline_steps = joblib.load(file_name)
        pipeline = DataProcessingPipeline()
        pipeline.pipeline = pipeline_steps
        return pipeline

    def apply_pipeline(self, data):
        for step_name, parameters in self.pipeline:
            if hasattr(self, step_name):
                step_function = getattr(self, step_name)
                data = step_function(data, **parameters)
            else:
                raise ValueError(f"Step '{step_name}' is not a method of DataProcessingPipeline")
        return data
