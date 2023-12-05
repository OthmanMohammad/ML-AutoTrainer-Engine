import streamlit as st
import pandas as pd
from .model_training import train_model, evaluate_model
from .feature_extraction import (
    apply_pca, apply_ica, apply_lda, apply_feature_agglomeration, 
    select_k_best, apply_variance_threshold
)

def select_target_column(data):
    target_column = st.selectbox("Select Target Column", data.columns)
    return target_column

def apply_filters(data):
    filter_conditions = st.number_input(
        "Number of filter conditions:", min_value=1, value=1
    )

    filter_query = []
    condition_operators = []  # store the conditions between each filter

    for i in range(filter_conditions):
        col = st.selectbox(f"Select column {i+1}", data.columns)
        op = st.selectbox(
            f"Select operator {i+1}",
            options=["==", "!=", ">", "<", ">=", "<="]
        )
        val = st.text_input(f"Enter value {i+1}")

        # Quote the column name if it contains a space
        if " " in col:
            col = f"`{col}`"
            
        # For string values, wrap them in quotes
        if val.isalpha():
            val = f"'{val}'"
        
        filter_query.append(f"{col} {op} {val}")

        # Add a condition selector for each filter, except the last one
        if i < filter_conditions - 1:
            condition = st.selectbox(f"Condition between filter {i+1} and {i+2}", ["and", "or"])
            condition_operators.append(condition)

    filtered_data = None
    if st.button("Apply Filters", key='apply_filters_button'):
        # Intermingle filter_query and condition_operators to create the final query
        final_query_parts = []
        for i, query_part in enumerate(filter_query):
            final_query_parts.append(query_part)
            if i < len(condition_operators):
                final_query_parts.append(condition_operators[i])
        
        final_query = " ".join(final_query_parts)
        filtered_data = data.query(final_query)

    return filtered_data

def feature_extraction_ui(data, target_column):
    extraction_methods = {
        "PCA": "Principal Component Analysis - Dimensionality reduction",
        "ICA": "Independent Component Analysis - Signal separation",
        "LDA": "Linear Discriminant Analysis - Class separability maximization",
        "Feature Agglomeration": "Hierarchical clustering of features",
        "SelectKBest": "Select features based on top K highest scores",
        "Variance Threshold": "Feature selector that removes all low-variance features",
        "None": "No extraction or selection"
    }

    extraction_method = st.selectbox("Select Feature Extraction Method", list(extraction_methods.keys()), format_func=lambda x: f"{x} - {extraction_methods[x]}")

    data_transformed = None

    if extraction_method == "PCA":
        n_components = st.slider("Number of Principal Components", 1, data.shape[1]-1)
    elif extraction_method == "ICA":
        n_components = st.slider("Number of Independent Components", 1, data.shape[1]-1)
    elif extraction_method == "LDA":
        classes = len(data[target_column].unique())
        if classes > 2:
            n_components = st.slider("Number of Components for LDA", 1, min(data.shape[1], classes - 1))
        else:
            st.warning("LDA requires more than 2 unique target classes.")
    elif extraction_method == "Feature Agglomeration":
        n_clusters = st.slider("Number of clusters for Feature Agglomeration", 1, data.shape[1]-1)
    elif extraction_method == "SelectKBest":
        k = st.slider("Number of top features to select", 1, data.shape[1]-1)
    elif extraction_method == "Variance Threshold":
        threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.05, 0.01)

    # Confirmation button
    if st.button("Confirm Feature Extraction", key='confirm_feature_extraction_button'):
        if extraction_method == "PCA":
            data_transformed = apply_pca(data.drop(target_column, axis=1), n_components)
        elif extraction_method == "ICA":
            data_transformed = apply_ica(data.drop(target_column, axis=1), n_components)
        elif extraction_method == "LDA":
            data_transformed = apply_lda(data.drop(target_column, axis=1), data[target_column], n_components)
        elif extraction_method == "Feature Agglomeration":
            data_transformed = apply_feature_agglomeration(data.drop(target_column, axis=1), n_clusters)
        elif extraction_method == "SelectKBest":
            data_transformed = select_k_best(data.drop(target_column, axis=1), data[target_column], k)
        elif extraction_method == "Variance Threshold":
            data_transformed = apply_variance_threshold(data.drop(target_column, axis=1), threshold)

        # Transform to DataFrame and set column names
        if data_transformed is not None:
            data_transformed = pd.DataFrame(data_transformed)
            data_transformed.columns = [f'Feature_{i}' for i in range(data_transformed.shape[1])]
            return data_transformed
        else:
            st.warning("No feature extraction method selected or no changes applied.")

    return None

def train_model_ui(data, target_column):
    # Model Type Selection
    model_type = st.selectbox("Select Model Type", ["Classification", "Regression", "Clustering"])

    # Model Algorithm Selection based on Model Type
    if model_type == "Classification":
        model_name = st.selectbox("Select Classification Algorithm", ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"])
    elif model_type == "Regression":
        model_name = st.selectbox("Select Regression Algorithm", ["Linear Regression", "Decision Tree Regressor", "Gradient Boosting Regressor"])
    elif model_type == "Clustering":
        model_name = st.selectbox("Select Clustering Algorithm", ["KMeans Clustering"])
    else:
        model_name = None

    if model_name:
        # Train Button with unique key
        if st.button("Train Model", key="train_model_button"):
            model, X_test, y_test = train_model(data, target_column, model_name)
            evaluation = evaluate_model(model, X_test, y_test, model_name)
            return evaluation
