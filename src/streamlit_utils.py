import streamlit as st
import pandas as pd
from .model_training import train_model, evaluate_model
from .feature_extraction import (
    apply_pca, apply_ica, apply_lda, apply_feature_agglomeration, 
    select_k_best, apply_variance_threshold
)
import joblib
import os
from .projects import save_model 

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

def feature_extraction_ui(data, target_column=None):
    extraction_methods = {
        "PCA": "Principal Component Analysis - Dimensionality reduction",
        "ICA": "Independent Component Analysis - Signal separation",
        "LDA": "Linear Discriminant Analysis - Class separability maximization",
        "Feature Agglomeration": "Hierarchical clustering of features",
        "SelectKBest": "Select features based on top K highest scores",
        "Variance Threshold": "Feature selector that removes all low-variance features",
        "None": "No extraction or selection"
    }

    extraction_method = st.selectbox(
        "Select Feature Extraction Method",
        list(extraction_methods.keys()),
        format_func=lambda x: f"{x} - {extraction_methods[x]}"
    )

    data_transformed = None
    params = {}

    if extraction_method != "None":
        # Exclude target variable for transformation
        if target_column:
            features = data.drop(target_column, axis=1)
            st.write(f"Dropped {target_column}. Features shape now:", features.shape)  # Debug print: After dropping target
        else:
            features = data
            st.write("No target column specified. Features shape:", features.shape)  # Debug print: No target dropped


        n_features = features.shape[1]
        min_features = 1

        if extraction_method in ["PCA", "ICA", "Feature Agglomeration", "SelectKBest"]:
            # Slider for components or features selection
            n_components = st.slider(
                f"Number of Components for {extraction_method}",
                min_value=min_features, max_value=n_features, value=min(n_features // 2, min_features)
            )
            params['n_components'] = n_components

        elif extraction_method == "LDA":
            if target_column is not None:
                classes = len(data[target_column].unique())
                n_features = min(features.shape[1], classes - 1)
                min_features = min(n_features, 2)
                # Slider for components or features selection for LDA
                n_components = st.slider(
                    f"Number of Components for {extraction_method}",
                    min_value=min_features, max_value=n_features, value=min_features
                )
                params['n_components'] = n_components
            else:
                st.error("Please specify the target column for LDA.")
                return None, None
        
        elif extraction_method == "Variance Threshold":
            # Variance Threshold slider
            threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.05, 0.01)
            params['threshold'] = threshold

        # Confirmation button
        if st.button("Confirm Feature Extraction", key='confirm_feature_extraction_button'):
            params['method'] = extraction_method  # Save the method name

            if extraction_method == "PCA":
                data_transformed = apply_pca(features, params['n_components'])
            elif extraction_method == "ICA":
                data_transformed = apply_ica(features, params['n_components'])
            elif extraction_method == "LDA" and target_column:
                data_transformed = apply_lda(features, data[target_column], params['n_components'])
            elif extraction_method == "Feature Agglomeration":
                data_transformed = apply_feature_agglomeration(features, params['n_components'])
            elif extraction_method == "SelectKBest" and target_column:
                data_transformed = select_k_best(features, data[target_column], params['n_components'])
            elif extraction_method == "Variance Threshold":
                data_transformed = apply_variance_threshold(features, params['threshold'])

            # Transform to DataFrame and add target column back if necessary
            if data_transformed is not None:
                data_transformed = pd.DataFrame(data_transformed)
                st.write("Transformed data shape before adding target:", data_transformed.shape)  # Debug print: Transformed shape before adding target
                if target_column:
                    data_transformed[target_column] = data[target_column].reset_index(drop=True)
                    st.write(f"Added {target_column} back. Final data shape:", data_transformed.shape)  # Debug print: Final shape after adding target
                data_transformed.columns = [f'Feature_{i}' for i in range(data_transformed.shape[1] - (1 if target_column else 0))] + ([target_column] if target_column else [])
                return data_transformed, params
            else:
                st.warning("No feature extraction method selected or no changes applied.")

    # In case of "None" or no confirmation, return None and the empty params dictionary
    return None, params


def train_model_ui(data, target_column, project_name):
    # Initialize session state for the trained model, model name, and evaluation
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'trained_model_name' not in st.session_state:
        st.session_state.trained_model_name = ""
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = ""

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

    # If the selected model name changes, reset the trained model and evaluation
    if st.session_state.trained_model_name != model_name:
        st.session_state.trained_model_name = model_name
        st.session_state.trained_model = None
        st.session_state.evaluation = ""

    if model_name:
        if st.button("Train Model", key="train_model_button"):
            st.session_state.trained_model, X_test, y_test = train_model(data, target_column, model_name)
            st.session_state.evaluation = evaluate_model(st.session_state.trained_model, X_test, y_test, model_name)
            # Save the trained model into the project directory
            save_model(project_name, st.session_state.trained_model)
        
        # Show evaluation result if available
        if st.session_state.evaluation:
            st.write(st.session_state.evaluation)

        # Adding a button for downloading the trained model
        if st.session_state.trained_model:
            model_file_path = download_model(st.session_state.trained_model, model_name)
            with open(model_file_path, "rb") as f:
                bytes_data = f.read()
            st.download_button(
                label=f"Download {model_name}",
                data=bytes_data,
                file_name=model_name.replace(" ", "_").lower() + ".pkl",
                mime="application/octet-stream"
            )

def download_model(model, model_name):
    model_file = model_name.replace(" ", "_").lower() + ".pkl"
    joblib.dump(model, model_file)
    return model_file
