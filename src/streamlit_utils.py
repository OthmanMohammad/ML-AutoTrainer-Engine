import streamlit as st
from .model_training import train_model

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
    if st.button("Apply Filters"):
        # Intermingle filter_query and condition_operators to create the final query
        final_query_parts = []
        for i, query_part in enumerate(filter_query):
            final_query_parts.append(query_part)
            if i < len(condition_operators):
                final_query_parts.append(condition_operators[i])
        
        final_query = " ".join(final_query_parts)
        filtered_data = data.query(final_query)

    return filtered_data

def train_model_ui(data, target_column):
    model_type = st.selectbox("Select Model Type", ["Classification", "Regression"])
    if model_type == "Classification":
        model_name = st.selectbox("Select Classification Algorithm", ["Logistic Regression", "Random Forest Classifier"])
    elif model_type == "Regression":
        model_name = st.selectbox("Select Regression Algorithm", ["Linear Regression", "Decision Tree Regressor"])

    if st.button("Train Model"):
        return train_model(data, target_column, model_name)
    return None
