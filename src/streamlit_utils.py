import streamlit as st

def select_target_column(data):
    """Let the user select a target column and return it."""
    target_column = st.selectbox(
        "Select the target column for your ML task:",
        options=data.columns
    )
    return target_column
