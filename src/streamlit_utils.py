import streamlit as st

def select_target_column(data):
    target_column = st.selectbox("Select Target Column", data.columns)
    return target_column

def apply_filters(data):
    filter_conditions = st.number_input(
        "Number of filter conditions:", min_value=1, value=1
    )

    filter_query = []
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

    filtered_data = None
    if st.button("Apply Filters"):
        final_query = " and ".join(filter_query)
        filtered_data = data.query(final_query)

    return filtered_data
