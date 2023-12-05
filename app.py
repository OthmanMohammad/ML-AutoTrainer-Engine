import streamlit as st
from src import data_processing, streamlit_utils

def main():
    st.title("ML AutoTrainer Enginer")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        data = data_processing.read_csv(uploaded_file)
        st.write(data.head())

        # Feature to select columns to drop
        columns_to_drop = st.multiselect(
            'Select columns to drop', data.columns)
        if columns_to_drop:
            data = data_processing.drop_columns(data, columns_to_drop)
            st.write('Updated Data after dropping columns')
            st.write(data.head())

        target_column = streamlit_utils.select_target_column(data)
        st.write(f"Selected Target Column: {target_column}")

        # Filtering section
        with st.expander("Filter Data"):
            filtered_data = streamlit_utils.apply_filters(data)
            if filtered_data is not None:
                st.dataframe(filtered_data)

        # Model training section
        with st.expander("Model Training"):
            results = streamlit_utils.train_model_ui(filtered_data or data, target_column)
            if results:
                st.write(results)

if __name__ == "__main__":
    main()
