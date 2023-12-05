import streamlit as st
from src import data_processing
from src import streamlit_utils

def main():
    st.title("ML AutoTrainer Enginer")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        data = data_processing.read_csv(uploaded_file)
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = data.copy()

        st.write(st.session_state.processed_data.head())

        target_column = streamlit_utils.select_target_column(st.session_state.processed_data)
        st.write(f"Selected Target Column: {target_column}")

        # Drop columns section
        with st.expander("Drop Columns"):
            columns_to_drop = st.multiselect("Select columns to drop", st.session_state.processed_data.columns)
            if st.button("Drop Selected Columns"):
                st.session_state.processed_data = data_processing.drop_columns(st.session_state.processed_data, columns_to_drop)
                st.write(st.session_state.processed_data.head())

        # Handle missing values section
        with st.expander("Handle Missing Values"):
            strategy = st.radio("Choose a strategy", ["Drop Rows", "Fill with Mean"])
            if st.button("Apply Strategy"):
                if strategy == "Drop Rows":
                    st.session_state.processed_data = data_processing.handle_missing_values(st.session_state.processed_data, strategy="drop")
                elif strategy == "Fill with Mean":
                    st.session_state.processed_data = data_processing.handle_missing_values(st.session_state.processed_data, strategy="mean")
                st.write(st.session_state.processed_data.head())

        # Convert categorical columns to numerical
        with st.expander("Convert Categorical Columns to Numerical"):
            columns_to_convert = st.multiselect("Select columns to convert", st.session_state.processed_data.columns)
            if columns_to_convert and st.button("Convert"):
                st.session_state.processed_data = data_processing.convert_categorical_to_numerical(st.session_state.processed_data, columns_to_convert)
                st.write(st.session_state.processed_data.head())

        # Filtering section
        with st.expander("Filter Data"):
            filtered_data = streamlit_utils.apply_filters(st.session_state.processed_data)
            if filtered_data is not None:
                st.dataframe(filtered_data)

        # Model training section
        with st.expander("Model Training"):
            results = streamlit_utils.train_model_ui(st.session_state.processed_data, target_column)
            if results:
                st.write(results)

if __name__ == "__main__":
    main()
