import streamlit as st
from src import data_processing
from src import streamlit_utils
from src import projects

def main():
    st.title("ML AutoTrainer Enginer")

    with st.sidebar:
        st.header('Projects')
        projects.add_project_form()
        selected_project = st.selectbox('Select a project', [''] + projects.get_project_names())
        if selected_project:
            st.session_state.selected_project = selected_project
            if st.button('Load Project Data'):
                loaded_data = projects.load_project_data(selected_project)
                if loaded_data is not None:
                    st.session_state.processed_data = loaded_data
                    st.success('Project data loaded successfully.')
                else:
                    st.error('No processed data found for this project, please upload new data.')

    if 'selected_project' in st.session_state and st.session_state.selected_project:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            data = data_processing.read_csv(uploaded_file)
            # Only update session state if 'processed_data' is not set or empty.
            if 'processed_data' not in st.session_state or st.session_state.processed_data.empty:
                st.session_state.processed_data = data.copy()
                projects.save_project_data(st.session_state.selected_project, data)

        if 'processed_data' in st.session_state and not st.session_state.processed_data.empty:
            st.write(st.session_state.processed_data.head())
            
            target_column = streamlit_utils.select_target_column(st.session_state.processed_data)
            st.write(f"Selected Target Column: {target_column}")

            with st.expander("Drop Columns"):
                columns_to_drop = st.multiselect("Select columns to drop", st.session_state.processed_data.columns)
                if columns_to_drop and st.button("Drop Selected Columns"):
                    st.session_state.processed_data = data_processing.drop_columns(st.session_state.processed_data, columns_to_drop)
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                    st.write(st.session_state.processed_data.head())

            with st.expander("Handle Missing Values"):
                strategy_map = {
                    "Drop Rows": "drop",
                    "Fill with Mean": "mean",
                    "Fill with Median": "median",
                    "Fill with Mode": "mode"
                }
                strategy = st.radio("Choose a strategy", list(strategy_map.keys()))
                if st.button("Apply Strategy"):
                    strategy_code = strategy_map[strategy]
                    st.session_state.processed_data = data_processing.handle_missing_values(st.session_state.processed_data, strategy_code)
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                    st.write(st.session_state.processed_data.head())

            with st.expander("Convert Categorical Columns to Numerical"):
                columns_to_convert = st.multiselect("Select columns to convert", st.session_state.processed_data.select_dtypes(include=['object']).columns)
                if columns_to_convert and st.button("Convert"):
                    st.session_state.processed_data = data_processing.convert_categorical_to_numerical(st.session_state.processed_data, columns_to_convert)
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                    st.write(st.session_state.processed_data.head())

            with st.expander("Feature Extraction"):
                extracted_data = streamlit_utils.feature_extraction_ui(st.session_state.processed_data, target_column)
                if extracted_data is not None:
                    st.session_state.processed_data = extracted_data
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                st.dataframe(st.session_state.processed_data)

            with st.expander("Filter Data"):
                filtered_data = streamlit_utils.apply_filters(st.session_state.processed_data)
                if filtered_data is not None:
                    st.session_state.processed_data = filtered_data
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                st.dataframe(st.session_state.processed_data)

            with st.expander("Model Training"):
                if 'selected_project' in st.session_state and st.session_state.selected_project:
                    results = streamlit_utils.train_model_ui(st.session_state.processed_data, target_column, st.session_state.selected_project)
                    if results:
                        st.write(results)
                        
            with st.expander("Load Model"):
                if st.button('Load Trained Model'):
                    loaded_model = projects.load_model(st.session_state.selected_project)
                    if loaded_model is not None:
                        st.session_state.trained_model = loaded_model
                        st.success('Model loaded successfully.')


if __name__ == "__main__":
    main()
