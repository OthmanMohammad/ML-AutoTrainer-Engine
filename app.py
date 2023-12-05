import streamlit as st
from src import data_processing, streamlit_utils, projects, predictions
from src.data_recorder import DataProcessingRecorder
import pandas as pd
from src.data_processing_pipeline import DataProcessingPipeline
import json
import os

# Initialize the recorder in Streamlit's state if it's not already present
if 'data_processing_recorder' not in st.session_state:
    st.session_state['data_processing_recorder'] = DataProcessingRecorder()

# Initialize the uploaded file name in Streamlit's state if it's not already present
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None

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
                    # Make sure loaded_data is a DataFrame before assigning
                    if isinstance(loaded_data, tuple):
                        loaded_data = loaded_data[0]  # Assuming the first element of the tuple is the DataFrame
                    st.session_state.processed_data = loaded_data
                    st.success('Project data loaded successfully.')
                    # Load recorded steps if exist
                    steps = projects.load_processing_steps(selected_project)
                    if steps:
                        st.session_state.data_processing_recorder.load_steps(steps)
                else:
                    st.error('No processed data found for this project, please upload new data.')

    if 'selected_project' in st.session_state and st.session_state.selected_project:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            # Check if the uploaded file is different from the previously processed file
            if st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.uploaded_file_name = uploaded_file.name
                data = data_processing.read_csv(uploaded_file)
                st.session_state.processed_data = data.copy()
                projects.save_project_data(st.session_state.selected_project, data)

        # Check if processed_data is in session state and is a DataFrame and not empty
        if ('processed_data' in st.session_state and 
                isinstance(st.session_state.processed_data, pd.DataFrame) and 
                not st.session_state.processed_data.empty):
            st.write(st.session_state.processed_data.head())

            target_column = streamlit_utils.select_target_column(st.session_state.processed_data)
            st.write(f"Selected Target Column: {target_column}")

            with st.expander("Drop Columns"):
                columns_to_drop = st.multiselect("Select columns to drop", st.session_state.processed_data.columns)
                if columns_to_drop and st.button("Drop Selected Columns"):
                    st.session_state.processed_data = data_processing.drop_columns(st.session_state.processed_data, columns_to_drop)
                    # Record the step
                    st.session_state.data_processing_recorder.record_step('drop_columns', columns=columns_to_drop)
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
                    # Record the step
                    st.session_state.data_processing_recorder.record_step('handle_missing_values', strategy=strategy_code)
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                    st.write(st.session_state.processed_data.head())

            with st.expander("Convert Categorical Columns to Numerical"):
                columns_to_convert = st.multiselect("Select columns to convert", st.session_state.processed_data.select_dtypes(include=['object']).columns)
                if columns_to_convert and st.button("Convert"):
                    st.session_state.processed_data = data_processing.convert_categorical_to_numerical(st.session_state.processed_data, columns_to_convert)
                    # Record the step
                    st.session_state.data_processing_recorder.record_step('convert_categorical_to_numerical', columns=columns_to_convert)
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                    st.write(st.session_state.processed_data.head())

            with st.expander("Feature Extraction"):
                extracted_data, extraction_params = streamlit_utils.feature_extraction_ui(
                    st.session_state.processed_data, target_column=target_column)
                if extracted_data is not None:
                    st.session_state.processed_data = extracted_data
                    # Record the step with the extraction method and parameters
                    st.session_state.data_processing_recorder.record_step('feature_extraction', **extraction_params)
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                    st.dataframe(st.session_state.processed_data)

            with st.expander("Filter Data"):
                filtered_data = streamlit_utils.apply_filters(st.session_state.processed_data)
                if filtered_data is not None:
                    st.session_state.processed_data = filtered_data
                    # Record the step
                    st.session_state.data_processing_recorder.record_step('filter_data')
                    projects.save_project_data(st.session_state.selected_project, st.session_state.processed_data)
                st.dataframe(st.session_state.processed_data)

            with st.expander("Model Training"):
                if 'selected_project' in st.session_state and st.session_state.selected_project:
                    results = streamlit_utils.train_model_ui(st.session_state.processed_data, target_column, st.session_state.selected_project)
                    if results:
                        st.write(results)
                        # Record the step
                        st.session_state.data_processing_recorder.record_step('model_training', target_column=target_column)


            with st.expander("Apply Saved Pipeline to New Data"):
                new_data_file = st.file_uploader("Upload New Data CSV File", type="csv")
                target_column = st.text_input("Enter the name of the target column (it will be excluded from processing):")

                if new_data_file and target_column and 'selected_project' in st.session_state and st.session_state.selected_project:
                    # Define the path to the steps file
                    steps_file_path = os.path.join('projects', st.session_state.selected_project, 'processing_steps.json')

                    # Check if the steps file exists
                    if os.path.exists(steps_file_path):
                        # Load the steps from the JSON file
                        with open(steps_file_path, 'r') as file:
                            steps = json.load(file)

                        # Create a new DataProcessingPipeline instance
                        pipeline = DataProcessingPipeline()

                        # Load the new data
                        new_data = pd.read_csv(new_data_file)

                        # Remove the target column before applying the pipeline
                        if target_column in new_data.columns:
                            new_data = new_data.drop(columns=[target_column])

                        # Apply each step in the pipeline
                        try:
                            for step in steps:
                                step_name = step["step"]
                                parameters = step.get("parameters", {})
                                if hasattr(pipeline, step_name):
                                    # Get the function associated with the step name
                                    step_function = getattr(pipeline, step_name)
                                    # Call the function with the arguments from the steps file
                                    new_data = step_function(new_data, **parameters)
                                else:
                                    st.error(f"Step '{step_name}' is not a method of DataProcessingPipeline")

                            st.success('Pipeline applied successfully.')
                            st.write(new_data.head())

                            # Save the processed new data to a file
                            processed_new_data_path = os.path.join('projects', st.session_state.selected_project, 'processed_new_data.csv')
                            new_data.to_csv(processed_new_data_path, index=False)
                            st.session_state['processed_new_data_path'] = processed_new_data_path
                            st.success(f"Processed new data saved to {processed_new_data_path}")

                            # Add a button to download the processed new data
                            with open(processed_new_data_path, "rb") as f:
                                # Read the saved processed data file
                                processed_new_data = f.read()
                                st.download_button(
                                    label="Download Processed Data as CSV",
                                    data=processed_new_data,
                                    file_name="processed_new_data.csv",
                                    mime="text/csv",
                                    key='download-csv'
                                )

                        except Exception as e:
                            st.error(f"An error occurred: {e}")

                    else:
                        st.error(f"Steps file does not exist: {steps_file_path}")



            with st.expander("Model Prediction"):
                # Path to the trained model
                model_path = os.path.join('projects', st.session_state.selected_project, 'model.joblib')
                # Path where the predictions should be saved
                predictions_path = os.path.join('projects', st.session_state.selected_project, 'predictions.csv')

                # Load the model
                try:
                    model = predictions.load_model(model_path)
                    st.success("Model loaded successfully.")
                except Exception as e:
                    st.error(f"An error occurred while loading the model: {e}")
                    model = None

                # Load the processed data for predictions
                if 'processed_new_data_path' in st.session_state:
                    try:
                        data_for_prediction = pd.read_csv(st.session_state['processed_new_data_path'])
                        st.write("Data for prediction loaded successfully.")
                        st.write(data_for_prediction.head())
                    except Exception as e:
                        st.error(f"An error occurred while loading the data for prediction: {e}")
                        data_for_prediction = None
                else:
                    st.warning("No processed data available for predictions. Please apply the pipeline to new data first.")
                    data_for_prediction = None

                # Make predictions and save results
                if model and data_for_prediction is not None:
                    if st.button("Make Predictions"):
                        try:
                            preds = predictions.predict(model, data_for_prediction)
                            st.success("Predictions made successfully.")
                            
                            # Here I am not writing preds directly because I want to show them with the features
                            combined_results = data_for_prediction.copy()
                            combined_results['Predictions'] = preds
                            st.write(combined_results.head())

                            # Save the predictions along with the features
                            if predictions.save_predictions(data_for_prediction, preds, predictions_path):
                                st.success(f"Predictions saved to {predictions_path}")
                                
                                # Create a link to download the predictions CSV
                                with open(predictions_path, "rb") as file:
                                    btn = st.download_button(
                                        label="Download Predictions CSV",
                                        data=file,
                                        file_name="predictions.csv",
                                        mime="text/csv",
                                    )
                        except Exception as e:
                            st.error(f"An error occurred while making predictions: {e}")

                        
            # Save the processing steps to the project after each interaction
            projects.save_processing_steps(st.session_state.selected_project, st.session_state.data_processing_recorder.save_steps())

if __name__ == "__main__":
    main()
