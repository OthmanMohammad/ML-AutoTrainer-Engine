import streamlit as st
from src import data_processing, streamlit_utils

def main():
    st.title('ML AutoTrainer Engine')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = data_processing.read_csv(uploaded_file)
        st.write(data.head())
        
        target_column = streamlit_utils.select_target_column(data)  # Separate function for selecting target
        st.write(f"Selected Target Column: {target_column}")

if __name__ == "__main__":
    main()
