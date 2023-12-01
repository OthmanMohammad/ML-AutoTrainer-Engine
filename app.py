import streamlit as st
from src import data_processing
from src import streamlit_utils

def main():
    st.title("DataQueue AutoML App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        data = data_processing.read_csv(uploaded_file)
        st.write(data.head())

        target_column = streamlit_utils.select_target_column(data)
        st.write(f"Selected Target Column: {target_column}")

        # Filtering section
        with st.expander("Filter Data"):
            filtered_data = streamlit_utils.apply_filters(data)
            if filtered_data is not None:
                st.dataframe(filtered_data)

if __name__ == "__main__":
    main()
