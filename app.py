import streamlit as st
import pandas as pd
from src import data_processing

st.title('ML AutoTrainer Engine')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = data_processing.read_csv(uploaded_file)
    st.write(data)
