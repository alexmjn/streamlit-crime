import streamlit as st
import numpy as np
import pandas as pd

st.title('Exploring Crime Clearance Rates')

@st.cache
def load_data():
    return pd.read_pickle("data_file.bz2")


if st.checkbox('Show dataframe heading'):
    df = load_data()

    df
