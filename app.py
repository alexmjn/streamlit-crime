import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


st.title('Exploring Crime Clearance Rates')

@st.cache
def load_data():
    return pd.read_pickle("data_file.bz2")

@st.cache
def load_model():
    model = XGBClassifier()
    model.load_model("xgb_model.txt")

df = load_data()
# need to rename columns, cast hour, year, and month to integers

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

if st.sidebar.checkbox('Show summary statistics'):
    st.write(df.head())
