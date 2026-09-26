import pandas as pd 
import streamlit as st

# Human Written OMGGGGGGGGGGGGGG
input_file = st.file_uploader("Upload predictions", type="csv")

if input_file is not None: 
    file_csv = pd.read_csv(input_file)

