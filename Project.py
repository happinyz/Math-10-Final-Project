import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pandas.api.types import is_numeric_dtype

st.title("Alvin's Amazing Project")
st.markdown("Created by: Alvin Zou")

@st.cache
def get_data():
    url = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/2021_LoL_esports_match_data_from_OraclesElixir_20211204.csv"
    return pd.read_csv(url)

df = get_data()

st.write(df.shape)