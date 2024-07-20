import streamlit as st
from measures.TTR import ttr
from measures.MATTR import mattr
from measures.compare_multiple import compare

def intro():
    st.write("# Lexical Diversity Measures")
    st.write('Some general information about the application')

    st.sidebar.success("Select a measure above.")

page_names_to_funcs = {
    "—": intro,
    "Token Type Ratio (TTR)": ttr,
    "Moving-Average Type-Token Ratio (MATTR)": mattr,
    "Compare Custom Texts": compare,
}

measure_name = st.sidebar.selectbox("Choose a measure", page_names_to_funcs.keys())

# Default to the home page, if something goes wrong
if measure_name == None: measure_name = "—"

page_names_to_funcs[measure_name]()