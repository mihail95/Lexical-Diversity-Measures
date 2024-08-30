import streamlit as st
from measures.TTR import ttr
from measures.MATTR import mattr
from measures.HDD import hdd
from measures.MTLD import mtld
from measures.compare_multiple import compare

def intro():
    st.write("# Lexical Diversity Measures")
    st.write('This application allows you to visualize various Lexical Diversity measures by adjusting the variables used in their calculations and plotting the results. It also offers the flexibility to input custom texts and generate plots for their Lexical Diversity measures.')

    st.write('All measures are implemented using the [lexical_diversity Python library by Kristopher Kyle](https://github.com/kristopherkyle/lexical_diversity).')

    st.write("## What is Lexical Diversity?")
    st.write("Jarvis (2013) describes *lexical diversity* as a concept that quantifies the proportion of types (unique words) in a language sample. The term is often used interchangeably with *lexical variability*, *lexical variation*, and *lexical variety*. In the fields of first and second language acquisition, bilingualism, multilingualism, and language testing and assessment, multiple language features, such as the range, variety, and diversity of words in a subject's language use, are thought to indicate complexity of vocabulary knowledge and overall language proficiency. Numerous measures of lexical diversity have been suggested throughout the years, most of which are based on statistical relationships between word types and tokens, i.e. they reflect the frequency of word repetition.")

    st.write("------------------")
    st.write("#### References")
    st.write("Jarvis, S. (2013), Capturing the Diversity in Lexical Diversity. Language Learning, 63: 87-106. https://doi.org/10.1111/j.1467-9922.2012.00739.x")


    st.sidebar.success("Select a measure above.")

page_names_to_funcs = {
    "—": intro,
    "Token Type Ratio (TTR)": ttr,
    "Moving-Average Type-Token Ratio (MATTR)": mattr,
    'Hypergeometric distribution D (HDD)': hdd,
    'Measure of lexical textual diversity (MTLD)': mtld,
    "Compare Multiple Custom Texts": compare,
}

measure_name = st.sidebar.selectbox("Choose a measure", page_names_to_funcs.keys())

# Default to the home page, if something goes wrong
if measure_name == None: measure_name = "—"

page_names_to_funcs[measure_name]()