import streamlit as st
import numpy as np
import pandas as pd
import statistics
from measures.helpers.text_generation import generate_repeating_text_naive, generate_random_text_naive, generate_random_text_zipf, genearate_alphabet_permuatations
from lexical_diversity import lex_div as ld
import matplotlib.pyplot as plt

def hdd():
    st.header('Hypergeometric distribution D (HDD)', divider='gray')
    st.markdown("The hypergeometric distribution calculates the probability of drawing a certain number of tokens of a specific type from a sample without replacement. In the HD-D index, this distribution is used to determine the probability of encountering each lexical type in a random sample of 42 words from the text. These probabilities are summed to create an LD index for the text (McCarthy & Jarvis, 2010).")
    st.write("Fergadiotis et al. (2015) shows that HDD covaries with TTR in confirmatory factor analysis, suggesting that it also is affected by text length.")

    with st.expander("HDD Algorithm", icon=":material/expand_circle_down:"):
        st.markdown("1. For each type $t$ in $V$ calculate the probability $HD_t$ of finding it in a random sample using the Hypergeometric Distribution.")
        st.markdown(r"2. Calculate the HD-D index: $\displaystyle\sum\limits_t^{|V|}HD_t$")

    st.subheader('Dynamic Graph', divider='gray')
    # False = Vary Vocab; True = Vary Text Length
    vocab_or_len = st.toggle("Vary Vocabulary Size / Vary Text Length", value= False, help="Left = Vary Vocabulary; Right = Vary Text Length")
    compare_custom_text = st.toggle("Custom Text Comparison", value= False)

    with st.form("HDD-Form"):
        text_gen_alg = st.selectbox(
            "Which text generation algorithm should be used?",
            ("Sequential", "Random", "Zipf Distribution"),
            index=None,
            placeholder="Select text generation algorithm...",
            help="Sequential: Repeats all vocabulary items in the same order until the maximum text length is reached\n\nRandom: Picks random vocabulary types (each has the same probability\n\nZipf Distribution: Picks random vocabulary types from a zipf distribution (value of the n-th entry is inversely proportional to n)"
        )

        if vocab_or_len:
            # Vary Text Length => Constant vocabulary size and different starting and max text lengths
            min_vocab_size = st.slider("Vocabulary Size", min_value=1, max_value=500, value=10, step=1)
            max_vocab_size = min_vocab_size
            min_text_length = st.slider("Starting Text Length", min_value=1, max_value=500, value=10, step=1)
            max_text_length = st.slider("Maximum Text Length", min_value=1, max_value=500, value=100, step=1)
        else:
            # Vary Vocab size => Different starting and maximum vocab sizes and constant text length
            min_vocab_size = st.slider("Starting Vocabulary Size", min_value=1, max_value=500, value=10, step=1)
            max_vocab_size = st.slider("Maximum Vocabulary Size", min_value=1, max_value=500, value=50, step=1)
            min_text_length = st.slider("Text Length", min_value=1, max_value=500, value=50, step=1)
            max_text_length = min_text_length
        
        smoothing_runs = st.slider("Smoothing Runs Count", min_value=1, max_value=50, value=10, step=1)

        custom_text = ""
        if compare_custom_text:
            custom_text = st.text_area(f"Enter Custom Text Here")

        if st.form_submit_button("Run Test"):
            if ((max_vocab_size < min_vocab_size) or (max_text_length < min_text_length)):
                st.warning("Maximum variable bounds should be higher than the minimum!", icon=":material/warning:")
            elif(text_gen_alg == None):
                st.warning("Please select a text generation algorithm!", icon=":material/warning:")
            else:
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                progress_bar.empty()

                ########## HDD MAIN LOOP ##################################
                text_gen_map = {
                    "Sequential": generate_repeating_text_naive,
                    "Random": generate_random_text_naive,
                    "Zipf Distribution": generate_random_text_zipf,
                }
                gen_function = text_gen_map[text_gen_alg]

                total_cycles = (1 + max_text_length - min_text_length) * (1 + max_vocab_size - min_vocab_size) * smoothing_runs
                alphabet = sorted(list(genearate_alphabet_permuatations(ngrams = 2)))
                chart = st.line_chart()
                progress_ctr = 1

                all_rows = []
                indeces = []

                for curr_vocab_size in range(min_vocab_size, max_vocab_size + 1):
                    vocab = set(alphabet[:curr_vocab_size])
                    for curr_text_len in range(min_text_length, max_text_length + 1):
                        len_hdd_scores = []
                        for _ in range(smoothing_runs):
                            text = gen_function(vocab, curr_text_len).split()
                            len_hdd_scores.append(ld.hdd(text))
                            status_text.text(f"{round(progress_ctr/total_cycles, 2) * 100}% Complete")
                            progress_bar.progress(round(progress_ctr/total_cycles, 2))
                            progress_ctr += 1
                        hdd_score = statistics.mean(len_hdd_scores)
                        df_idx = curr_text_len if vocab_or_len else curr_vocab_size
                        chart.add_rows(pd.DataFrame([hdd_score], columns=["hdd"], index=[df_idx]))
                        all_rows.append(hdd_score)
                        indeces.append(df_idx)
                        
                # Custom Text Comparison
                if compare_custom_text:
                    st.subheader('Custom Text Comparison', divider='gray')
                    st.markdown("The circle represents a the HDD measure for the given custom text")
                    st.markdown(":black_circle: - HDD")

                    fig = plt.figure() 
                    all_data = pd.DataFrame(all_rows, index=indeces)
                    plt.plot(all_data)

                    custom_text_split = custom_text.split()
                    x_axis = len(custom_text_split) if vocab_or_len else len(set(custom_text_split))
                    plt.plot(x_axis, ld.hdd(custom_text_split), 'ko')
                    st.pyplot(fig)

    st.write("------------------")
    st.write("#### References")
    st.write("Fergadiotis, G., Wright, H. H., & Green, S. B. (2015). Psychometric Evaluation of Lexical Diversity Indices: Assessing Length Effects. Journal of speech, language, and hearing research : JSLHR, 58(3), 840–852. https://doi.org/10.1044/2015_JSLHR-L-14-0280")
    st.write("McCarthy, P.M., Jarvis, S. MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. Behavior Research Methods 42, 381–392 (2010). https://doi.org/10.3758/BRM.42.2.381") 