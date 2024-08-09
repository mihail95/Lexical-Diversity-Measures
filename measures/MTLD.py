import streamlit as st
import numpy as np
import pandas as pd
import statistics
from measures.helpers.text_generation import generate_repeating_text_naive, generate_random_text_naive, generate_random_text_zipf, genearate_alphabet_permuatations
from lexical_diversity import lex_div as ld
import matplotlib.pyplot as plt

def mtld():
    st.header('Measure of lexical textual diversity (MTLD)', divider='gray')
    st.markdown("Include some background on MTLD")

    st.subheader('Dynamic Graph', divider='gray')

    # False = Vary Vocab; True = Vary Text Length
    vocab_or_len = st.toggle("Vary Vocabulary Size / Vary Text Length", value= False, help="Left = Vary Vocabulary; Right = Vary Text Length")

    columns = st.columns([6,1,6,1,1,1,1,1,1,1,1,1])
    with columns[0]:
        show_bidir = st.toggle("Bidirectional MTLD", value= False)
    with columns[2]:
        show_wrap = st.toggle("Wrapping MTLD", value= False)

    compare_custom_text = st.toggle("Custom Text Comparison", value= False)

    with st.form("MTLD-Form"):
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
        
        text_gen_alg = st.selectbox(
            "Which text generation algorithm should be used?",
            ("Sequential", "Random", "Zipf Distribution"),
            index=None,
            placeholder="Select text generation algorithm...",
            help="Sequential: Repeats all vocabulary items in the same order until the maximum text length is reached\n\nRandom: Picks random vocabulary types (each has the same probability\n\nZipf Distribution: Picks random vocabulary types from a zipf distribution (value of the n-th entry is inversely proportional to n)"
        )
        
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

                ########## MTLD MAIN LOOP ##################################
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

                cols = ["mtld"]
                if show_bidir: cols.append("mtld_bid")
                if show_wrap: cols.append("mtld_wrap")

                all_rows = []
                indeces = []

                for curr_vocab_size in range(min_vocab_size, max_vocab_size + 1):
                    vocab = set(alphabet[:curr_vocab_size])
                    for curr_text_len in range(min_text_length, max_text_length + 1):
                        len_mtld_scores = []
                        len_mtld_bid_scores = []
                        len_mtld_wrap_scores = []
                        for _ in range(smoothing_runs):
                            text = gen_function(vocab, curr_text_len).split()
                            len_mtld_scores.append(ld.mtld(text))
                            if show_bidir: len_mtld_bid_scores.append(ld.mtld_ma_bid(text))
                            if show_wrap: len_mtld_wrap_scores.append(ld.mtld_ma_wrap(text))
                            status_text.text(f"{round(progress_ctr/total_cycles, 2) * 100}% Complete")
                            progress_bar.progress(round(progress_ctr/total_cycles, 2))
                            progress_ctr += 1
                        mtld_score = statistics.mean(len_mtld_scores)
                        data = [mtld_score]
                        df_idx = curr_text_len if vocab_or_len else curr_vocab_size
                        all_rows.append(data)
                        indeces.append(df_idx)
                        if show_bidir or show_wrap:
                            data = [data]
                            if show_bidir:
                                mtld_bid_score = statistics.mean(len_mtld_bid_scores)
                                data[0].append(mtld_bid_score)
                            if show_wrap:
                                mtld_wrap_score = statistics.mean(len_mtld_wrap_scores)
                                data[0].append(mtld_wrap_score)
                        
                        chart.add_rows(pd.DataFrame(data, columns=cols, index=[df_idx]))

                # Custom Text Comparison
                if compare_custom_text:
                    st.subheader('Custom Text Comparison', divider='gray')
                    st.markdown("Each circle represents a different MTLD measure for the given custom text")

                    st.markdown(":black_circle: - MTLD &nbsp;&nbsp; :red_circle: - Bidirectional MTLD &nbsp;&nbsp; :large_blue_circle: - Wrappint MTLD")

                    fig = plt.figure() 
                    all_data = pd.DataFrame(all_rows, columns=cols, index=indeces)
                    plt.plot(all_data)

                    custom_text_split = custom_text.split()
                    x_axis = len(custom_text_split) if vocab_or_len else len(set(custom_text_split))
                    plt.plot(x_axis, ld.mtld(custom_text_split), 'ko')
                    if show_bidir: plt.plot(x_axis, ld.mtld_ma_bid(custom_text_split), 'ro', alpha=0.85)
                    if show_wrap: plt.plot(x_axis, ld.mtld_ma_wrap(custom_text_split), 'bo', alpha=0.85)
                    st.pyplot(fig)
                    