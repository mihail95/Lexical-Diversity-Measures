import streamlit as st
import numpy as np
import pandas as pd
import statistics
from measures.helpers.text_generation import generate_repeating_text_naive, generate_random_text_naive, generate_random_text_zipf, genearate_alphabet_permuatations
from lexical_diversity import lex_div as ld
import matplotlib.pyplot as plt

def mattr():
    st.header('Moving-Average Type-Token Ratio (MATTR)', divider='gray')
    st.markdown("The Moving-Average Type-Token Ratio (Covington & McFall, 2010) attempts to solve the text length problem by calculating TTRs within a sliding window of tokens, starting from the first token and continuing through to the last. Each window has its own TTR, and the MATTR is determined by averaging these TTRs.")

    with st.expander("MATTR Algorithm", icon=":material/expand_circle_down:"):
        st.write("MATTR choses a window $W_i$ of size **W** and uses its vocabulary size (token count) $V_{W_i}$ and the number of tokens in the text $N_{W_i}$ for the calculation.")
        st.markdown("1. Choose window size: *for example* $W = 15$")
        st.markdown(r"2. Calculate TTR for current window: $W_1 = \{N_1 ... N_{15}\};\ TTR_{W_1} = \dfrac{V_{W_1}}{N_{W_1}}$")
        st.markdown(r"3. Shift window to the right by 1 token: $W_2 = \{N_2 ... N_{16}\}$")
        st.markdown("4. End if less than $W$ tokens are left, else go back to step 2")
        st.markdown("5. Calulate mean of all TTRs")


    st.subheader('Dynamic Graph', divider='gray')
    # False = Vary Vocab; True = Vary Text Length
    vocab_or_len = st.toggle("Vary Vocabulary Size / Vary Text Length", value= False, help="Left = Vary Vocabulary; Right = Vary Text Length")
    compare_custom_text = st.toggle("Custom Text Comparison", value= False)

    with st.form("MATTR-Form"):
        text_gen_alg = st.selectbox(
            "Which text generation algorithm should be used?",
            ("Sequential", "Random", "Zipf Distribution"),
            index=None,
            placeholder="Select text generation algorithm...",
            help="Sequential: Repeats all vocabulary items in the same order until the maximum text length is reached\n\nRandom: Picks random vocabulary types (each has the same probability\n\nZipf Distribution: Picks random vocabulary types from a zipf distribution (value of the n-th entry is inversely proportional to n)"
        )

        window_size = st.slider("Window Size", min_value=1, max_value=200, value=15, step=1)

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

                ########## MATTR MAIN LOOP ##################################
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
                        len_ttr_scores = []
                        len_mattr_scores = []
                        for _ in range(smoothing_runs):
                            text = gen_function(vocab, curr_text_len).split()
                            len_ttr_scores.append(ld.ttr(text))
                            len_mattr_scores.append(ld.mattr(text, window_size))
                            status_text.text(f"{round(progress_ctr/total_cycles, 2) * 100}% Complete")
                            progress_bar.progress(round(progress_ctr/total_cycles, 2))
                            progress_ctr += 1
                        ttr_score = statistics.mean(len_ttr_scores)
                        mattr_score = statistics.mean(len_mattr_scores)
                        df_idx = curr_text_len if vocab_or_len else curr_vocab_size
                        chart.add_rows(pd.DataFrame([[mattr_score, ttr_score]], columns=["MATTR", "TTR"], index=[df_idx]))
                        all_rows.append([mattr_score, ttr_score])
                        indeces.append(df_idx)
                
                # Custom Text Comparison
                if compare_custom_text:
                    st.subheader('Custom Text Comparison', divider='gray')
                    st.markdown("Each circle represents a different measure for the given custom text")

                    st.markdown(":black_circle: - TTR &nbsp;&nbsp; :red_circle: - MATTR")

                    fig = plt.figure() 
                    all_data = pd.DataFrame(all_rows, columns=["MATTR", "TTR"], index=indeces)
                    plt.plot(all_data)

                    custom_text_split = custom_text.split()
                    x_axis = len(custom_text_split) if vocab_or_len else len(set(custom_text_split))
                    plt.plot(x_axis, ld.ttr(custom_text_split), 'ko')
                    plt.plot(x_axis, ld.mattr(custom_text_split), 'ro', alpha=0.85)
                    st.pyplot(fig)

    st.write("------------------")
    st.write("#### References")
    st.write("Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio (MATTR). Journal of Quantitative Linguistics, 17(2), 94–100. https://doi.org/10.1080/09296171003643098")