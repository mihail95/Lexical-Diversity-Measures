import streamlit as st
import numpy as np
import pandas as pd
import statistics
from measures.helpers.text_generation import generate_repeating_text_naive, generate_random_text_naive, generate_random_text_zipf, genearate_alphabet_permuatations
from lexical_diversity import lex_div as ld
import matplotlib.pyplot as plt

def ttr():
    st.header('Token-Type Ratio (TTR)', divider='gray')
    st.markdown("TTR is an long-established method of measuring lexical diversity, develeped by Wendell Johnson to help measure vocabulary \"flexibility\" or \"variability\" (Johnson, 1944). This measure uses a simple ratio of types (unique words in a sample) to tokens (all words in the sample).")
    st.markdown("A significant flaw of this method, however, is that the TTR Score of a sample is affected by the sample's lenght. Several approaches (Root TTR, Log TTR, Maas TTR) have been proposed with the claim of modifying the calculation to be length-agnostic, but none of them have been found to actually succeed in this task (Covington & McFall, 2010; Fergadiotis et al., 2015).")

    with st.expander("Measure Calculations", icon=":material/expand_circle_down:"):
        st.write("All measures use the vocabulary size (token count) **V** and the number of tokens in the text **N** for the calculation.")
        st.markdown('''
            <style>
            .katex-html {
                text-align: left;
            }
            </style>''',
            unsafe_allow_html=True
        )
        st.latex(r'''\textbf{TTR} = \frac{V}{N}''')
        st.latex(r'''\textbf{Root TTR} = \frac{V}{\sqrt{N}}\hspace{2mm} \text{(Guiraud, 1954, as cited in Tweedie \& Baayen, 1998)}''')
        st.latex(r'''\textbf{Log TTR} = \frac{log\ V}{log\ N}\hspace{2mm} \text{(Herdan, 1960, as cited in Tweedie \& Baayen, 1998)}''')
        st.latex(r'''\textbf{Maas TTR} = \frac{log\ N - log\ V}{log\ N^2}\hspace{2mm} \text{(Maas, 1972, as cited in Tweedie \& Baayen, 1998)}''')

    st.subheader('Dynamic Graph', divider='gray')
    # False = Vary Vocab; True = Vary Text Length
    vocab_or_len = st.toggle("Vary Vocabulary Size / Vary Text Length", value= False, help="Left = Vary Vocabulary; Right = Vary Text Length")

    columns = st.columns([5,5,5,1,1,1,1,1,1,1,1,1,1,1,1])
    with columns[0]:
        show_root_ttr = st.toggle("Root TTR", value= False)
    with columns[1]:
        show_log_ttr = st.toggle("Log TTR", value= False)
    with columns[2]:
        show_maas_ttr = st.toggle("Maas TTR", value= False)
    
    compare_custom_text = st.toggle("Custom Text Comparison", value= False)

    with st.form("TTR-Form"):
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

                ########## TTR MAIN LOOP ##################################
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

                cols = ["ttr"]
                if show_root_ttr: cols.append("root_ttr")
                if show_log_ttr: cols.append("log_ttr")
                if show_maas_ttr: cols.append("maas_ttr")

                all_rows = []
                indeces = []

                for curr_vocab_size in range(min_vocab_size, max_vocab_size + 1):
                    vocab = set(alphabet[:curr_vocab_size])
                    for curr_text_len in range(min_text_length, max_text_length + 1):
                        len_ttr_scores = []
                        len_root_ttr_scores = []
                        len_log_ttr_scores = []
                        len_maas_ttr_scores = []
                        for _ in range(smoothing_runs):
                            text = gen_function(vocab, curr_text_len).split()
                            len_ttr_scores.append(ld.ttr(text))
                            if show_root_ttr: len_root_ttr_scores.append(ld.root_ttr(text))
                            if show_log_ttr: len_log_ttr_scores.append(ld.log_ttr(text))
                            if show_maas_ttr: len_maas_ttr_scores.append(ld.maas_ttr(text))
                            status_text.text(f"{round(progress_ctr/total_cycles, 2) * 100}% Complete")
                            progress_bar.progress(round(progress_ctr/total_cycles, 2))
                            progress_ctr += 1
                        ttr_score = statistics.mean(len_ttr_scores)
                        data = [ttr_score]
                        df_idx = curr_text_len if vocab_or_len else curr_vocab_size
                        all_rows.append(data)
                        indeces.append(df_idx)
                        if show_root_ttr or show_log_ttr or show_maas_ttr:
                            data = [data]
                            if show_root_ttr:
                                root_ttr_score = statistics.mean(len_root_ttr_scores)
                                data[0].append(root_ttr_score)
                            if show_log_ttr:
                                log_ttr_score = statistics.mean(len_log_ttr_scores)
                                data[0].append(log_ttr_score)
                            if show_maas_ttr:
                                maas_ttr_score = statistics.mean(len_maas_ttr_scores)
                                data[0].append(maas_ttr_score)

                        
                        chart.add_rows(pd.DataFrame(data, columns=cols, index=[df_idx]))

                # Custom Text Comparison
                if compare_custom_text:
                    st.subheader('Custom Text Comparison', divider='gray')
                    st.markdown("Each circle represents a different TTR measure for the given custom text")

                    st.markdown(":black_circle: - TTR &nbsp;&nbsp; :red_circle: - Root TTR &nbsp;&nbsp; :large_blue_circle: - Log TTR &nbsp;&nbsp; :large_blue_diamond: - Maas TTR")

                    fig = plt.figure() 
                    all_data = pd.DataFrame(all_rows, columns=cols, index=indeces)
                    plt.plot(all_data)

                    custom_text_split = custom_text.split()
                    x_axis = len(custom_text_split) if vocab_or_len else len(set(custom_text_split))
                    plt.plot(x_axis, ld.ttr(custom_text_split), 'ko')
                    if show_root_ttr: plt.plot(x_axis, ld.root_ttr(custom_text_split), 'ro', alpha=0.85)
                    if show_log_ttr: plt.plot(x_axis, ld.log_ttr(custom_text_split), 'bo', alpha=0.85)
                    if show_maas_ttr: plt.plot(x_axis, ld.maas_ttr(custom_text_split), 'bD', alpha=0.85)
                    st.pyplot(fig)

    st.write("------------------")
    st.write("#### References")
    st.write("Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio (MATTR). Journal of Quantitative Linguistics, 17(2), 94–100. https://doi.org/10.1080/09296171003643098")
    st.write("Fergadiotis, G., Wright, H. H., & Green, S. B. (2015). Psychometric Evaluation of Lexical Diversity Indices: Assessing Length Effects. Journal of speech, language, and hearing research : JSLHR, 58(3), 840–852. https://doi.org/10.1044/2015_JSLHR-L-14-0280")
    st.write("Johnson, W. (Ed.). (1944). Studies in language behavior: A program of research. Psychological Monographs, 56(2), 1-15.")
    st.write("Tweedie. F.J. and Baayen, R.H. (1998). How Variable May a Constant Be? Measures of Lexical Richness in Perspective. Computers and the Humanities, 32(5), 323-352.")