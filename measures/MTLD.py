import streamlit as st
import numpy as np
import pandas as pd
import statistics
from measures.helpers.text_generation import generate_repeating_text_naive, generate_random_text_naive, generate_random_text_zipf, genearate_alphabet_permuatations
from lexical_diversity import lex_div as ld
import matplotlib.pyplot as plt

def mtld():
    st.header('Measure of lexical textual diversity (MTLD)', divider='gray')
    st.markdown("MTLD calculates a text's lexical diversity by assessing the mean length of word sequences that maintain a specific type-token ratio (TTR), typically set at 0.720. As each word is added, TTR is recalculated, and once it drops below the threshold, the factor count increases, and the process resets. The final MTLD score is obtained by dividing the total word count by the factor count, including adjustments for incomplete factors (Fergadiotis et al., 2015; McCarthy & Jarvis, 2010).")
    st.markdown("Two variants of the measure are Bidirectional MTLD and Wrapping MTLD. The former is desribed by McCarthy & Jarvis (2010) as simply calculating the MTLD for a reversed copy of the text and then calculating the mean of the two scores. The latter is a variant, implemented in the [lexical_diversity Python library by Kristopher Kyle](https://github.com/kristopherkyle/lexical_diversity) in which no partial factor is calculated - the MTLD calculation wraps around to the start of the sentence until the threshold is reached.")

    with st.expander("MTLD Algorithm", icon=":material/expand_circle_down:"):
        st.markdown(r"1. Define a TTR threshold: &nbsp; $TTR_{max} = 0.72$")
        st.markdown(r"2. Start sequence at first token and calculate $TTR_{i}$")
        st.markdown(r"3. Extend the sequence until $TTR_i < TTR_{max}$")
        st.markdown(r"4. Increment the sequence count and start a new sequence")
        st.markdown(r"5. Repeat until EOT")
        st.markdown(r"6. *conditional:* If last window doesn't reach threshold - calculate the partial factor $TTR_{part}$")
        st.markdown(r"7. Final score = $i_{max} + TTR_{part}$")
        
        st.markdown("**For Bidirectional MTLD:**")
        st.markdown(r"8. Repeat in the other direction (last to first token)")
        st.markdown(r"9. Calculate mean of the two scores")

        st.markdown("**For Wrapping MTLD:**")
        st.markdown(r"6. *conditional:* If last window doesn't reach threshold - wrap to the start of the text and extend the sequence until threshold is reached")
        st.markdown(r"7. Final score = $i_{max}$")

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

    st.write("------------------")
    st.write("#### References")
    st.write("Fergadiotis, G., Wright, H. H., & Green, S. B. (2015). Psychometric Evaluation of Lexical Diversity Indices: Assessing Length Effects. Journal of speech, language, and hearing research : JSLHR, 58(3), 840–852. https://doi.org/10.1044/2015_JSLHR-L-14-0280")
    st.write("McCarthy, P.M., Jarvis, S. MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. Behavior Research Methods 42, 381–392 (2010). https://doi.org/10.3758/BRM.42.2.381")                 