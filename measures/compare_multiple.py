import streamlit as st
import spacy
import numpy as np
from lexical_diversity import lex_div as ld
import altair as alt
import pandas as pd
from measures.helpers.text_generation import generate_repeating_text_naive, generate_random_text_naive, generate_random_text_zipf, genearate_alphabet_permuatations

          

def compare():
    st.header('Compare Multiple Measures', divider='gray')

    if "text_input_compare_count" not in st.session_state:
        st.session_state["text_input_compare_count"] = 1

    columns = st.columns([4,1,1,1,3,3,3,3])
    with columns[0]:
        st.markdown("**Add/Remove Text Field (max 5)**")
    with columns[1]:
        button_add = st.button(r"**+**")
    with columns[2]:
        button_sub = st.button(r"**-**")
    with columns[4]:
        show_ttr = st.toggle("TTR", value= False)
    with columns[5]:
        show_mattr = st.toggle("MATTR", value= False)
    with columns[6]:
        show_hdd = st.toggle("HDD", value= False)
    with columns[7]:
        show_mtld = st.toggle("MTLD", value= False)
    
    if button_add and st.session_state["text_input_compare_count"] < 5: 
        st.session_state["text_input_compare_count"] += 1
    if button_sub and st.session_state["text_input_compare_count"] > 1:
        st.session_state["text_input_compare_count"] -= 1
    
    columns = st.columns([4,2,7])
    text_gen_alg = None
    with columns[0]:
        show_generated_text = st.toggle("Add generated text", value= False, help="Generates and evaluates a text, using the selected generation algorithm and the vocabulary and length of text1.")
    with columns[2]:
        if show_generated_text:
            text_gen_alg = st.selectbox(
            "Which text generation algorithm should be used?",
            ("Sequential", "Random", "Zipf Distribution"),
            index=None,
            placeholder="Select text generation algorithm...",
            help="Sequential: Repeats all vocabulary items in the same order until the maximum text length is reached\n\nRandom: Picks random vocabulary types (each has the same probability\n\nZipf Distribution: Picks random vocabulary types from a zipf distribution (value of the n-th entry is inversely proportional to n)"
        )
    
    lemmatize_lang = None
    with columns[0]:
        lemmatize_texts = st.toggle("Lemmatize texts", value= False, help="Toggles the lemmatization for the custom texts.")
    with columns[2]:
        if lemmatize_texts:
            lemmatize_lang = st.selectbox(
            "What language are the texts in?",
            ("German", "English"),
            index=None,
            placeholder="Select text language..."
        )

    with st.form("Compare-Form"):
        texts = { f"text{ctr + 1}" : "" for ctr in range(st.session_state['text_input_compare_count']) }

        mattr_window_length = 50
        mtld_min_segment_len = 10
        if show_mattr:
            mattr_window_length = st.slider("MATTR Window Length", min_value=1, max_value=200, value=50, step=1)
        if show_mtld:
            mtld_min_segment_len = st.slider("MTLD Minimum Segment Length", min_value=1, max_value=200, value=10, step=1)

        for text in texts.keys():
            texts[text] = st.text_area(f"Enter a new text to compare (ID: {text}) ...", key=text)
        
        if st.form_submit_button("Run Comparison"):
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            progress_bar.empty()
            progress_ctr = 1
            
            text_gen_map = {
                "Sequential": generate_repeating_text_naive,
                "Random": generate_random_text_naive,
                "Zipf Distribution": generate_random_text_zipf,
            }
            gen_text = ""
            if text_gen_alg != None:
                gen_function = text_gen_map[text_gen_alg]
                text1 = texts['text1'].split()
                vocab = set(text1)
                text_len = len(text1)
                gen_text = gen_function(vocab, text_len)
                texts['generated_text'] = gen_text


            if lemmatize_lang != None:
                lemmatize_lang_map = {
                    "German": ("de_core_news_sm", ['parser', 'senter', 'ner']), 
                    "English": ("en_core_web_sm", ['parser', 'senter', 'ner'])
                }
                lemmatize_options = lemmatize_lang_map[lemmatize_lang]
                nlp = spacy.load(lemmatize_options[0], disable = lemmatize_options[1])

                for text_id, text in texts.items():
                    doc = nlp(text)
                    texts[text_id] = " ".join([token.lemma_ for token in doc])

            cols = ['TextID', 'Text Length', 'Vocabulary Size']
            if show_ttr: cols.append('TTR')
            if show_mattr: cols.append('MATTR')
            if show_hdd: cols.append('HDD')
            if show_mtld: cols.append('MTLD')

            texts_df = pd.DataFrame(columns=cols)               
            plot_data = []
            # Calculate measures for all texts and write them in an array of objects
            # example: [{"id": "text1", "measure": "ttr", "value": 22}, {"id": "text1", "measure": "mattr", "value": 12}, {"id": "text2", "measure": "ttr", "value": 43}]

            for text_id, text in texts.items():
                text_tokens = text.split()

                if len(text_tokens) == 0:
                    progress_ctr += 1
                    continue
                
                text_metrics = []
                if show_ttr:
                    ttr = ld.ttr(text_tokens)
                    plot_data.append({"id": text_id, "measure": "ttr", "value": ttr})
                    text_metrics.append(ttr)

                if show_mattr:
                    mattr = ld.mattr(text_tokens, mattr_window_length)
                    plot_data.append({"id": text_id, "measure": "mattr", "value": mattr})
                    text_metrics.append(mattr)

                if show_hdd:
                    hdd = ld.hdd(text_tokens)
                    plot_data.append({"id": text_id, "measure": "hdd", "value": hdd})
                    text_metrics.append(hdd)
                
                if show_mtld:
                    mtld = ld.mtld(text_tokens, mtld_min_segment_len)
                    plot_data.append({"id": text_id, "measure": "mtld", "value": mtld})
                    text_metrics.append(mtld)

                texts_df.loc[len(texts_df)] = [text_id, len(text.split()), len(set(text.split())), *text_metrics]

                status_text.text(f"{round(progress_ctr/(st.session_state['text_input_compare_count'] + int(show_generated_text)), 2) * 100}% Complete")
                progress_bar.progress(round(progress_ctr/(st.session_state["text_input_compare_count"] + int(show_generated_text)), 2))
                progress_ctr += 1

            df = pd.DataFrame(plot_data)

            if len(gen_text) > 0:
                st.text_area(f"Automatically generated text using the {text_gen_alg} algorithm.",placeholder= gen_text, disabled= True)
            
            st.dataframe(data = texts_df, hide_index=True)

            if not df.empty:
                if show_ttr:
                    st.subheader("TTR Chart", divider='gray')
                    ttr_chart = alt.Chart(df[df["measure"] == "ttr"]).mark_bar().encode(x = "id", y = "value", color = "id")
                    st.altair_chart(ttr_chart, theme=None, use_container_width=True)

                if show_mattr:
                    st.subheader("MATTR Chart", divider='gray')
                    mattr_chart = alt.Chart(df[df["measure"] == "mattr"]).mark_bar().encode(x = "id", y = "value", color = "id")
                    st.altair_chart(mattr_chart, theme=None, use_container_width=True)

                if show_hdd:
                    st.subheader("HDD Chart", divider='gray')
                    hdd_chart = alt.Chart(df[df["measure"] == "hdd"]).mark_bar().encode(x = "id", y = "value", color = "id")
                    st.altair_chart(hdd_chart, theme=None, use_container_width=True)
                
                if show_mtld:
                    st.subheader("MTLD Chart", divider='gray')
                    mtld_chart = alt.Chart(df[df["measure"] == "mtld"]).mark_bar().encode(x = "id", y = "value", color = "id")
                    st.altair_chart(mtld_chart, theme=None, use_container_width=True)
