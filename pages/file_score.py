import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
import streamlit as st

# st.set_page_config("LLM Evaluator", page_icon=":gear:")
def calc_bleu(references, hypothesis):
    bleu_score = []

    for ref, hyp in zip(references, hypothesis):
        score = corpus_bleu([ref.split()], hyp.split())
        bleu_score.append(score)

    return bleu_score

def bleu_file(file_input):
    df = pd.read_csv(file_input)

    references = df['reference'].tolist()
    hypothesis = df['hypothesis'].tolist()

    skor_bleu = calc_bleu(references, hypothesis)

    return df, skor_bleu

with st.form("corpus_level_file_upload", clear_on_submit=True):
    file_input = st.file_uploader("Upload CSV", type=['csv'])
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        if file_input is not None:
            df, bleu_score = bleu_file(file_input)
            df['bleu_score'] = bleu_score
            st.write(df)
            rata_rata = sum(bleu_score) / len(bleu_score)
            st.write(f"Rata-rata skor BLEU keseluruhan: {rata_rata:.2f}")

def app():
    st.title('Score for file upload')