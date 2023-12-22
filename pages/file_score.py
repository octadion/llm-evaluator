import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
import streamlit as st
from nltk.translate.meteor_score import meteor_score
import numpy as np
import nltk
nltk.download("wordnet")
# st.set_page_config("LLM Evaluator", page_icon=":gear:")
st.header("LLM Evaluator")
def calc_bleu(references, hypothesis):
    bleu_score = []

    for ref, hyp in zip(references, hypothesis):
        reference = []
        hypo = []
        reference.append([ref_text.split() for ref_text in ref.splitlines()])
        hypo.append(hyp.split())
        score = corpus_bleu(reference, hypo)
        bleu_score.append(score)
    return bleu_score
def calc_meteor(references, hypothesis):
    meteor = []

    for ref, hyp in zip(references, hypothesis):
        reference = []
        hypo = []
        reference.append([ref_text.split() for ref_text in ref.splitlines()])
        hypo.append(hyp.split())
        meteor_score_sentences_list = list()
        [
            meteor_score_sentences_list.append(meteor_score(expect, predict))
            for expect, predict in zip(reference, hypo)
        ]
        meteor_score_res = float(np.mean(meteor_score_sentences_list))
        meteor.append(meteor_score_res)
    return meteor
def evaluate(file_input):
    df = pd.read_csv(file_input)

    references = df['reference'].tolist()
    hypothesis = df['hypothesis'].tolist()
    bleu_score = calc_bleu(references, hypothesis)
    meteor_score = calc_meteor(references, hypothesis)
    print(meteor_score)
    print(bleu_score)
    return df, bleu_score, meteor_score

with st.form("corpus_level_file_upload", clear_on_submit=True):
    file_input = st.file_uploader("Upload CSV", type=['csv'])
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        if file_input is not None:
            df, bleu_score, meteor_score = evaluate(file_input)
            df['bleu_score'] = bleu_score
            df['meteor_score'] = meteor_score
            st.dataframe(df, use_container_width=True)
            avg_bleu = sum(bleu_score) / len(bleu_score)
            avg_meteor = sum(meteor_score) / len(meteor_score)
            st.write(f"Rata-rata skor BLEU keseluruhan: {avg_bleu:.2f}")
            st.write(f"Rata-rata skor Meteor keseluruhan: {avg_meteor:.2f}")

def app():
    st.title('Score for file upload')