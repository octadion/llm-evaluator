import pandas as pd
from nltk.translate.bleu_score import (
    corpus_bleu,
    SmoothingFunction
)
import streamlit as st
from nltk.translate.meteor_score import meteor_score
import numpy as np
import base64
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import nltk
nltk.download("wordnet")
# st.set_page_config("LLM Evaluator", page_icon=":gear:")
st.header("LLM Evaluator")

with st.sidebar:
    with st.expander("Meteor Parameters"):
        alpha = st.number_input(
            "Alpha",
            value=0.9,
            help="parameter for controlling relative weights of precision and recall.",
        )
        beta = st.number_input(
            "Beta",
            value=3,
            help="parameter for controlling shape of penalty as a function of fragmentation.",
        )
        gamma = st.number_input(
            "Gamma",
            value=0.5,
            help="relative weight assigned to fragmentation penalty.",
        )

    with st.expander("BLEU Parameters"):
        n_weights = st.number_input("Max Order of N-Grams", 1, 10, 4, 1, "%d")
        weights = []
        for i in range(1, n_weights + 1):
            weight = st.number_input(
                f"Weight of {i}-gram", 0.0, 2.0, 1 / n_weights, format="%.2f"
            )
            weights.append(weight)
        smoothing_function = st.toggle(
            "Smoothing Function", False, help="Option to smooth the harsh (0) scores."
        )
        auto_reweigh = st.toggle(
            "Auto Reweigh", False, help="Option to re-normalize the weights uniformly."
        )
        weights = tuple(weights)
        smoothing_function = (
            SmoothingFunction().method1 if smoothing_function else None
        )

def calc_bleu(references, hypothesis, weights, smoothing_function, auto_reweigh):
    bleu_score = []

    for ref, hyp in zip(references, hypothesis):
        reference = []
        hypo = []
        reference.append([ref_text.split() for ref_text in ref.splitlines()])
        hypo.append(hyp.split())
        score = corpus_bleu(reference, hypo, weights=weights, smoothing_function=smoothing_function, auto_reweigh=auto_reweigh)
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
        for expect, predict in zip(reference, hypo):
            score = meteor_score(expect, predict)
            meteor_score_sentences_list.append(score)
        meteor_score_res = float(np.mean(meteor_score_sentences_list))
        meteor.append(meteor_score_res)
    return meteor

def evaluate(file_input, weights, smoothing_function,
             auto_reweigh):
    df = pd.read_csv(file_input)

    references = df['reference'].tolist()
    hypothesis = df['hypothesis'].tolist()
    bleu_score = calc_bleu(references, hypothesis, weights, smoothing_function, auto_reweigh)
    meteor_score = calc_meteor(references, hypothesis)
    print(meteor_score)
    print(bleu_score)
    return df, bleu_score, meteor_score
def create_download_link(df, title="Download CSV file", filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    return html

with st.form("corpus_level_file_upload", clear_on_submit=True):
    file_input = st.file_uploader("Upload CSV", type=['csv'])
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        if file_input is not None:
            df, bleu_score, meteor_score = evaluate(file_input, weights, smoothing_function, auto_reweigh)
            df['bleu_score'] = bleu_score
            df['meteor_score'] = meteor_score
            st.dataframe(df, use_container_width=True)
            avg_bleu = sum(bleu_score) / len(bleu_score)
            avg_meteor = sum(meteor_score) / len(meteor_score)
            st.write(f"Rata-rata skor BLEU keseluruhan: {avg_bleu:.2f}")
            st.write(f"Rata-rata skor Meteor keseluruhan: {avg_meteor:.2f}")
            avg = {'bleu_score': f"Rata-rata skor BLEU keseluruhan: {avg_bleu:.2f}",
                      'meteor_score': f"Rata-rata skor Meteor keseluruhan: {avg_meteor:.2f}"}
            df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
            st.markdown(create_download_link(df), unsafe_allow_html=True)

def app():
    st.title('Score for file upload')