import streamlit as st
import pandas as pd
import os
import nltk
from transformers import pipeline
nltk.download('vader_lexicon')
nltk.download('movie_reviews')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as senti

sia = senti()
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")



st.markdown("<h1 style='text-align: center; '>I T F</h1>", unsafe_allow_html=True)
st.markdown("***")
st.markdown("<h1 style='text-align: center; '>Insight From Text</h1>", unsafe_allow_html=True)
st.markdown("***")
user_input = st.text_area("ENTER THE TEXT DATA THAT HAVE TO BE SUMMARIZED : ", height=300)
slider = st.slider('Length of summary data should be summarized (in words) ', 50, 250)
if st.button('           Summarize and get Polarity           '):
    st.markdown("***")
    with st.spinner(text="This may take a moment..."):
        st.subheader("Summary :")
        result=summarizer(user_input, min_length=5, max_length=slider)
        ffinal=result[0]['summary_text']
        valu=sia.polarity_scores(str(ffinal))
    st.write('%s' % ffinal)
    st.markdown("***")
    st.subheader("Polarity :")
    st.text('(The polarity scores varies with number of words in the generated text.) ')
    st.write(str(valu))
    st.bar_chart(list(valu))
    st.markdown("***")
    st.download_button('Download summrized text :', ffinal)
