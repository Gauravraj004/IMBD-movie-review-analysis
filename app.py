import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.datasets import imdb

model = load_model('simple_rnn_model.h5')

st.title("IMDB Movie Sentiment Analysis")

user_review = st.text_area("Enter your movie review here:")

word_index = imdb.get_word_index()


def preprocess_input(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=400)  
    return padded_review

def predict_review(text):

    padded_review = preprocess_input(text)
    prediction = model.predict(padded_review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


if st.button("Predict Sentiment"):
    if user_review:
        sentiment, score = predict_review(user_review)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Score: {score:.4f}")
    else:
        st.warning("Please enter a review to analyze.")
