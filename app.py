import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Load the trained model
model = load_model('simple_rnn_model.h5')

# Set up the Streamlit app title
st.title("IMDB Movie Sentiment Analysis")

# Create a text area for user input
user_review = st.text_area("Enter your movie review here:")

# Load the IMDB word index
word_index = imdb.get_word_index()

def preprocess_input(text):
    # The vocabulary size the model was trained with
    max_features = 10000

    words = text.lower().split()
    encoded_review = []

    # Iterate through each word to check its index against the vocabulary size
    for word in words:
        index = word_index.get(word)

        # Only use the index if it exists and is within the model's vocabulary
        if index is not None and index < max_features:
            encoded_review.append(index + 3)

    # Pad the sequence to the required length (400)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=400)
    return padded_review

def predict_review(text):
    """Preprocesses text and returns the sentiment and score."""
    padded_review = preprocess_input(text)
    prediction = model.predict(padded_review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Create a button to trigger the prediction
if st.button("Predict Sentiment"):
    if user_review:
        # If there is input, make a prediction
        sentiment, score = predict_review(user_review)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Score:** {score:.4f}")
    else:
        # If there is no input, show a warning
        st.warning("Please enter a review to analyze.")
