# IMDB Movie Sentiment Analysis using Simple RNN

This project demonstrates how to build and deploy a sentiment analysis model for IMDB movie reviews. The model is a Simple Recurrent Neural Network (RNN) implemented using TensorFlow and Keras. It is trained on the standard IMDB dataset to classify movie reviews as either "Positive" or "Negative". A web-based user interface is provided through a Streamlit application.

## Features

* **Sentiment Analysis**: Classifies movie reviews into positive or negative categories.
* **Simple RNN Model**: Utilizes a basic RNN architecture for sequence processing.
* **Word Embeddings**: Employs an Embedding layer to represent words as dense vectors.
* **Interactive Web App**: A Streamlit application allows users to input their own reviews and get instant sentiment predictions.
* **Jupyter Notebooks**: Includes notebooks for both training the model from scratch and for making predictions.

## Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* Numpy
* Pandas

## File Descriptions

* **simpleRNN.ipynb**: A Jupyter notebook that covers the complete process of building and training the Simple RNN model. This includes loading the IMDB dataset, preprocessing the data (padding sequences), defining the model architecture, and training it.
* **prediction.ipynb**: This notebook demonstrates how to load the pre-trained model and use it to make sentiment predictions on new, unseen review text. It contains helper functions for preprocessing user input to match the model's expected format.
* **embedding.ipynb**: A supplementary notebook that explains the concept of word embeddings. It shows how to convert text into one-hot representations and then into dense embedding vectors using Keras' `Embedding` layer.
* **app.py**: A Python script that creates a simple web application using Streamlit. Users can enter a movie review into a text box, and the app will display the predicted sentiment and the corresponding confidence score.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/imdb-sentiment-analysis.git](https://github.com/your-username/imdb-sentiment-analysis.git)
    cd imdb-sentiment-analysis
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing the necessary packages like tensorflow, streamlit, pandas, and numpy).*

## Usage

### Training the Model

To train the model yourself, you can run the `simpleRNN.ipynb` notebook in a Jupyter environment. This will generate the `simple_rnn_model.h5` file.

### Running the Prediction Notebook

The `prediction.ipynb` notebook can be used to see how predictions are made on individual reviews using the saved model.

### Running the Streamlit Web App

To start the interactive sentiment analysis application, run the following command in your terminal:

```bash
streamlit run app.py
