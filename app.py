import streamlit as st
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datasets import load_dataset

# Function to tokenize and pad input data
def tokenize_pad(df, max_sequence_length=34):
    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(df['headline'].values)
    sequences = tokenizer.texts_to_sequences(df['headline'].values)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# Function to make predictions
def predict_model(df, model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Tokenize and pad the input data
    padded_sequences = tokenize_pad(df)
    
    # Convert the padded_sequences to a NumPy array
    padded_sequences = np.array(padded_sequences)
    
    # Make predictions
    predictions = model.predict(padded_sequences)
    
    # Convert predictions to labels
    df['label'] = (predictions > 0.5)
    df['label'] = df['label'].apply(lambda x: 'sarcastic' if x else 'non-sarcastic')
    
    return df

def main():
    # Load the dataset from Hugging Face
    dataset = load_dataset("Rianknow/sarcastic_headline", split='test', streaming=True)

    df = pd.DataFrame(dataset)

    # Specify the path to your trained model
    model_filepath = 'model/model.h5'

    # Create a Streamlit app
    st.title("Sarcastic Headline Detection")

    # Load data and display as dataframe
    if st.button("Load Data"):
        st.write(df)

    # Predict and display results
    if st.button("Predict"):
        result_df = predict_model(df, model_filepath)
        st.write("Predicted Results:")
        st.write(result_df)

if __name__ == "__main__":
    main()
