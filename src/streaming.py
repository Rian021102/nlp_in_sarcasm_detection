import time
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
from datasets import load_dataset
import numpy as np
from huggingface_hub import from_pretrained_keras



# Define a function to tokenize and pad the data
def tokenize_pad(data, tokenizer, max_sequence_length=34):
    sequences = tokenizer.texts_to_sequences(data)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

def predict_one_row(model, tokenizer, headline):
    # Tokenize and pad the input data for one row
    sequence = tokenizer.texts_to_sequences([headline])
    padded_sequence = pad_sequences(sequence, maxlen=34)

    # Make a prediction for one row
    prediction = model.predict(padded_sequence)

    # Convert the prediction to a label
    label = 'sarcastic' if prediction[0][0] > 0.5 else 'non-sarcastic'

    return label

def main():
    # Load the dataset from Hugging Face
    dataset = load_dataset("Rianknow/sarcastic_headline", split='test', streaming=True)
    
    # Initialize an empty list to collect headlines
    headlines = []
    
    # Access the dataset splits (e.g., 'train', 'validation', 'test')
    #split_name = 'test'  # Change this to the desired split
    
    # Collect headlines using a loop
    for data_point in dataset:
        headlines.append(data_point['headline'])

    # Create a DataFrame from the collected headlines
    df = pd.DataFrame({'headline': headlines})
    df=df.head(100)
    
    # Create and fit the tokenizer (reuse the one from the first code)
    tokenizer = Tokenizer(num_words=2000, split=' ')
    tokenizer.fit_on_texts(df['headline'])
    
    # Specify the path to your trained model (reuse the one from the first code)
    model_filepath = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/model.h5'
    
    # Load the model
    model = tf.keras.models.load_model(model_filepath)

    # Start processing data one row at a time
    for new_data_point in df['headline']:
        headline = new_data_point

        # Perform prediction for one row
        label = predict_one_row(model, tokenizer, headline)

        # Print the prediction result along with the headline text
        print(f"Prediction for '{headline}': {label}")

if __name__ == "__main__":
    main()