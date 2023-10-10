import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel
from transformers import pipeline


def tokenize_pad(df, max_sequence_length=35):
    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(df['headline'].values)
    sequences = tokenizer.texts_to_sequences(df['headline'].values)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

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
    dataset = load_dataset("Rianknow/sarcastic_headline", split='test',streaming=True)

    df = pd.DataFrame(dataset)

    # Specify the path to your trained model
    model_filepath = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/model.h5'
    
    # Predict using the model
    df = predict_model(df, model_filepath)

    print(df)
    
    # Save predictions to CSV
    df.to_csv('/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/prediction.csv', index=False)

if __name__ == "__main__":
    main()
