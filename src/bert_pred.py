import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFAutoModelForSequenceClassification
import pandas as pd

def load_data_for_prediction(path):
    df = pd.read_csv(path)
    df.rename(columns={'headline': 'text'}, inplace=True)
    return df

def encode_data_for_prediction(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128)

def predict_and_append(model_path, tokenizer_path, data_path):
    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load and prepare the data
    df = load_data_for_prediction(data_path)
    encodings = encode_data_for_prediction(tokenizer, list(df['text']))
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(dict(encodings)).batch(32)
    
    # Prediction
    logits = model.predict(dataset).logits
    predictions = tf.argmax(logits, axis=1)
    
    # Append predictions to the DataFrame
    df['prediction'] = predictions.numpy()
    return df

# Specify your paths
model_path = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/distilbert_model'
tokenizer_path = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/distilbert_tokenizer'
data_path = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/test.csv'  # Update this path to your actual data file

# Make predictions and get the updated DataFrame
updated_df = predict_and_append(model_path, tokenizer_path, data_path)
print(updated_df.head())
#save to csv
updated_df.to_csv('/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/data_with_predictions.csv', index=False)  # Update the path to save the file