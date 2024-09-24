import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df.rename(columns={'headline': 'text', 'is_sarcastic': 'label'}, inplace=True)
    return df

def split_data(df):
    X = list(df['text'])
    y = df['label'].values  # Ensure y is a numpy array of integers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def encode_data(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128)

# Path to dataset
path = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/train.csv'
df = load_data(path)
X_train, X_test, y_train, y_test = split_data(df)


# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = encode_data(tokenizer, X_train)
test_encodings = encode_data(tokenizer, X_test)

# TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(32)

# Model
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Compile model with explicit loss function and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Summary and Training
model.summary()
print("Training data types and shapes:")
for inputs, targets in train_dataset.take(1):
    print(f"Inputs: {inputs}, Targets: {targets}")
model.fit(train_dataset, validation_data=test_dataset, epochs=3)
# Evaluate the model
model.evaluate(test_dataset)
# print classification report
y_pred = model.predict(test_dataset)
y_pred = tf.argmax(y_pred.logits, axis=1)
print(classification_report(y_test, y_pred))
# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Save the model
model.save_pretrained('/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/distilbert_model')
tokenizer.save_pretrained('/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/distilbert_tokenizer')
# plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


