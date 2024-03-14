from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
#import compute_class_weight
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def tokenize_pad_split(df):
    max_features = 2000

    X=df[['headline']]
    y=df[['is_sarcastic']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(X_train['headline'])

    X_train = tokenizer.texts_to_sequences(X_train['headline'].values)
    X_train = pad_sequences(X_train,maxlen=20)  # Pad to a fixed length

    tokenizer.fit_on_texts(X_test['headline'].values)
    X_test = tokenizer.texts_to_sequences(X_test['headline'].values)
    X_test = pad_sequences(X_test, maxlen=20)  # Pad to a fixed length

    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test,y_train,y_test


def calculate_class_weights(y_train):
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights[class_label] = class_weight

    #
    return class_weights
