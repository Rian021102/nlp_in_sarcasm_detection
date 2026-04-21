from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def tokenize_pad_split(df):
    max_features = 2000
    maxlen = 20

    X = df[['headline']]
    y = df[['is_sarcastic']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build vocabulary from training data only (0=pad, 1=unknown)
    word_counts = {}
    for text in X_train['headline']:
        for word in str(text).split():
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)[:max_features - 2]
    word_index = {word: idx + 2 for idx, word in enumerate(vocab)}

    def encode_and_pad(texts):
        sequences = []
        for text in texts:
            seq = [word_index.get(w, 1) for w in str(text).split()]
            seq = seq[:maxlen]
            padded = [0] * (maxlen - len(seq)) + seq
            sequences.append(padded)
        return np.array(sequences, dtype=np.int64)

    X_train = encode_and_pad(X_train['headline'].values)
    X_test = encode_and_pad(X_test['headline'].values)

    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test, y_train, y_test


def calculate_class_weights(y_train):
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights[class_label] = class_weight

    #
    return class_weights
