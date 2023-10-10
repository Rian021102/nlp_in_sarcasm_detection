from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

def tokenize_pad_split(X_train, X_test):
    max_features = 2000
    max_seq_length = 35  # Set the desired sequence length

    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(X_train['headline'].values)

    X_train = tokenizer.texts_to_sequences(X_train['headline'].values)
    X_train = pad_sequences(X_train, maxlen=max_seq_length)  # Pad to a fixed length

    tokenizer.fit_on_texts(X_test['headline'].values)
    X_test = tokenizer.texts_to_sequences(X_test['headline'].values)
    X_test = pad_sequences(X_test, maxlen=max_seq_length)  # Pad to a fixed length

    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test
