from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
def tokenize_pad_split(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['headline'])
    X = tokenizer.texts_to_sequences(df['headline'])
    X = pad_sequences(X, maxlen=100)
    y = df['is_sarcastic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test