from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

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
