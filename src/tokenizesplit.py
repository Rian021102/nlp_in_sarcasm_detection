from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
def tokenize_pad_split(df):
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(df['headline'].values)
    X = tokenizer.texts_to_sequences(df['headline'].values)
    X = pad_sequences(X)
    y=df['is_sarcastic'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)
    return X_train, X_test, y_train, y_test