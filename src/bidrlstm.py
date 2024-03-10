import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def trainmodel(X_train, y_train, X_test, y_test):
    with tf.device('/cpu:0'):
        # Reshape input data to add the batch size dimension
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the model
        import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def trainmodelbid(X_train, y_train, X_test, y_test):
    with tf.device('/cpu:0'):
        # Reshape input data to add the batch size dimension
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the model
        model = Sequential()
        model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), input_shape=(X_train.shape[1], 1)))
        model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        # Train the model
        #set early stopping monitor so the model stops training when it won't improve anymore
        earlystop = EarlyStopping(monitor='val_loss', 
                                  patience=5, 
                                  mode='min',
                                  restore_best_weights=True)
        modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/LSTMmodel.h5',
            monitor='val_loss',
            save_best_only=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                            callbacks=[earlystop, modelcheckpoint])
        #evaluate the model
        score, acc = model.evaluate(X_test, y_test, batch_size=64)
        print('Test score:', score)
        print('Test accuracy:', acc)

        # Make predictions without converting to binary
        y_pred = model.predict(X_test)

        # Make predictions with converting to binary
        y_pred = (y_pred > 0.5)

        # Print classification report
        print(classification_report(y_test, y_pred))
        #plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.show()

        #plot epoch vs loss and accuracy in one graph side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Loss (training data)')
        plt.plot(history.history['val_loss'], label='Loss (validation data)')
        plt.title('Loss for LSTM Model')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Accuracy (training data)')
        plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
        plt.title('Accuracy for LSTM Model')
        plt.ylabel('Accuracy value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()
        return model


