import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dense, GlobalAveragePooling1D, BatchNormalization, SimpleRNN
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def trymodeling(X_train,y_train, X_test,y_test):
    with tf.device('/cpu:0'):
        model = tf.keras.Sequential([
        Embedding(2000, 20, input_length=20),
        SimpleRNN(64, return_sequences=True),
        SimpleRNN(32),
        Dense(64, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())

        # earlystop = EarlyStopping(monitor='val_loss',
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=5,
                                                     mode='min',
                                                     restore_best_weights=True)
        modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/RNNmodel.h5',
            monitor='val_loss',
            save_best_only=True)
        
        num_epochs = 100
        history = model.fit(X_train, y_train,
                            epochs=num_epochs,
                            batch_size=32,
                            validation_data=(X_test, y_test),
                            verbose=1)
        
        #evaluate model
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
        
        