import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dense, GlobalAveragePooling1D
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation
import numpy as np

def trymodeling(X_train,y_train, X_test,y_test):
    with tf.device('/cpu:0'):
        model = tf.keras.Sequential([Embedding(2000, 20, input_length=20),
                                     GlobalAveragePooling1D(),
                                     Dense(64, activation='relu'),
                                     Dense(32, activation='relu'),
                                     Dense(1, activation='sigmoid')
                                     ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        num_epochs = 50
        history = model.fit(X_train, y_train,
                            epochs=num_epochs,
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

        # Save the model
        model.save('/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/model.h5')
        
        return model
        
        