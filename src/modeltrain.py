import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation
import numpy as np
import json

def trainmodel(X_train, y_train, X_test, y_test):
    with tf.device('/cpu:0'):
        # Reshape input data to add the batch size dimension
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the model
        model = Sequential()
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
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

        # Define your model configuration
        model_config = {
            "layers": [
                {
                    "name": "LSTM",
                    "config": {
                        "units": 128,
                        "dropout": 0.2,
                        "recurrent_dropout": 0.2,
                        "input_shape": (X_train.shape[1], 1)
                    }
                },
                {
                    "name": "Dense",
                    "config": {
                        "units": 1,
                        "activation": "sigmoid"
                    }
                }
            ],
            "compile_args": {
                "loss": "binary_crossentropy",
                "optimizer": "adam",
                "metrics": ["accuracy"]
            }
        }

        # Save the configuration to a JSON file
        with open('/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model/config.json', 'w') as config_file:
            json.dump(model_config, config_file, indent=4)

        print("Model configuration saved to config.json")

        return model

