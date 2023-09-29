from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import os



class CustomModelTrainer:
    def __init__(self):
        pass

    @staticmethod
    def build_model(hp,X_train):
        with tf.device('/cpu:0'):
            model = Sequential()
            model.add(LSTM(hp.Int('units',min_value=32,max_value=512,step=32), input_shape=((X_train.shape[1], 1))))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam',metrics = [tf.keras.metrics.MeanSquaredError()])
            return model

    def tune_hyperparameters(self, X_train, y_train):
        tuner = RandomSearch(
            self.build_model,
            objective='mean_squared_error',
            max_trials=10,  # Adjust the number of trials as needed
            directory='keras_tuner',  # Directory to store logs and results
            project_name='custom_model'
        )

        # Define a callback to stop training early if necessary
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[stop_early])

        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hps

    def train_best_model(self, X_train, y_train, X_test, y_test):
        best_hps = self.tune_hyperparameters(X_train, y_train)

        # Build the best model with the tuned hyperparameters
        best_model = self.build_model(best_hps)

        # Train the best model
        best_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluate the best model
        self.evaluate_model(best_model, X_test, y_test, save_path="/Users/rianrachmanto/miniforge3/project/sarcastic_detection/model")
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, save_path=None):
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)  # Adjust the threshold as needed
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model.save(os.path.join(save_path, 'trained_model.h5'))