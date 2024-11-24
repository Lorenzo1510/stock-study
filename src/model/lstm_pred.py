import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore


class TimeSeriesPredictor:
    def __init__(self, 
                 df: pd.DataFrame, 
                 feature: str = 'Adj Close', 
                 lookback: int = 60, 
                 out_steps: int = 5, 
                 units: int = 50):
        self.df = df
        self.feature = feature
        self.lookback = lookback
        self.out_steps = out_steps  # Numero di passi futuri da prevedere
        self.units = units  # Unità LSTM
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def call(self): 
        """Esegue tutti i metodi della classe in sequenza.""" 
        logging.info("Esecuzione completa del processo di previsione serie temporale") 
        self.prepare_data() 
        self.train_model() 
        predictions = self.predict() 
        mae = self.evaluate_model() 
        logging.info("Processo completato") 
        return {
            "predictions": predictions, 
            "mae": mae,
        }

    def normalize(self):
        """Normalizza i dati del dataframe sulla feature selezionata."""
        self.df['ADJ_NORMALIZE'] = self.scaler.fit_transform(self.df[[self.feature]])

    def prepare_data(self):
        """Prepara i dati per l'addestramento del modello."""
        logging.info("info - Preparazione dati per previsione serie temporale")
        
        # Normalizzazione dei dati
        self.normalize()
        data = self.df['ADJ_NORMALIZE'].values

        X, y = [], []
        for i in range(len(data) - self.lookback - self.out_steps):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.out_steps])

        self.X = np.array(X)
        self.y = np.array(y)

        # Aggiungere una dimensione in più per le feature
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))
        self.y = self.y.reshape((self.y.shape[0], self.y.shape[1], 1))

        # Divisione in training e test set
        split = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:split], self.X[split:]
        self.y_train, self.y_test = self.y[:split], self.y[split:]
        
        logging.info(f'len di X_train: {len(self.X_train)}, len di X_test: {len(self.X_test)}, len di y_test: {len(self.y_test)}, len di y_train: {len(self.y_train)}')
        logging.info("Shape di X_train: {}".format(self.X_train.shape))
        logging.info("Shape di X_test: {}".format(self.X_test.shape))
        logging.info("Preparazione dati completata")

    def train_model(self):
        """Addestra il modello autoregressivo LSTM."""
        logging.info("Addestramento modello autoregressivo LSTM")
        
        # Definizione del modello autoregressivo con LSTM
        self.model = FeedBack(self.units, self.out_steps, self.X_train.shape[2])  # Utilizzo del modello FeedBack
        
        # Compilazione del modello
        self.model.compile(optimizer='adam', loss='mse')
        
        # Addestramento
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0)

    def predict(self): 
        """Genera previsioni con il modello addestrato.""" 
        logging.info("Previsione serie temporale autoregressiva") 
        # Usa l'ultimo segmento di dati storici come input per la previsione 
        last_data = self.X_test[-1].reshape(1, self.lookback, 1) 
        predictions = self.model.predict(last_data) 
        # Restituisce le previsioni denormalizzate 
        predictions_inverse = self.scaler.inverse_transform(predictions.reshape(-1, 1)) 
        return predictions_inverse.flatten() 
    
    def evaluate_model(self): 
        """Calcola e restituisce il MAE generico per il test set.""" 
        predictions = self.predict() 
        y_test_denorm = self.scaler.inverse_transform(self.y_test.reshape(-1, 1)) 
        mae = mean_absolute_error(y_test_denorm[-self.out_steps:], predictions) 
        return mae
    

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)
        
        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)
        
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])  # Organizza le previsioni nella forma desiderata
        return predictions
