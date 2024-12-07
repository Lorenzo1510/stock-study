import logging
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import LSTM, Dense # type: ignore

from src.model.modelABC import ModelABC


class TimeSeriesPredictor(ModelABC):
    def __init__(self, 
                 df: pd.DataFrame, 
                 feature: str = 'Adj Close', 
                 lookback: int = 60, 
                 out_steps: int = 5, 
                 units: int = 50):
        self.df = df
        self.feature = feature
        self.lookback = lookback
        self.out_steps = out_steps
        self.units = units
        self.model: Any = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def call(self) -> Dict[str, Any]: 
        """Esegue tutti i metodi della classe in sequenza.""" 
        logging.info("info - start forecast") 
        self.prepare_data() 
        self.train_model() 
        predictions = self.predict() 
        mae = self.evaluate_model() 
        logging.info("DONE! - forecast successfully") 
        return {
            "predictions": predictions, 
            "mae": mae,
        }

    def normalize(self) -> None:
        """Normalizza i dati del dataframe sulla feature selezionata."""
        self.df['ADJ_NORMALIZE'] = self.scaler.fit_transform(self.df[[self.feature]])

    def prepare_data(self) -> None:
        """Prepara i dati per l'addestramento del modello."""
        logging.info("Preparazione dati per previsione serie temporale")
        
        self.normalize()
        data = self.df['ADJ_NORMALIZE'].values

        X, y = self.create_sequences(data)
        self.X, self.y = np.array(X), np.array(y)

        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))
        self.y = self.y.reshape((self.y.shape[0], self.y.shape[1], 1))

        self.split_data()
        
        logging.info("Preparazione dati completata")

    def create_sequences(self, data: np.ndarray) -> Tuple[list, list]:
        """Crea sequenze di dati per l'addestramento."""
        X, y = [], []
        for i in range(len(data) - self.lookback - self.out_steps):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.out_steps])
        return X, y

    def split_data(self) -> None:
        """Divide i dati in set di addestramento e test."""
        split = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:split], self.X[split:]
        self.y_train, self.y_test = self.y[:split], self.y[split:]

    def train_model(self) -> None:
        """Addestra il modello autoregressivo LSTM."""
        logging.info("Addestramento modello autoregressivo LSTM")
        
        self.model = FeedBack(self.units, self.out_steps, self.X_train.shape[2])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0)

    def predict(self) -> np.ndarray: 
        """Genera previsioni con il modello addestrato.""" 
        logging.info("Previsione serie temporale autoregressiva") 
        last_data = self.X_test[-1].reshape(1, self.lookback, 1) 
        predictions = self.model.predict(last_data) 
        predictions_inverse = self.scaler.inverse_transform(predictions.reshape(-1, 1)) 
        return predictions_inverse.flatten() 
    
    def evaluate_model(self) -> float: 
        """Calcola e restituisce il MAE generico per il test set.""" 
        predictions = self.predict() 
        y_test_denorm = self.scaler.inverse_transform(self.y_test.reshape(-1, 1)) 
        mae = mean_absolute_error(y_test_denorm[-self.out_steps:], predictions) 
        return mae
    

class FeedBack(tf.keras.Model):
    def __init__(self, units: int, out_steps: int, num_features: int):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, list]:
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)
        
        for _ in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)
        
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
