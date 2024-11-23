import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore

class TimeSeriesPredictor:
    def __init__(self, df: pd.DataFrame, feature: str = 'ADJ_NORMALIZE', lookback: int = 60):
        self.df = df
        self.feature = feature
        self.lookback = lookback
        self.model = None  # Placeholder per il modello ML/Deep Learning

    def prepare_data(self):
        # Creazione di sequenze basate sulla finestra temporale (lookback)
        logging.info("Preparazione dati per previsione serie temporale")
        data = self.df[[self.feature]].values
        X, y = [], []

        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])

        self.X = np.array(X)
        self.y = np.array(y)

        # Divisione in training e test set
        split = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:split], self.X[split:]
        self.y_train, self.y_test = self.y[:split], self.y[split:]
        logging.info("Preparazione dati completata")

    def train_model(self):
        logging.info("Addestramento modello LSTM")
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.X_train, self.y_train, epochs=20, batch_size=32, verbose=1)

    def predict(self):
        logging.info("Previsione serie temporale")
        predictions = self.model.predict(self.X_test)
        
        # Applicare la denormalizzazione alle previsioni (se i dati sono stati normalizzati prima)
        if self.feature == 'ADJ_NORMALIZE':
            # Utilizziamo lo scaler per riportare i valori alla scala originale
            # Calcoliamo lo scaler solo se non è stato ancora fatto
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(self.df[['Adj Close']].values)  # Si suppone che 'Adj Close' contenga i dati originali non scalati
            predictions_inverse = self.scaler.inverse_transform(predictions)

            return predictions_inverse
        else:
            return predictions
