import logging
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore


class EnhancedTimeSeriesPredictor:
    def __init__(self, 
                 df: pd.DataFrame, 
                 feature_columns: List[str], 
                 lookback: int = 60, 
                 out_steps: int = 5, 
                 units: int = 50):
        """
        Inizializza la classe per la previsione delle serie temporali.
        :param df: DataFrame con i dati di input.
        :param feature_columns: Lista delle colonne da utilizzare come feature.
        :param lookback: Numero di osservazioni precedenti da considerare.
        :param out_steps: Numero di passi di previsione futuri.
        :param units: Numero di unità LSTM.
        """
        self.df = df
        self.feature_columns = feature_columns
        self.lookback = lookback
        self.out_steps = out_steps
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.param_grid = {
            'lookback': [30, 60, 90],
            'out_steps': [5, 10],
            'units': [50, 100],
            'epochs': [20, 50],
            'batch_size': [16, 32]
        }

    def normalize(self) -> None:
        """Normalizza tutte le colonne selezionate nel DataFrame."""
        self.df[self.feature_columns] = self.scaler.fit_transform(self.df[self.feature_columns])

    def add_lag_features(self, lags: int = 5) -> None:
        """
        Aggiunge lag features per ogni colonna selezionata.
        :param lags: Numero di ritardi temporali da aggiungere.
        """
        for col in self.feature_columns:
            for lag in range(1, lags + 1):
                self.df[f"{col}_lag{lag}"] = self.df[col].shift(lag)
        self.df.dropna(inplace=True)  # Rimuove righe con valori NaN dovuti ai lag

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea sequenze di dati di input e target.
        :param data: Array numpy con i dati normalizzati.
        :return: Tuple di (X, y).
        """
        X, y = [], []
        for i in range(len(data) - self.lookback - self.out_steps):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.out_steps, 0])  # Target: prima colonna (e.g., Adj Close)
        return np.array(X), np.array(y)

    def split_data(self) -> None:
        """
        Divide i dati in set di addestramento e test.
        """
        data = self.df[self.feature_columns].values
        X, y = self.create_sequences(data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self) -> None:
        """
        Costruisce il modello LSTM multivariato.
        """
        num_features = self.X_train.shape[2]
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.lookback, num_features)),
            Dropout(0.2),
            LSTM(self.units, dropout=0.2),
            Dense(self.out_steps)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Addestra il modello LSTM.
        :param epochs: Numero di epoche per l'addestramento.
        :param batch_size: Dimensione del batch.
        """
        logging.info("Inizio addestramento del modello")
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        logging.info("Addestramento completato")

    def predict(self) -> np.ndarray:
        """
        Genera previsioni sull'ultimo batch del test set.
        :return: Previsioni inverse trasformate per 'Adj Close'.
        """
        last_sequence = self.X_test[-1].reshape(1, self.lookback, -1)
        predictions = self.model.predict(last_sequence)
        
        # Trasformazione inversa solo per la colonna 'Adj Close'
        adj_close_scaler = MinMaxScaler()
        adj_close_scaler.min_, adj_close_scaler.scale_ = self.scaler.min_[0], self.scaler.scale_[0]
        predictions_inverse = adj_close_scaler.inverse_transform(predictions)
        
        return predictions_inverse.flatten()  # Appiattisce l'array

    def evaluate_model(self) -> float:
        """
        Valuta il modello calcolando il Mean Absolute Error (MAE).
        :return: Valore MAE.
        """
        predictions = self.predict()
        
        # Trasformazione inversa solo per la colonna 'Adj Close'
        adj_close_scaler = MinMaxScaler()
        adj_close_scaler.min_, adj_close_scaler.scale_ = self.scaler.min_[0], self.scaler.scale_[0]
        y_test_denorm = adj_close_scaler.inverse_transform(self.y_test)

        # Allineamento delle dimensioni
        y_test_denorm = y_test_denorm[-self.out_steps:].flatten()  # Appiattisce
        predictions = predictions[:self.out_steps].flatten()  # Assicura compatibilità
        
        mae = mean_absolute_error(y_test_denorm, predictions)
        logging.info(f"MAE calcolato: {mae}")
        return mae

    def call(self) -> Dict[str, Any]:
        """
        Esegue l'intera pipeline di previsione.
        :param param_grid: Griglia di parametri per la ricerca degli iperparametri.
        :return: Dizionario con previsioni e MAE.
        """
        logging.info("Inizio pipeline di previsione")

        # Normalizzazione e creazione lag features
        self.normalize()
        self.add_lag_features()

        # Preparazione dei dati
        self.split_data()

        # Costruzione del modello
        self.build_model()

        # Ricerca degli iperparametri (opzionale)
        if self.param_grid:
            logging.info("Ricerca degli iperparametri in corso")
            best_params = self.grid_search(self.param_grid)
            logging.info(f"Migliori parametri trovati: {best_params}")
        else:
            best_params = {'epochs': 50, 'batch_size': 32}

        # Addestramento del modello
        self.train_model(epochs=best_params['epochs'], batch_size=best_params['batch_size'])

        # Generazione di previsioni ed evaluation
        predictions = self.predict()
        mae = self.evaluate_model()
        return {"predictions": predictions, "mae": mae}

    def grid_search(self, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Ricerca esaustiva dei migliori iperparametri.
        :param param_grid: Dizionario con la griglia di parametri.
        :return: Migliori parametri trovati.
        """
        best_mae = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            logging.info(f"Testando parametri: {params}")
            self.build_model()
            self.train_model(epochs=params['epochs'], batch_size=params['batch_size'])
            mae = self.evaluate_model()
            if mae < best_mae:
                best_mae = mae
                best_params = params

        return best_params
