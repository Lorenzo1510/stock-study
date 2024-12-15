import logging
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import LSTM, Dense # type: ignore

from src.model.modelABC import ModelABC
from src.model.feed_back import FeedBack


class TimeSeriesPredictor(ModelABC):
    def __init__(self, 
                 df: pd.DataFrame, 
                 feature: str = 'Adj Close', 
                 lookback: int = 60, 
                 out_steps: int = 5, 
                 units: int = 50):
        self.df = df
        self.ticker_name = df.columns.levels[1][0]
        self.feature = feature
        self.lookback = lookback
        self.out_steps = out_steps
        self.units = units
        self.model: Any = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.param_grid = {
            'lookback': [30, 60, 90],
            'units': [50, 100],
            'epochs': [20, 50],
            'batch_size': [16, 32]
        }

    def call(self) -> Dict[str, Any]: 
        """Esegue tutti i metodi della classe in sequenza.""" 
        logging.info("info - start forecast") 
        self.prepare_data() 
        self.train_model(param_grid=self.param_grid) 
        predictions = self.predict() 
        mae = self.evaluate_model() 
        logging.info("DONE! - forecast successfully") 
        return {self.ticker_name: {
            "predictions": predictions, 
            "mae": mae,
            }
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

    def train_model(self, param_grid: Dict[str, list] = None) -> None:
        """
        Addestra il modello autoregressivo LSTM.
        Esegue una GridSearch per ottimizzare i parametri se viene fornita una griglia.
        :param param_grid: Dizionario con i parametri per la GridSearch, se fornito.
        """

        # Esegui GridSearch se param_grid è fornito
        if param_grid:
            best_config = self.grid_search(param_grid)
            logging.info(f"Migliori parametri trovati: {best_config['best_params']}")

            # Aggiorna i parametri del modello con la configurazione ottimale
            self.lookback = best_config['best_params']['lookback']
            self.units = best_config['best_params']['units']
            epochs = best_config['best_params']['epochs']
            batch_size = best_config['best_params']['batch_size']
        else:
            logging.info("GridSearch non eseguito, utilizzo parametri predefiniti")
            epochs = 50
            batch_size = 32

        # Preparare i dati (con eventuali nuovi parametri lookback)
        self.prepare_data()

        # Creare e addestrare il modello con i parametri selezionati
        self.model = FeedBack(self.units, self.out_steps, self.X_train.shape[2])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def grid_search(self, param_grid: Dict[str, list]) -> Dict[str, Any]:
        """
        Esegue la ricerca esaustiva dei parametri con GridSearch.
        :param param_grid: Dizionario con i parametri da ottimizzare e i loro valori.
        :return: Dizionario con la migliore configurazione e il relativo MAE.
        """
        best_mae = float('inf')
        best_params = None

        # Creare combinazioni di parametri
        grid = ParameterGrid(param_grid)

        for params in grid:
            logging.info(f"Test combinazione parametri: {params}")

            # Aggiornare i parametri del modello
            self.lookback = params.get('lookback', self.lookback)
            self.units = params.get('units', self.units)
            epochs = params.get('epochs', 50)
            batch_size = params.get('batch_size', 32)

            # Preparare i dati e ricreare il modello
            self.prepare_data()
            self.model = FeedBack(self.units, self.out_steps, self.X_train.shape[2])
            self.model.compile(optimizer='adam', loss='mse')

            # Addestrare il modello con i parametri correnti
            self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # Valutare il modello
            mae = self.evaluate_model()
            logging.info(f"MAE con parametri {params}: {mae}")

            # Aggiornare i migliori parametri
            if mae < best_mae:
                best_mae = mae
                best_params = params

        logging.info(f"Best MAE: {best_mae} con parametri {best_params}")
        return {"best_params": best_params, "best_mae": best_mae}

    def predict(self) -> np.ndarray: 
        """Genera previsioni con il modello addestrato.""" 
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
