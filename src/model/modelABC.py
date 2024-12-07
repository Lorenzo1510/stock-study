from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import logging


class ModelABC(ABC):
    def __init__(self, 
                 df: pd.DataFrame, 
                 feature: str = 'Adj Close', 
                 lookback: int = 60, 
                 out_steps: int = 5, 
                 units: int = 50):
        """
        Classe astratta per un predittore di serie temporali.
        
        :param df: DataFrame contenente i dati.
        :param feature: La colonna della feature da prevedere.
        :param lookback: Numero di passi temporali da utilizzare come input.
        :param out_steps: Numero di passi temporali da prevedere.
        :param units: Numero di unità del modello LSTM.
        """
        self.df = df
        self.feature = feature
        self.lookback = lookback
        self.out_steps = out_steps
        self.units = units
        self.model: Any = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    @abstractmethod
    def call(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_sequences(self, data: np.ndarray) -> Tuple[list, list]:
        """Crea sequenze di dati per l'addestramento."""
        pass

    @abstractmethod
    def split_data(self) -> None:
        """Divide i dati in set di addestramento e test."""
        pass

    @abstractmethod
    def train_model(self) -> None:
        """Addestra il modello autoregressivo."""
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Genera previsioni con il modello addestrato."""
        pass

    @abstractmethod
    def evaluate_model(self) -> float:
        """Valuta il modello calcolando il MAE."""
        pass
