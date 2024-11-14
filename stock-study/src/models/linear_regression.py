import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import sys
sys.path.append(r'C:\Users\loren\OneDrive\Desktop\Workbench\stock-study')

from src.models.model import ABCModel

class LinearRegressionModel(ABCModel):
    def __init__(self, df, seq_length=30):
        """
        Inizializza il modello di regressione lineare.
        
        Parameters:
        df (pd.DataFrame): DataFrame contenente i dati delle azioni.
        seq_length (int): Numero di giorni precedenti da usare per fare previsioni.
        """
        super().__init__()
        self.df = df
        self.seq_length = seq_length
        self.model = LinearRegression()
        self.params = None
        
        # Prepara i dati
        self.X, self.y = self.prepare_data(df['Adj Close'].values)
        
    def prepare_data(self, adj_close):
        """
        Prepara i dati per l'addestramento del modello creando sequenze temporali.
        
        Parameters:
        adj_close (np.ndarray): Array dei prezzi 'Adj Close'.
        
        Returns:
        X (np.ndarray): Dati di input come sequenze temporali.
        y (np.ndarray): Etichette (prezzi futuri da prevedere).
        """
        sequences = []
        labels = []
        
        for i in range(len(adj_close) - self.seq_length):
            sequences.append(adj_close[i:i + self.seq_length])
            labels.append(adj_close[i + self.seq_length])  # Valore del giorno successivo
        
        return np.array(sequences), np.array(labels)
    
    def fit(self, X=None, y=None):
        """
        Allena il modello sui dati preparati.
        
        Parameters:
        X (np.ndarray): Sequenze temporali (opzionale).
        y (np.ndarray): Etichette (prezzi futuri da prevedere) (opzionale).
        """
        if X is None or y is None:
            X, y = self.X, self.y
        
        # Addestriamo il modello con una regressione lineare
        X_reshaped = X.reshape(X.shape[0], -1)  # Reshape per adattarsi al modello
        self.model.fit(X_reshaped, y)
        
        # Salviamo i parametri
        self.params = self.model.coef_
    
    def predict(self, X):
        """
        Esegue la previsione sui dati forniti.
        
        Parameters:
        X (np.ndarray): Dati di input (sequenze temporali).
        
        Returns:
        np.ndarray: Previsioni del modello.
        """
        X_reshaped = X.reshape(X.shape[0], -1)  # Reshape per adattarsi al modello
        return self.model.predict(X_reshaped)
    
    def evaluate(self, X, y):
        """
        Calcola l'errore quadratico medio (MSE) sulle previsioni.
        
        Parameters:
        X (np.ndarray): Dati di input.
        y (np.ndarray): Etichette reali.
        
        Returns:
        float: MSE tra le previsioni e le etichette reali.
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        return mse
    
    def split(self, test_size=0.2):
        """
        Suddivide i dati in un set di addestramento e un set di test.
        
        Parameters:
        test_size (float): Proporzione dei dati da usare per il test.
        
        Returns:
        X_train, X_test, y_train, y_test: Set di addestramento e di test.
        """
        return train_test_split(self.X, self.y, test_size=test_size, shuffle=False)
