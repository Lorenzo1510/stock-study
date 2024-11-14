from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import jax.numpy as jnp

class ABCModel(ABC):

    @abstractmethod
    def fit(self, X, y, epochs=1000, lr=0.01):
        """
        Metodo per addestrare il modello.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Metodo per fare previsioni con il modello addestrato.
        """
        pass

    def split(self, X, y, test_size=0.2, random_state=None):
        """
        Suddivide i dati in set di addestramento e test.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def evaluate(self, X, y):
        """
        Valuta il modello usando la perdita MSE (Mean Squared Error) per la regressione.
        """
        predictions = self.predict(X)
        mse = jnp.mean((y - predictions) ** 2)
        return mse
