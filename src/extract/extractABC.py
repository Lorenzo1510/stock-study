from abc import ABC, abstractmethod
import pandas as pd

class DataExtractABC(ABC):
    """
    Classe astratta per il caricamento e il processamento dei dati.
    Le classi derivate devono implementare il metodo process().
    """

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """
        Metodo astratto che deve essere implementato nelle classi derivate
        per elaborare i dati.
        """
        pass