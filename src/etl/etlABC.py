from abc import ABC, abstractmethod
from typing import List


class ETL(ABC):
    def __init__(self, extract_list, transform, load_list):
        """
        Classe astratta ETL per gestire il processo di Extract, Transform, Load.
        """
        self.extract_list = extract_list
        self.transform = transform
        self.load_list = load_list 

    def process(self):
        """
        Metodo che esegue il flusso ETL completo: read -> call -> write.
        """

        extracted_data = []
        for extractor in self.extract_list:
            df = extractor.read()
            extracted_data.append(df)

        transformed_data = None
        if self.transform:
            self.transform.call(extracted_data)  # Passa i dati direttamente

        # TODO: rimuovere lo zero in futuro
        self.load_list[0].write()