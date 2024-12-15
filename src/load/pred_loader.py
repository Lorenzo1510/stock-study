import json
import os
from typing import Dict, List

import numpy as np


class PredLoader():
    def __init__(self, data: List[Dict[str, int]], output_path: str):
        self.data = data
        self.output_path = output_path
        
    def write(self):
        # Funzione per gestire i tipi non JSON-serializzabili
        def convert(obj):
            if isinstance(obj, np.ndarray):  # Converti array numpy in lista
                return obj.tolist()
            elif isinstance(obj, np.generic):  # Converti tipi numpy scalar (es. np.float64) in tipi Python
                return obj.item()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Salva i dati in un file JSON
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, default=convert, indent=4)