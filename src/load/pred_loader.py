import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


class PredLoader:
    def __init__(self, data, output_path):
        self.data = data
        self.output_path = output_path

    def write(self):
        """
        Scrive i dati trasformati nel percorso specificato.
        """
        def convert(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(f"Type {type(obj)} not serializable")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, default=convert, indent=4)