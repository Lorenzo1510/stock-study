from datetime import datetime
import sys

sys.path.append(r'C:\Users\loren\OneDrive\Desktop\Workbench\stock-project')

from src.extract.extract import DataExtract
from src.model.lstm_pred import MultiTimeSeriesPredictor
from src.etl.etlABC import ETL
from src.load.pred_loader import PredLoader


class PredETL(ETL):
    def __init__(self):

        # self.ticker_list = ['IBM', 'MSFT', 'BPE.MI']
        self.ticker_list = ['IBM', 'BPE.MI']
        self.year = datetime.today().strftime('%Y')
        self.today = datetime.today().strftime('%d%m')

        self.output_path = fr"C:\Users\loren\OneDrive\Desktop\Workbench\stock-project\data\pred\{self.year}\{self.today}.json"

        self.extract_list = [DataExtract(ticker) for ticker in self.ticker_list]

        # print(self.extract_list)
        # exit()

        self.transform = MultiTimeSeriesPredictor()

        # print(transform)

        self.load_list = [PredLoader(data=self.transform, output_path=self.output_path)]
        super().__init__(self.extract_list, self.transform, self.load_list)