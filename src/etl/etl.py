from datetime import datetime
import sys

sys.path.append(r'C:\Users\loren\OneDrive\Desktop\Workbench\stock-project')

from src.extract.extract import DataExtract
from src.model.lstm_pred import TimeSeriesPredictor
from src.etl.etlABC import ETLABC
from src.load.pred_loader import PredLoader


class predETL(ETLABC):
    def __init__(self):

        # self.ticker_list = ['IBM', 'MSFT', 'BPE.MI']
        self.ticker_list = ['IBM', 'BPE.MI']
        self.year = datetime.today().strftime('%Y')
        self.today = datetime.today().strftime('%d%m')

        self.output_path = fr"C:\Users\loren\OneDrive\Desktop\Workbench\stock-project\data\pred\{self.year}\{self.today}.json"

        self.extract_list = [DataExtract(ticker).read() for ticker in self.ticker_list]

        # print(extract_list)

        self.transform = [TimeSeriesPredictor(df_ticker).call() for df_ticker in self.extract_list]

        # print(transform)

        self.load_list = [PredLoader(data=self.transform, output_path=self.output_path).write()]
        super().__init__()