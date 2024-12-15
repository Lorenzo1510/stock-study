from datetime import datetime
import sys

sys.path.append(r'C:\Users\loren\OneDrive\Desktop\Workbench\stock-project')

from src.extract.extract import DataExtract
from src.model.lstm_pred import TimeSeriesPredictor
from src.etl.etlABC import ETLABC
from src.load.pred_loader import PredLoader


class predETL(ETLABC):

    # ticker_list = ['IBM', 'MSFT', 'BPE.MI']
    ticker_list = ['IBM', 'BPE.MI']
    year = datetime.today().strftime('%Y')
    today = datetime.today().strftime('%d%m')

    output_path = fr"C:\Users\loren\OneDrive\Desktop\Workbench\stock-project\data\pred\{year}\{today}.json"

    extract_list = [DataExtract(ticker).read() for ticker in ticker_list]

    # print(extract_list)

    transform = [TimeSeriesPredictor(df_ticker).call() for df_ticker in extract_list]

    # print(transform)

    load_list = [PredLoader(data=transform, output_path=output_path).write()]