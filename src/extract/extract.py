from datetime import datetime, timedelta
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import sys
sys.path.append(r'C:\Users\loren\OneDrive\Desktop\Workbench\stock-project')

from src.utils.download_ticker import download
from src.extract.extractABC import DataExtractABC

class DataExtract(DataExtractABC):
    def __init__(self, ticker: str) -> None:

        self.ticker = ticker

        self.end_date = datetime.today().strftime('%Y-%m-%d') 
        self.start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        self.df = download(self.ticker, self.start_date, self.end_date)

    def read(self) -> pd.DataFrame:
        logging.info(f"info - reading {self.ticker}")
        self.clean_data()
        self.normalize()
        self.add_moving_average(span=12)
        self.add_moving_average(span=26)
        self.add_rsi()
        self.add_macd()
        logging.info(f"DONE! - reading {self.ticker}")
        return self.df
    
    def validate_data(self) -> None:
        if self.df.empty:
            raise ValueError(f"Nessun dato disponibile per il ticker {self.ticker}.")
        if self.df.isnull().sum().sum() > 0:
            print("Avviso: Dati mancanti rilevati. Verranno rimossi.")
            self.df.dropna(inplace=True)
    
    def normalize(self) -> pd.DataFrame:
        if 'Adj Close' not in self.df.columns:
            raise ValueError("La colonna 'Adj Close' non è presente nel DataFrame.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.df['ADJ_NORMALIZE'] = scaler.fit_transform(self.df[['Adj Close']])
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        self.df.dropna(inplace=True)
        return self.df
    
    def add_moving_average(self, span: int) -> pd.DataFrame:
        self.df[f'EWM_{span}'] = self.df['Adj Close'].ewm(span=span, adjust=False).mean()
        return self.df
    
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        delta = self.df['Adj Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return self.df

    def add_macd(self, short_span: int = 12, long_span: int = 26, signal_span: int = 9) -> pd.DataFrame:
        short_ema = self.df['Adj Close'].ewm(span=short_span, adjust=False).mean()
        long_ema = self.df['Adj Close'].ewm(span=long_span, adjust=False).mean()
        self.df['MACD'] = short_ema - long_ema
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=signal_span, adjust=False).mean()
        self.df['MACD_hist'] = self.df['MACD'] - self.df['MACD_signal']
        return self.df

    def plot_data(self, columns=['Adj Close', 'ADJ_NORMALIZE']) -> None:
        # Plotting the selected columns
        self.df[columns].plot(figsize=(14, 7))
        
        # Aggiungiamo titolo, etichette e legenda
        plt.title(f'{self.ticker} Stock Data')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(columns)
        
        # Visualizzare il grafico
        plt.show()

    def save_to_csv(self) -> None:
        self.df.to_csv(fr'C:/Users/loren/OneDrive/Desktop/Workbench/stock-study/data/{self.ticker}.csv', index=True)


if(__name__ == '__main__'):
    d = DataExtract('MSFT')
    d.read()
    # d.save_to_csv()
    print(d.df)
