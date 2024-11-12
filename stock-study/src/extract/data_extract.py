from datetime import datetime, timedelta

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib as plt

import sys
sys.path.append(r'C:\Users\loren\OneDrive\Desktop\Workbench\stock-study')

from src.utils.download_ticker import download
from extractABC import dataExtractABC

class dataExtract(dataExtractABC):
    def __init__(self, ticker: str) -> None:

        self.ticker = ticker

        self.end_date = datetime.today().strftime('%Y-%m-%d') 
        self.start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        self.df = download(self.ticker, self.start_date, self.end_date)

    def process(self) -> pd.DataFrame:
        self.clean_data()
        self.normalize()
        self.add_moving_average(span=12)
        self.add_moving_average(span=26)
        self.add_rsi()
        return self.df
    
    def normalize(self) -> pd.DataFrame: 
        """
        Normalizes the 'Adj Close' column of the DataFrame to a range between 0 and 1, 
        using Min-Max scaling, and adds the result as a new column 'ADJ_NORMALIZE'.

        Returns:
            pd.DataFrame: The DataFrame with an added column 'ADJ_NORMALIZE' containing 
            the normalized 'Adj Close' values.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))

        self.df['ADJ_NORMALIZE'] = self.df['Adj Close'].values.reshape(-1, 1)
        self.df['ADJ_NORMALIZE'] = scaler.fit_transform(self.df['Adj Close'])
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        self.df.dropna(inplace=True)
        return self.df
    
    def add_moving_average(self, span: int) -> pd.DataFrame:
        self.df[f'EWM_{span}'] = self.df['Adj Close'].ewm(span=span, adjust=False).mean()
        return self.df
    
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        delta = self.df['Adj Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return self.df
    
    def plot_data(self, columns=['Adj Close', 'ADJ_NORMALIZE']) -> None:
        self.df[columns].plot(figsize=(14, 7))
        plt.title(f'{self.ticker} Stock Data')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(columns)
        plt.show()

    def save_to_csv(self) -> None:
        self.df.to_csv(fr'C:/Users/loren/OneDrive/Desktop/Workbench/stock-study/data/{self.ticker}.csv', index=True)


if(__name__ == '__main__'):
    d = dataExtract('MSFT')
    d.process()
    d.save_to_csv()
    print(d.df)