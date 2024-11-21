import yfinance as yf
import pandas as pd

def download(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scarica i dati storici di un titolo azionario utilizzando il modulo `yfinance`.

    Args:
        ticker (str): Il simbolo del titolo azionario (ad esempio "AAPL" per Apple).
        start_date (str): La data di inizio per il periodo da scaricare, nel formato 'YYYY-MM-DD'.
        end_date (str): La data di fine per il periodo da scaricare, nel formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Un DataFrame contenente i dati storici del titolo richiesto, 
                      con colonne tipiche come 'Open', 'High', 'Low', 'Close', 'Adj Close' e 'Volume'.
    """
    df = yf.download(ticker, start_date, end_date)
    return df
