import pandas as pd
import yfinance as yf

def get_price_history(ticker:str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get daily closes, dividends and splits for a ticker from Yahoo Finance.

    'close' is split-adjusted but not dividend-adjusted (price returns, as options settle on);
    'adj_close' is adjusted for both (total returns). Yahoo restates both after every split, so
    compare prices within one download (e.g. a ratio of two closes) instead of archiving them.

    Parameters:
    - ticker (str): The ticker symbol for the asset.
    - start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the data in 'YYYY-MM-DD' format (exclusive).

    Returns:
    - pd.DataFrame: Columns close, adj_close, dividends, splits indexed by date.
    """
    data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d", auto_adjust=False, actions=True)
    if data.empty:
        raise ValueError(f"Yahoo Finance returned no prices for '{ticker}' between {start_date} and {end_date}.")

    data.index = data.index.tz_localize(None).normalize()
    data.index.name = 'date'
    return data.rename(columns={'Close': 'close', 'Adj Close': 'adj_close', 'Dividends': 'dividends', 'Stock Splits': 'splits'})[
        ['close', 'adj_close', 'dividends', 'splits']
    ]


def get_close_prices(ticker:str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get the closing prices for a given ticker from Yahoo Finance.

    Parameters:
    - ticker (str): The ticker symbol for the asset.
    - start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing the closing prices with dates as the index.
    """
    return get_price_history(ticker, start_date, end_date)[['close']].rename(columns={'close': ticker})


def get_current_price(ticker:str) -> float:
    """
    Get the latest closing price for a given ticker from Yahoo Finance.

    Parameters:
    - ticker (str): The ticker symbol for the asset.

    Returns:
    - float: The current price of the asset.
    """
    data = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
    return float(data['Close'].iloc[-1])
