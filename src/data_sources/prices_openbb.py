from openbb import obb
import pandas as pd
from datetime import date, timedelta

def get_close_prices(ticker:str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get the closing prices for a given ticker from OpenBB.

    Parameters:
    - ticker (str): The ticker symbol for the asset.
    - start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing the closing prices with dates as the index.
    """
    # Fetch the data using OpenBB and convert to a dataframe
    data = obb.equity.price.historical(
        symbol=ticker,
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    ).to_dataframe()

    # Return only the closing prices
    return data[['close']].rename(columns={'close': ticker})


def get_current_price(ticker:str) -> float:
    """
    Get the current price for a given ticker from OpenBB.

    Parameters:
    - ticker (str): The ticker symbol for the asset.

    Returns:
    - float: The current price of the asset.
    """
    # Fetch the latest price using OpenBB
    data = obb.equity.price.historical(symbol=ticker, start_date=date.today() - timedelta(days=1), end_date=date.today()).to_dataframe()
    
    # Extract and return the current price
    return data['close'].iloc[-1]

