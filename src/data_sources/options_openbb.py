from openbb_yfinance.models.options_chains import YFinanceOptionsChainsFetcher
import pandas as pd

# Get option chains for a specific stock for a specific date

def get_option_chains(symbol: str) -> pd.DataFrame:
    """
    Get the latest option chains for a specific stock.

    Parameters:
    symbol (str): The stock symbol to get option chains for.

    Returns:
    pd.DataFrame: A DataFrame containing the option chains for the specified stock.
    """
    # Initialize the specific fetcher you need
    fetcher = YFinanceOptionsChainsFetcher()
    
    # Fetch the data using the provider directly
    # 'test_callable' is the standard method for direct execution in v4
    result = fetcher.test_callable({"symbol": symbol})
    
    # This returns the exact same DataFrame you were expecting
    return result.to_df()