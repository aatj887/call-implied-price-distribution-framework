from openbb import obb
import pandas as pd

# Get option chains for a specific stock for a specific date

def get_option_chains(symbol: str, expiration_date: str) -> pd.DataFrame:
    """
    Get the latest option chains for a specific stock.

    Parameters:
    symbol (str): The stock symbol to get option chains for.
    expiration_date (str): The expiration date of the options in the format 'YYYY-MM-DD'.

    Returns:
    pd.DataFrame: A DataFrame containing the option chains for the specified stock and date.
    """
    # Get the option chains for the specified stock and date
    option_chains = obb.derivatives.options.chains(symbol=symbol).to_dataframe()

    # Convert the result to a DataFrame
    

    return option_chains