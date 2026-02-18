from openbb import obb # In charge of the collection of SOFR data we will use to calculate risk free.
import pandas as pd # Used to interact with the data as dataframes 
import numpy as np
from datetime import date, timedelta

def load_sofr_history(start:str, end:str):
    """
    Obtains the SOFR data from FRED on the defined start and end dates.
    start and end must be in format YYYY-MM-DD.
    """
    df = obb.economy.fred_series("SOFR", start_date=start, end_date=end).to_dataframe()
    s = df/100
    s = s['SOFR']
    return s

def risk_free_value():
    """
    Returns the most recent risk-free rate (SOFR) as a decimal.
    """
    sofr_history = load_sofr_history((date.today()-timedelta(days=7)).strftime("%Y-%m-%d"), date.today().strftime("%Y-%m-%d"))
    r = sofr_history.tail(1).iloc[0]
    return r






