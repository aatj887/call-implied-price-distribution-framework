import pandas as pd # Used to interact with the data as dataframes
from datetime import date, timedelta

# Public FRED CSV download: no API key required
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"

def load_sofr_history(start:str, end:str) -> pd.Series:
    """
    Obtains the SOFR data from FRED on the defined start and end dates, as decimals.
    start and end must be in format YYYY-MM-DD.
    """
    df = pd.read_csv(
        FRED_CSV_URL.format(series="SOFR", start=start, end=end),
        index_col=0, parse_dates=True, na_values=".",
    )
    s = df['SOFR'].dropna()/100
    s.index.name = 'date'
    return s

def risk_free_asof(day) -> float:
    """
    Returns the latest SOFR published on or before the given date as a decimal. Used for backtests.
    day (str | date) : Date in format YYYY-MM-DD.
    """
    day = pd.Timestamp(day).date()
    sofr_history = load_sofr_history((day-timedelta(days=14)).strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))
    if sofr_history.empty:
        raise ValueError(f"No SOFR observations found in the two weeks before {day}.")
    return float(sofr_history.iloc[-1])

def risk_free_value() -> float:
    """
    Returns the most recent risk-free rate (SOFR) as a decimal.
    """
    return risk_free_asof(date.today())
