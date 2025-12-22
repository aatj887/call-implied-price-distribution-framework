from openbb import obb # In charge of the collection of SOFR data we will use to calculate risk free.
import pandas as pd # Used to interact with the data as dataframes 
import math # Supports calculations of specific mathematical functions in the paper
import numpy as np

def load_sofr_history(start:str, end:str):
    """
    Obtains the SOFR data from FRED on the defined start and end dates.
    start and end must be in format YYYY-MM-DD.
    """
    df = obb.economy.fred_series("SOFR", start_date=start, end_date=end).to_dataframe()
    s = df/100
    s = s['SOFR']
    return s

def historical_rigorous_rf(
    start:str,
    end:str,
    horizon_days:int,
    day_count=360,
):
    """
    Generates the exact r value for the call pricing formula in page 12 of the PDF.
    It grabs the daily annualised SOFR rates from FRED (which are simple rates, not compounding) and calculates the daily return based on the annualised number.
    Interest Accrued in one day = SOFR annualised rate * 1/ 360
    start: Starting date of the needed dates. String in the form 'YYYY-MM-DD'
    end: End date of the needed dates. String in the form 'YYYY-MM-DD'
    horizon_days: In the context of the paper, this would be the time to maturity for the option in days (T-t). Integer.
    day_count: 360 or 365 depending on what type of denominator we want to use for the simple interest calculation. Set at 360 as it is more commonly used.
    """
    sofr = load_sofr_history(start, end)

    # Ensure daily grid and forward-fill
    idx = pd.date_range(start, end, freq="D")
    sofr = sofr.reindex(idx).ffill()

    dt = 1.0 / day_count
    one_day_growth = 1.0 + sofr * dt

    # Growth over next horizon_days
    growth_N = one_day_growth.shift(-1).rolling(horizon_days).apply(np.prod, raw=True)

    # Discount factor (e^{-r(T-t)})
    df_N = 1.0 / growth_N

    # tau = T-t in years
    tau = horizon_days / day_count
    # Final annualised values to use in the pricing equation
    r_ann = -np.log(df_N) / tau

    # The output is a pandas series where the index is a date,
    # and the value representing the date is the literal value of r in the pricing formula on page 12 of the PDF.
    return r_ann.dropna().rename(f"rf_{horizon_days}d_continuous_SOFR")




