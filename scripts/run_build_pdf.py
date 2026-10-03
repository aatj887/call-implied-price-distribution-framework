from src.modelling.minimization import objective_from_chain
from src.data_sources.risk_free import risk_free_value
from src.data_sources.cboe import get_option_chains
from scipy.optimize import minimize
import numpy as np
import pandas as pd

def fit_distribution(chain: pd.DataFrame, r: float, days_to_expiry: int) -> np.ndarray:
    """
    Returns the optimal distribution parameters [a1, b1, a2, b2, q] for one expiry of an option chain.
    Does no I/O, so it works the same on a live chain and on a stored snapshot.

    chain (pd.DataFrame) : Option chain with strike, bid, ask, option_type, dte and underlying_price columns.
    r (float) : Risk-free rate as of the chain's quote date.
    days_to_expiry (int) : Expiry to fit, in days to expiration.
    """

    x0 = [0.9*np.log(chain['underlying_price'].mean()), 0.2,
          1.1*np.log(chain['underlying_price'].mean()), 0.5,
          0.5]

    bounds = [
        (None, None), (1e-6, None),
        (None, None), (1e-6, None),
        (0, 1)
    ]

    # Remove stale data from the chain
    chain = chain[(chain['bid'] > 0) & (chain['ask'] > 0)]
    chain = chain[chain['ask'] > chain['bid']]
    chain = chain[abs(chain['strike'] - chain['underlying_price']) < 0.5*chain['underlying_price']]
    chain = chain[chain['dte'] == days_to_expiry]

    if chain.empty:
        raise ValueError(f"No usable quotes left for the {days_to_expiry} DTE expiry after filtering.")

    res = minimize(objective_from_chain, x0, args=(chain, r), bounds=bounds)

    return res.x

def return_optimal_distribution_parameters(ticker:str, days_to_expiry:int):
    """
    Returns the optimal distribution parameters for a given ticker and expiry date using live data.
    """
    return fit_distribution(get_option_chains(ticker), risk_free_value(), days_to_expiry)
