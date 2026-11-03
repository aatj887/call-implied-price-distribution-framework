from src.modelling.minimization import objective_from_chain
from src.data_sources.risk_free import risk_free_value
from src.data_sources.cboe import get_option_chains
from scipy.optimize import minimize, OptimizeResult
import numpy as np
import pandas as pd

BOUNDS = [
    (None, None), (1e-6, None),
    (None, None), (1e-6, None),
    (0, 1)
]

def clean_chain(chain: pd.DataFrame, days_to_expiry: int) -> pd.DataFrame:
    """
    Keeps the quotes of one expiry that are usable for the fit: two-sided, not crossed and with
    strikes within 50% of the underlying price.
    """
    chain = chain[(chain['bid'] > 0) & (chain['ask'] > 0)]
    chain = chain[chain['ask'] > chain['bid']]
    chain = chain[abs(chain['strike'] - chain['underlying_price']) < 0.5*chain['underlying_price']]
    return chain[chain['dte'] == days_to_expiry]

def starting_points(chain: pd.DataFrame, r: float) -> list[list[float]]:
    """
    Initial guesses for [a1, b1, a2, b2, q]. The optimiser often stops in a different local minimum
    depending on where it starts, so the fit tries each one and keeps the best.
    """
    S0 = chain['underlying_price'].iloc[0]
    tau = chain['dte'].iloc[0] / 365

    # Original guess: one component below and one above the spot price
    log_s0 = np.log(S0)
    original = [0.9*log_s0, 0.2, 1.1*log_s0, 0.5, 0.5]

    # Both components around the forward, with a width taken from the at-the-money straddle
    # (straddle ~ 0.8 * S0 * sigma * sqrt(tau))
    distance = (chain['strike'] - S0).abs()
    atm = chain[distance == distance.min()]
    straddle = (0.5*(atm['bid'] + atm['ask'])).groupby(atm['option_type']).mean().sum()
    s = max(straddle / (0.8*S0), 1e-3)
    m = np.log(S0) + r*tau - 0.5*s*s
    around_forward = [m - 0.5*s, 1.5*s, m + 0.2*s, 0.8*s, 0.3]

    return [original, around_forward]

def fit_distribution_result(chain: pd.DataFrame, r: float, days_to_expiry: int) -> tuple[OptimizeResult, pd.DataFrame]:
    """
    Fits one expiry and returns the best optimiser result (lowest objective over the starting points)
    together with the quotes that were used, for fit diagnostics.
    """
    chain = clean_chain(chain, days_to_expiry)
    if chain.empty:
        raise ValueError(f"No usable quotes left for the {days_to_expiry} DTE expiry after filtering.")

    best = None
    with np.errstate(over='ignore', invalid='ignore'):  # Extreme trial points overflow; they just score badly
        for x0 in starting_points(chain, r):
            res = minimize(objective_from_chain, x0, args=(chain, r), bounds=BOUNDS)
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
    if best is None:
        raise ValueError(f"The fit diverged for the {days_to_expiry} DTE expiry.")
    return best, chain

def fit_distribution(chain: pd.DataFrame, r: float, days_to_expiry: int) -> np.ndarray:
    """
    Returns the optimal distribution parameters [a1, b1, a2, b2, q] for one expiry of an option chain.
    Does no I/O, so it works the same on a live chain and on a stored snapshot.

    chain (pd.DataFrame) : Option chain with strike, bid, ask, option_type, dte and underlying_price columns.
    r (float) : Risk-free rate as of the chain's quote date.
    days_to_expiry (int) : Expiry to fit, in days to expiration.
    """
    return fit_distribution_result(chain, r, days_to_expiry)[0].x

def return_optimal_distribution_parameters(ticker:str, days_to_expiry:int):
    """
    Returns the optimal distribution parameters for a given ticker and expiry date using live data.
    """
    return fit_distribution(get_option_chains(ticker), risk_free_value(), days_to_expiry)
