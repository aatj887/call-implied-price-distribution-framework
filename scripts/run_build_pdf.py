from src.modelling.minimization import objective_from_chain
from src.data_sources.risk_free import risk_free_value
from src.data_sources.options_openbb import get_option_chains
from scipy.optimize import minimize
import numpy as np

def return_optimal_distribution_parameters(ticker:str, days_to_expiry:str):
    """
    Returns the optimal distribution parameters for a given ticker and expiry date.
    """

    chain = get_option_chains(ticker)
    r = risk_free_value()

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

    res = minimize(objective_from_chain, x0, args=(chain, r), bounds=bounds)

    return res.x