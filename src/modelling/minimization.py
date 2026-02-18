import numpy as np
from scipy.stats import lognorm, norm

def lognormal_pdf(s:float, a:float, b:float):
    """
    Probability density function of the Lognormal Distribution. Used in the calculation of the mixture distribution in page 12.
    s (float) : Value whose probability we want to calculate
    a (float) : Mean of log-price.
    b (float) : Standard Deviation of log-price
    """
    return lognorm.pdf(s, s=b, scale=np.exp(a))

def mixture_pdf(s:float, a1:float, b1:float, a2:float, b2:float, q:float):
    """
    Calculates the weighted pdf from two parametrised log-normal distributions. Used in page 12 as the final pdf for our prices.
    s (float) : Value whose probability we want to calculate
    a1 (float) : Mean of log-price for the first distribution.
    b1 (float) : Standard Deviation of log-price for the first distribution.
    a2 (float) : Mean of log-price for the second distribution.
    b2 (float) : Standard Deviation of log-price for the second distribution.
    q (float) : Weight assigned to the first distribution (the second will have weight 1-q).
    """
    return q * lognormal_pdf(s, a1, b1) + (1 - q) * lognormal_pdf(s, a2, b2)

def implied_mean(a1, b1, a2, b2, q):
    """
    Calculates the implied mean of the calculated mixed_distribution. Will be used in the minimisation problem.

    a1 (float) : Mean of log-price for the first distribution.
    b1 (float) : Standard Deviation of log-price for the first distribution.
    a2 (float) : Mean of log-price for the second distribution.
    b2 (float) : Standard Deviation of log-price for the second distribution.
    q (float) : Weight assigned to the first distribution (the second will have weight 1-q).
    """
    return q * np.exp(a1 + 0.5*b1**2) + (1-q) * np.exp(a2 + 0.5*b2**2)

def _call_lognormal(X, r, tau, a, b):
    """
    Closed-form European call price when ln(S_T) ~ N(a, b^2).
    """

    if b < 1e-10:
        F = np.exp(a)
        return np.exp(-r*tau) * max(F - X, 0.0)

    d1 = (a - np.log(X) + b*b) / b
    d2 = d1 - b

    return np.exp(-r*tau) * (
        np.exp(a + 0.5*b*b) * norm.cdf(d1)
        - X * norm.cdf(d2)
    )

def call_price_model(X, r, tau, a1, b1, a2, b2, q):
    """
    Call price under a two-lognormal mixture.
    """

    return (
        q * _call_lognormal(X, r, tau, a1, b1)
        + (1 - q) * _call_lognormal(X, r, tau, a2, b2)
    )

def put_price_model(X, r, tau, a1, b1, a2, b2, q, S0):
    """
    Put price via put-call parity.
    """

    call = call_price_model(X, r, tau, a1, b1, a2, b2, q)
    return call - S0 + X*np.exp(-r*tau)


def objective_from_chain(theta, chain, r):
    """
    Objective function we want to minimise to find the optimal vector theta = [a1, b1, a2, b2, q]

    :param theta: Vector containing the distribution parameters and weights [a1, b1, a2, b2, q].
    :param chain: Dataframe containing the option chain data we will use for the calculations
    :param r: Adjusted risk-free rate calculated for the defined timeframe
    """

    a1, b1, a2, b2, q = theta

    # enforce ordering (identifiability constraint)
    if a1 > a2:
        return 1e10

    tau = chain['dte'].iloc[0] / 365
    S0  = chain['underlying_price'].iloc[0]
    S_forward = S0 * np.exp(r * tau)

    error = 0.0

    for _, row in chain.iterrows():
        X = row['strike']
        market_price = 0.5 * (row['bid'] + row['ask'])

        if row['option_type'].lower() == 'call':
            model_price = call_price_model(X, r, tau, a1, b1, a2, b2, q)
        else:
            model_price = put_price_model(X, r, tau, a1, b1, a2, b2, q, S0)

        error += (model_price - market_price)**2

    mean_err = (implied_mean(a1, b1, a2, b2, q) - S_forward)**2

    return error + mean_err

