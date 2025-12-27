import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize
from scipy.integrate import quad

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

def call_price_model(X, r, tau, a1, b1, a2, b2, q):
    """
    Call price formula, defined as the sum of all weighted possible payoffs from holding a call today using the mixed pdf.
    
    :param X: Strike Price of the call option.
    :param r: Risk-free rate (should always be the rate generated in the data sources).
    :param tau: Time to maturity in years.
    :param a1: Mean of log-price for the first distribution.
    :param b1: Standard Deviation of log-price for the first distribution.
    :param a2: Mean of log-price for the second distribution.
    :param b2: Standard Deviation of log-price for the second distribution.
    :param q: Weight assigned to the first distribution (the second will have weight 1-q).
    """
    integrand = lambda s: (s - X) * mixture_pdf(s, a1, b1, a2, b2, q)
    value, _ = quad(integrand, X, np.inf)
    return np.exp(-r * tau) * value

def put_price_model(X, r, tau, a1, b1, a2, b2, q):
    """
    Put price formula, defined as the sum of all weighted possible payoffs from holding a call today using the mixed pdf.
    
    :param X: Strike Price of the call option.
    :param r: Risk-free rate (should always be the rate generated in the data sources).
    :param tau: Time to maturity in years.
    :param a1: Mean of log-price for the first distribution.
    :param b1: Standard Deviation of log-price for the first distribution.
    :param a2: Mean of log-price for the second distribution.
    :param b2: Standard Deviation of log-price for the second distribution.
    :param q: Weight assigned to the first distribution (the second will have weight 1-q).
    """
    integrand = lambda s: (X - s) * mixture_pdf(s, a1, b1, a2, b2, q)
    val, _ = quad(integrand, 0, X)
    return np.exp(-r * tau) * val


def objective_from_chain(theta, chain, r):
    """
    Objective function we want to minimise to find the optimal vector theta = [a1, b1, a2, b2, q]

    :param theta: Vector containing the distribution parameters and weights [a1, b1, a2, b2, q].
    :param chain: Dataframe containing the option chain data we will use for the calculations
    :param r: Adjusted risk-free rate calculated for the defined timeframe
    """

    a1, b1, a2, b2, q = theta

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
            model_price = put_price_model(X, r, tau, a1, b1, a2, b2, q)

        error += (model_price - market_price)**2

    mean_err = (implied_mean(a1, b1, a2, b2, q) - S_forward)**2

    return error + mean_err

