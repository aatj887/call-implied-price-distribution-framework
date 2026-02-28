---
title: "call-implied-price-distribution-framework"
sdk: "docker"
app_file: gui.py
pinned: false
---

# call-implied-price-distribution-framework

This project utilises a variety of Python libraries and packages to generate numerical solutions to the analysis done in the paper 'Probability distributions of future asset prices implied by option prices' by Bhupinder Bahra.
Link: https://www.bankofengland.co.uk/-/media/boe/files/quarterly-bulletin/1996/probability-distributions-of-future-asset-prices-implied-by-option-prices.pdf

A fully working webapp version can be found at https://huggingface.co/spaces/aatj887/call-implied-price-distribution-framework. The code is also available in this repository and can be run locally by following the instructions in the 'Usage' section below.

## Implementation notes
### 1. Transforming the objective function
The paper establishes that the price of a call with strike price $X$ and with years to expiration $T-t$ is equal to the present-discounted expected value of the future payoffs of the call option:

$$c(X) = e^{-r(T-t)}\int_X^{\infty} q(S_T) \cdot (S_T - X)dS_T.$$

The same process is done to calculate the price of puts, changing the payoff value to $(X - S_T)$.

Initially, the code calculated the integrals described above using scipy.integrate, which is computationally intensive and took a long time to run in a consumer-level desktop.

A change had to be done based on the following fact:

The methodology in the paper describes $q(S_T)$ as a weighted sum of two log-normal distributions:
$$q(S_T) = \theta L(\alpha_1, \beta_1; S_T) + (1-\theta)L(\alpha_2, \beta_2; S_T).$$

then we can rewrite the price of a call as

$$c(X) = e^{-r(T-t)}\int_X^{\infty} (\theta L(\alpha_1, \beta_1; S_T) + (1-\theta)L(\alpha_2, \beta_2; S_T)) \cdot (S_T - X)dS_T,$$

which can be expanded as

$$c(X) = e^{-r(T-t)} (\theta \int_X^{\infty} L(\alpha_1, \beta_1; S_T)(S_T-X)dS_T + (1-\theta)\int_X^{\infty}L(\alpha_2, \beta_2; S_T) \cdot (S_T - X)dS_T).$$

This last equation can be represented using the Black-Scholes formula for call options, which has a closed-form solution. This makes it possible to rewrite the price of a call as an arithmetic expression rather than an integral, which is much faster to compute in Python.


### 2. Optimization process
The optimization process is done using scipy.optimize.minimize, which is a general-purpose optimization function that allows for the minimization of a given objective function. The objective function is defined as:

$$\min_{\alpha_1,\alpha_2,\beta_1,\beta_2,\theta} \sum_{i=1}^{n}[c(X_i) - \hat{c}_i]^2 + \sum_{i=1}^{n}[p(X_i) - \hat{p}_i]^2 + \left|\theta e^{\alpha_1+\frac{\beta_1^2}{2}} + (1-\theta)e^{\alpha_2+\frac{\beta_2^2}{2}} - e^{r(T-t)}S_t\right|^2$$


## Usage
The code has a working UI implemented in Streamlit, which allows the user to calculate the implied price distribution of a given asset by inputting an API key for FRED (which is free to obtain at https://fred.stlouisfed.org/), the ticker of the asset and any of the available days to expiration of the option chain data The UI also allows the user to input the strike prices and the corresponding call and put prices of the options analysed.

Although the UI is hosted currently on Streamlit, the code is modularized and can be easily adapted to be used inside a Python or Jupyter Notebook environment (see the 'notebooks' folder for an example of how to do this).


You can also run the UI by running the following command in the terminal:
'streamlit run gui.py'

You will need to install the required libraries and packages first, which can be done by running 'pip install -r requirements.txt' in the terminal.

## Data
The data used in the code is obtained from the FRED API, which provides a wide range of economic data, including interest rates and asset prices. The code uses the SOFR (Secured Overnight Financing Rate) as the risk-free rate, which is obtained from the FRED API using the 'SOFR' ticker. The code also uses the asset price data obtained from YFinance through the OpenBB library, which is a Python library that simplifies the access to financial data from sources like Yahoo Finance, Alpha Vantage, and others.

## Important remarks
- The code is not intended to be used for trading or investment purposes, but rather as a tool for educational and research purposes to understand the methodology described in the paper and to generate the implied price distribution of a given asset based on the option prices.
- The code is not optimized for performance and may take a long time to run, especially for large datasets or for a large number of options analysed. The optimization process can be computationally intensive, and the code is not intended to be used for real-time analysis or for large-scale applications without further optimization and improvements.
- The code is provided as is, without any warranty or guarantee of accuracy or reliability. The results obtained from the code should be interpreted with caution and should not be used as the sole basis for any financial decisions or actions.

## Possible improvements and future work
- In theory, we could calculate an expected value for the return of the asset at time T using the distribution obtained from the optimization process. This could then be compared to either a risk-free return or to the expected return of another asset to calculate a risk premium, which could be used as a supporting indicatior for trading or investment decisions. This is not currently implemented in the code, but it could be an interesting addition to the project in the future.
- The modelling process could be improved by using past data as well as current data through a time-series analysis of the parameters of the distribution. This could potentially improve the accuracy of the implied price distribution obtained from the optimization process, as it would take into account the historical behaviour of the asset and its options. Added to this, we could either validate or invalidate the theoretical framework described in the paper by backtesting the results obtained from the code against historical data and comparing the implied price distribution with the actual price distribution observed in the market.
- The code could be optimized for performance by using more efficient algorithms for the optimization process, such as gradient-based optimization methods or by using parallel processing to speed up the calculations. This would make the code more suitable for real-time analysis and for larger datasets, which could be useful for practical applications in trading or investment decisions.
- The code could be extended to include other types of options, such as put options or exotic options, which would allow for a more comprehensive analysis of the implied price distribution and its implications for trading and investment decisions.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or also feel free to fork the project!

## License
[MIT](https://choosealicense.com/licenses/mit/)
