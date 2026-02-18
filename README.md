# call-implied-price-distribution-framework
# THIS PROJECT IS STILL IN DEVELOPMENT. NO FINAL USABLE CODE IS AVAILABLE YET.
This repository utilises a variety of Python libraries and packages to generate numerical solutions to the analysis done in the paper 'Probability distributions of future asset prices implied by option prices' by Bhupinder Bahra.
Link: https://www.bankofengland.co.uk/-/media/boe/files/quarterly-bulletin/1996/probability-distributions-of-future-asset-prices-implied-by-option-prices.pdf

## Objectives
- Generate simple risk-free rate estimations for any defined timeframe by calculating an established SOFR-based discount factor based on the time to expiration of the options analysed.
- Generate the implied price distribution of a given asset using the expected return pricing method and the methodology described by the aforementioned paper using scipy.
- 



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

$$c(X) = e^{-r(T-t)}\int_X^{\infty} \theta L(\alpha_1, \beta_1; S_T) + (1-\theta)L(\alpha_2, \beta_2; S_T) \cdot (S_T - X)dS_T,$$

which can be expanded as

$$c(X) = e^{-r(T-t)} \theta \int_X^{\infty} L(\alpha_1, \beta_1; S_T)(S_T-X)dS_T + (1-\theta)\int_X^{\infty}L(\alpha_2, \beta_2; S_T) \cdot (S_T - X)dS_T.$$


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or also feel free to fork the project!

## License
[MIT](https://choosealicense.com/licenses/mit/)
