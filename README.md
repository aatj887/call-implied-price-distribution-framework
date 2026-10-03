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
The code has a working UI implemented in Streamlit, which allows the user to calculate the implied price distribution of a given asset by inputting the ticker of the asset and any of the available days to expiration of the option chain data. No API keys are needed.

Although the UI is hosted currently on Streamlit, the code is modularized and can be easily adapted to be used inside a Python or Jupyter Notebook environment (see the 'notebooks' folder for an example of how to do this).


You can also run the UI by running the following command in the terminal:
'streamlit run gui.py'

You will need to install the required libraries and packages first, which can be done by running 'pip install -r requirements.txt' in the terminal.

## Data
- **Option chains:** CBOE's free, 15-minute delayed quotes (the same source OpenBB's default `cboe` provider used), fetched directly in `src/data_sources/cboe.py`. Indices such as SPX or VIX are supported.
- **Risk-free rate:** SOFR (Secured Overnight Financing Rate) from FRED's public CSV download, so no FRED API key is required (`src/data_sources/risk_free.py`).
- **Underlying prices:** Yahoo Finance via `yfinance` (`src/data_sources/prices.py`), used for unadjusted and adjusted closes in notebooks and backtests (`pip install -r requirements-dev.txt`).

### Option chain archive (for backtesting)
After each trading day, `collector/snapshot.py` stores a full option-chain snapshot for the symbols in `collector/watchlist.toml` (core ETFs and indices) plus every current S&P 500 member (`include_sp500 = true`, list from [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies)). The GitHub Actions workflow `.github/workflows/snapshot.yml` runs it, and the snapshots go to a **private** Hugging Face Dataset with one Parquet file per table and day:

| Path | Content |
|---|---|
| `chains/date=YYYY-MM-DD/part-*.parquet` | every contract of every symbol (`underlying_symbol` column) |
| `underlying/date=YYYY-MM-DD/part-*.parquet` | one row per symbol: spot, OHLC, volume, IV30 |
| `rates/sofr/date=YYYY-MM-DD.parquet` | latest SOFR observation |
| `universe/date=YYYY-MM-DD.parquet` | S&P 500 members that day, to avoid survivorship bias in backtests |

The dataset is private because CBOE's website terms do not allow redistributing its quotes. A second scheduled run only fetches symbols that are missing for the day.

Setup: create a Hugging Face write token and add it to the GitHub repository as the `HF_TOKEN` secret. Optionally set the `HF_DATASET_REPO` variable (default `aatj887/option-chain-snapshots`). You can test locally with `python -m collector.snapshot --dry-run --out ./tmp_snap --allow-stale` (add `--symbols SPY QQQ` for a quick run).

Note that GitHub disables scheduled workflows in public repositories after 60 days without repository activity. Re-enable it from the Actions tab if that happens.

Query the archive with DuckDB (`hf auth login` first, since the dataset is private):
```sql
SELECT date, expiration, strike, option_type, bid, ask
FROM read_parquet('hf://datasets/aatj887/option-chain-snapshots/chains/*/*.parquet', hive_partitioning = true)
WHERE underlying_symbol = 'SPY' AND dte = 30;
```

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
