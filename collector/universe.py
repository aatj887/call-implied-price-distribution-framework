import logging
import tomllib
from pathlib import Path

import pandas as pd

# Community-maintained list scraped daily from Wikipedia's "List of S&P 500 companies"
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
WATCHLIST = Path(__file__).with_name("watchlist.toml")

log = logging.getLogger("collector")


def sp500_constituents() -> pd.DataFrame:
    """
    Current S&P 500 members with columns symbol, name, sector. Tickers keep their dot (BRK.B),
    which is also the format CBOE expects.
    """
    df = pd.read_csv(SP500_CSV_URL)
    df = df.rename(columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"})
    df["symbol"] = df["symbol"].str.strip().str.upper()
    return df[["symbol", "name", "sector"]].drop_duplicates("symbol").reset_index(drop=True)


def load_universe(path: Path = WATCHLIST) -> tuple[list[str], pd.DataFrame | None]:
    """
    Returns the symbols to snapshot (watchlist symbols first, then S&P 500 members if enabled)
    and the constituent table (None if it was not requested or could not be downloaded).
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)

    symbols = [s.upper() for s in config.get("symbols", [])]
    constituents = None
    if config.get("include_sp500", False):
        try:
            constituents = sp500_constituents()
            symbols += constituents["symbol"].tolist()
        except Exception as e:  # Still snapshot the core watchlist if the list is unreachable
            log.error("Could not download the S&P 500 constituents (%s). Using the core watchlist only.", e)

    return list(dict.fromkeys(symbols)), constituents
