import re
import time
from datetime import date, datetime

import pandas as pd
import requests

# Free, keyless delayed-quotes endpoint used by cboe.com (and by OpenBB's 'cboe' provider).
# It is unofficial: keep all parsing in this module so the source can be swapped if it changes.
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; call-implied-price-distribution-framework)"}

# Cash-settled indices are served under an underscore prefix (e.g. _SPX)
INDEX_SYMBOLS = {"SPX", "SPXW", "XSP", "VIX", "NDX", "RUT", "DJX", "OEX", "XEO"}

# OCC option symbol: ROOT + YYMMDD + C/P + strike * 1000 (8 digits), e.g. SPY261016C00500000
OCC_PATTERN = re.compile(r"^(?P<root>[A-Z0-9.]+?)(?P<expiry>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

OPTION_COLUMNS = [
    "contract_symbol", "expiration", "dte", "option_type", "strike",
    "bid", "ask", "bid_size", "ask_size", "last_trade_price", "last_trade_time",
    "open", "high", "low", "prev_day_close", "change", "percent_change", "tick",
    "volume", "open_interest", "iv", "delta", "gamma", "theta", "vega", "rho", "theo",
]


def to_cboe_symbol(symbol: str) -> str:
    """
    Maps a user ticker (e.g. 'spx', '^SPX', 'SPY') to the symbol the CBOE endpoint expects.
    """
    s = symbol.strip().upper().lstrip("^_")
    return f"_{s}" if s in INDEX_SYMBOLS else s


def fetch_chain_raw(symbol: str, retries: int = 5, timeout: float = 30) -> dict:
    """
    Downloads the full delayed option chain (all expiries) for a symbol as the raw CBOE JSON payload.

    Parameters:
    symbol (str): Ticker symbol, e.g. 'SPY' or 'SPX'.
    retries (int): Extra attempts on network errors and rate limiting (HTTP 429).
    timeout (float): Seconds before a request times out.
    """
    url = CBOE_URL.format(symbol=to_cboe_symbol(symbol))
    for attempt in range(retries + 1):
        try:
            # The CDN answers with a 307 redirect to cdn-api.cboe.com; requests follows it by default
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 403 or response.status_code == 404:
                raise ValueError(f"CBOE has no option chain for '{symbol}'.")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == retries:
                raise
            # Back off exponentially (2, 4, 8... s), longer when CBOE says we are sending too many requests
            wait = 2 ** (attempt + 1)
            if getattr(e.response, "status_code", None) == 429:
                retry_after = e.response.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else 5 * 2 ** attempt
            time.sleep(wait)


def quote_date(raw: dict) -> date:
    """
    Trading date the payload's quotes belong to (the underlying's last trade), not the download date.
    After the close and over weekends CBOE keeps serving the last session's quotes.
    """
    return datetime.fromisoformat(raw["data"]["last_trade_time"]).date()


def normalize_chain(raw: dict, asof: date | None = None) -> pd.DataFrame:
    """
    Converts a raw CBOE payload into the chain schema used by the model
    (strike, bid, ask, option_type, dte, underlying_price, ...).

    Parameters:
    raw (dict): Payload returned by fetch_chain_raw.
    asof (date): Date the days-to-expiry are measured from. Defaults to the payload's quote date.
    """
    data = raw["data"]
    asof = asof or quote_date(raw)
    if not data.get("options"):
        raise ValueError(f"CBOE lists no options for '{data.get('symbol')}'.")

    df = pd.DataFrame(data["options"]).rename(columns={"option": "contract_symbol"})
    parts = df["contract_symbol"].str.extract(OCC_PATTERN)
    if parts.isna().any().any():
        bad = df.loc[parts.isna().any(axis=1), "contract_symbol"].head(3).tolist()
        raise ValueError(f"Unrecognised option symbols in CBOE payload: {bad}")

    df["expiration"] = pd.to_datetime(parts["expiry"], format="%y%m%d").dt.date
    df["option_type"] = parts["cp"].map({"C": "call", "P": "put"})
    df["strike"] = parts["strike"].astype(int) / 1000
    df["dte"] = (pd.to_datetime(df["expiration"]) - pd.Timestamp(asof)).dt.days

    # Contracts that expired before the quote date can linger in the payload
    df = df[df["dte"] >= 0]

    df = df.reindex(columns=OPTION_COLUMNS)
    df["underlying_symbol"] = data["symbol"].lstrip("_")
    df["underlying_price"] = data["current_price"]
    df["quote_date"] = asof
    df["quote_time"] = data["last_trade_time"]

    return df.sort_values(["expiration", "strike", "option_type"]).reset_index(drop=True)


def get_option_chains(symbol: str) -> pd.DataFrame:
    """
    Get the latest (15-minute delayed) option chains for a specific stock or index.

    Parameters:
    symbol (str): The stock symbol to get option chains for.

    Returns:
    pd.DataFrame: A DataFrame containing the option chains for the specified stock.
    """
    return normalize_chain(fetch_chain_raw(symbol))
