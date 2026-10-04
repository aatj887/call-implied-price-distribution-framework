"""
Fills in what actually happened for every forecast whose expiry has passed.

For each forecasts/<track>/<expiry>.parquet without a matching outcomes/<track>/<expiry>.parquet,
it downloads closes from Yahoo Finance and writes one row per symbol with the settlement price
(in the same units as the forecast) and whether an earnings announcement fell inside the window.

Yahoo restates closes after splits, so the settlement price is spot * (Yahoo close at expiry /
Yahoo close on the forecast date): the ratio comes from one consistent series and the spot from the
snapshot the forecast was fitted on. A split between the two dates then needs no special handling.

Run from the project root:
    python -m backtest.outcomes --data ./tmp_snap
    HF_TOKEN=... python -m backtest.outcomes
"""
import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from backtest.forecasts import Archive
from collector.snapshot import DEFAULT_REPO_ID

log = logging.getLogger("outcomes")
NEW_YORK = ZoneInfo("America/New_York")
CLOSE = time(16, 0)


def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")  # BRK.B on CBOE is BRK-B on Yahoo


def download_closes(symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
    """
    Daily closes (split-adjusted, not dividend-adjusted) per symbol between start and end inclusive.
    Symbols Yahoo has no data for (delisted, renamed) are left out.
    """
    data = yf.download([yahoo_symbol(s) for s in symbols], start=start, end=end + timedelta(days=1),
                       auto_adjust=False, group_by="ticker", progress=False, multi_level_index=True)
    closes = {}
    for s in symbols:
        try:
            series = data[yahoo_symbol(s)]["Close"].dropna()
        except KeyError:
            continue
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        if not series.empty:
            closes[s] = series
    return closes


def earnings_in_window(symbol: str, forecast_date: date, expiry: date) -> bool | None:
    """
    True if an earnings announcement came after the forecast snapshot (the close on forecast_date) and
    before the close on expiry. None when Yahoo has no calendar for the symbol (ETFs) or the request fails.
    """
    try:
        dates = yf.Ticker(yahoo_symbol(symbol)).get_earnings_dates(limit=12)
    except Exception:
        return None
    if dates is None or dates.empty:
        return None
    start = datetime.combine(forecast_date, CLOSE, NEW_YORK)
    end = datetime.combine(expiry, CLOSE, NEW_YORK)
    stamps = pd.DatetimeIndex(dates.index)
    stamps = stamps.tz_localize(NEW_YORK) if stamps.tz is None else stamps.tz_convert(NEW_YORK)
    return bool(((stamps >= start) & (stamps < end)).any())


def settle(forecasts: pd.DataFrame, closes: dict[str, pd.Series]) -> pd.DataFrame:
    """
    One outcome row per forecast: settlement price in forecast units and the realised price return.
    """
    rows = []
    for f in forecasts.itertuples():
        row = {"track": f.track, "symbol": f.symbol, "forecast_date": f.forecast_date,
               "expiration": f.expiration, "status": "ok"}
        series = closes.get(f.symbol)
        start, end = pd.Timestamp(f.forecast_date), pd.Timestamp(f.expiration)
        if series is None or start not in series.index or end not in series.index:
            rows.append({**row, "status": "no_price"})
            continue
        ratio = float(series[end] / series[start])
        row.update(yahoo_close_start=float(series[start]), yahoo_close_end=float(series[end]),
                   realized_return=ratio - 1)
        if pd.notna(getattr(f, "spot", None)):
            row["settle"] = float(f.spot) * ratio
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_cycle(forecasts: pd.DataFrame, earnings: bool, workers: int = 4) -> pd.DataFrame:
    symbols = forecasts["symbol"].unique().tolist()
    start = min(forecasts["forecast_date"]) - timedelta(days=7)
    end = max(forecasts["expiration"])
    out = settle(forecasts, download_closes(symbols, start, end))

    if earnings:
        windows = list(zip(out["symbol"], out["forecast_date"], out["expiration"]))
        with ThreadPoolExecutor(max_workers=workers) as pool:  # Yahoo throttles large bursts
            out["earnings_in_window"] = list(pool.map(lambda w: earnings_in_window(*w), windows))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record realised prices for expired backtest forecasts.")
    parser.add_argument("--data", help="Local snapshot folder (default: the Hugging Face dataset).")
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO") or DEFAULT_REPO_ID)
    parser.add_argument("--force", action="store_true", help="Recompute expiries that already have a file.")
    parser.add_argument("--no-earnings", action="store_true", help="Skip the earnings calendar lookups.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    archive = Archive(local=Path(args.data)) if args.data else Archive(repo_id=args.repo_id, token=os.environ.get("HF_TOKEN"))
    today = datetime.now(NEW_YORK).date()

    for path in archive.files("forecasts/*/*.parquet"):
        out_path = path.replace("forecasts/", "outcomes/", 1)
        if not args.force and archive.exists(out_path):
            continue
        forecasts = archive.read_file(path)
        # Wait a day after expiry so Yahoo has the final close
        if forecasts.empty or max(forecasts["expiration"]) >= today:
            continue
        df = outcome_cycle(forecasts, earnings=not args.no_earnings)
        archive.write(df, out_path)
        log.info("%s: %d settled / %d. %s", out_path, (df["status"] == "ok").sum(), len(df),
                 df["status"].value_counts().to_dict())

    if not archive.written:
        log.info("No expired forecasts without an outcome.")
    elif args.data:
        log.info("Wrote %d files under %s", len(archive.written), archive.out / "outcomes")
    else:
        archive.upload(f"Outcomes up to {today}")
        log.info("Uploaded %d files to %s", len(archive.written), args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
