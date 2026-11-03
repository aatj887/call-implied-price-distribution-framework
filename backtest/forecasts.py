"""
Builds the forecast table for the backtest from the stored option-chain snapshots.

For every expiry cycle of a track, it picks the last snapshot taken at least `target_dte` days before
expiry, fits the implied distribution for each symbol, and writes one row per symbol:

    forecasts/monthly30/2026-11-20.parquet   <- 30-day forecasts for the November monthly expiry
    forecasts/weekly7/2026-10-09.parquet     <- 7-day forecasts for the ETFs' weekly expiry

Each row keeps the fitted parameters (so any probability can be recomputed later), a single-lognormal
fit on the same quotes as the comparison forecast, fit diagnostics and a few headline numbers.
Expiries whose file already exists are skipped, so it is safe to run every day after the collector.
The realised price at expiry is filled in later by a separate outcome step.

Run from the project root:
    python -m backtest.forecasts --data ./tmp_snap        # snapshots in a local folder (collector --dry-run)
    HF_TOKEN=... python -m backtest.forecasts             # read from / upload to the dataset repo
"""
import argparse
import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm

from collector.snapshot import DEFAULT_REPO_ID
from scripts.run_build_pdf import clean_chain, fit_distribution_result, starting_points
from src.data_sources.risk_free import risk_free_asof
from src.modelling.minimization import call_price_model, objective_from_chain, put_price_model

log = logging.getLogger("forecasts")

ETFS = ["SPY", "QQQ", "IWM", "DIA"]
MIN_STRIKES = 8  # Fewer distinct strikes than this cannot pin down five parameters
CHAIN_COLUMNS = ["underlying_symbol", "expiration", "dte", "option_type", "strike", "bid", "ask", "underlying_price"]


@dataclass(frozen=True)
class Track:
    target_dte: int    # Forecast horizon: use the last snapshot at least this many days before expiry
    tolerance: int     # ...but no more than target_dte + tolerance days before (gaps in the archive)
    monthly: bool      # Third-Friday expiries only, otherwise every Friday
    symbols: tuple | None  # None = ETFs + S&P 500 members on the forecast date


TRACKS = {
    "monthly30": Track(target_dte=30, tolerance=5, monthly=True, symbols=None),
    "weekly7": Track(target_dte=7, tolerance=2, monthly=False, symbols=tuple(ETFS)),
}


# ---------------------------------------------------------------- calendar

def expiry_fridays(start: date, end: date, monthly: bool) -> list[date]:
    """
    Fridays between start and end (inclusive); with monthly=True only third Fridays (day 15-21).
    When the Friday is a market holiday options expire on the Thursday, which fit_symbol handles.
    """
    first = start + timedelta(days=(4 - start.weekday()) % 7)
    fridays = [first + timedelta(weeks=i) for i in range((end - first).days // 7 + 1)] if first <= end else []
    return [f for f in fridays if not monthly or 15 <= f.day <= 21]


def pick_forecast_date(friday: date, snapshot_dates: list[date], track: Track) -> date | None:
    """
    Last snapshot at least target_dte days before the expiry, or None if there is none within the
    tolerance. Only call it once the archive has reached friday - target_dte, so later snapshots
    cannot change the answer.
    """
    cutoff = friday - timedelta(days=track.target_dte)
    eligible = [d for d in snapshot_dates if d <= cutoff]
    if not eligible or (friday - eligible[-1]).days > track.target_dte + track.tolerance:
        return None
    return eligible[-1]


# ---------------------------------------------------------------- distribution helpers

def mixture_cdf(s, a1, b1, a2, b2, q):
    """
    P(S_T <= s) under the two-lognormal mixture.
    """
    s = np.maximum(s, 1e-300)
    return q * norm.cdf((np.log(s) - a1) / b1) + (1 - q) * norm.cdf((np.log(s) - a2) / b2)


def mixture_quantile(p, a1, b1, a2, b2, q):
    """
    Price level with probability p of ending below it.
    """
    lo = np.exp(min(a1 - 10*b1, a2 - 10*b2))
    hi = np.exp(max(a1 + 10*b1, a2 + 10*b2))
    return brentq(lambda s: mixture_cdf(s, a1, b1, a2, b2, q) - p, lo, hi)


def price_rmse(theta, chain: pd.DataFrame, r: float) -> float:
    """
    Root mean squared gap between model and mid prices, in the underlying's currency.
    """
    a1, b1, a2, b2, q = theta
    tau = chain['dte'].iloc[0] / 365
    S0 = chain['underlying_price'].iloc[0]
    X = chain['strike'].to_numpy(dtype=float)
    mid = 0.5 * (chain['bid'].to_numpy(dtype=float) + chain['ask'].to_numpy(dtype=float))
    model = np.where(chain['option_type'].to_numpy() == 'call',
                     call_price_model(X, r, tau, a1, b1, a2, b2, q),
                     put_price_model(X, r, tau, a1, b1, a2, b2, q, S0))
    return float(np.sqrt(np.mean((model - mid)**2)))


def fit_lognormal(chain: pd.DataFrame, r: float):
    """
    Single-lognormal (Black-Scholes) fit on the same quotes: the comparison forecast.
    Uses the same objective with both mixture components forced equal.
    """
    _, _, m, s, _ = starting_points(chain, r)[1]
    with np.errstate(over='ignore', invalid='ignore'):
        res = minimize(lambda x: objective_from_chain([x[0], x[1], x[0], x[1], 1.0], chain, r),
                       [m, s], bounds=[(None, None), (1e-6, None)])
    return res.x


# ---------------------------------------------------------------- one symbol

def fit_symbol(symbol: str, chain: pd.DataFrame, friday: date, forecast_date: date, r: float) -> dict:
    """
    One forecast row. Never raises: problems end up in the status and error columns.
    """
    row = {"symbol": symbol, "forecast_date": forecast_date, "rate": r, "status": "ok", "error": None}
    expirations = set(chain["expiration"])
    expiry = friday if friday in expirations else friday - timedelta(days=1)
    if expiry not in expirations:
        return {**row, "expiration": friday, "status": "no_expiry"}

    dte = (expiry - forecast_date).days
    row.update(expiration=expiry, dte=dte, spot=float(chain["underlying_price"].iloc[0]))

    usable = clean_chain(chain, dte)
    row.update(n_quotes=len(usable), n_strikes=usable["strike"].nunique())
    if row["n_strikes"] < MIN_STRIKES:
        return {**row, "status": "too_few_strikes"}

    try:
        res, usable = fit_distribution_result(chain, r, dte)
        theta = res.x
        ln_a, ln_b = fit_lognormal(usable, r)
        row.update(dict(zip(["a1", "b1", "a2", "b2", "q"], map(float, theta))))
        row.update(ln_a=float(ln_a), ln_b=float(ln_b), converged=bool(res.success),
                   rmse=price_rmse(theta, usable, r),
                   rmse_ln=price_rmse([ln_a, ln_b, ln_a, ln_b, 1.0], usable, r))

        # Headline numbers (the parameters are enough to recompute anything else later)
        S0 = row["spot"]
        row["p_up"] = float(1 - mixture_cdf(S0, *theta))
        row["p_up_ln"] = float(1 - norm.cdf((np.log(S0) - ln_a) / ln_b))
        for p in (0.1, 0.5, 0.9):
            row[f"q{int(p*100)}"] = float(mixture_quantile(p, *theta))
    except Exception as e:
        row.update(status="fit_failed", error=str(e)[:200])
    return row


def _fit_symbol_args(args):
    return fit_symbol(*args)


# ---------------------------------------------------------------- storage

class Archive:
    """
    The snapshot archive: a local folder (collector --dry-run output) or the Hugging Face dataset.
    New files are written to a local folder and, for the dataset, uploaded together at the end.
    """
    def __init__(self, local: Path | None = None, repo_id: str | None = None, token: str | None = None):
        self.repo_id, self.written = repo_id, []
        if local is not None:
            self.root, self.out, self.fs = Path(local), Path(local), None
        else:
            from huggingface_hub import HfApi, HfFileSystem
            self.api, self.fs = HfApi(token=token), HfFileSystem(token=token)
            self.root, self.out = f"datasets/{repo_id}", Path(tempfile.mkdtemp(prefix="forecasts_"))

    def _glob(self, pattern: str) -> list[str]:
        if self.fs is None:
            return sorted(p.relative_to(self.root).as_posix() for p in self.root.glob(pattern))
        return sorted(p[len(self.root) + 1:] for p in self.fs.glob(f"{self.root}/{pattern}"))

    def _open(self, path: str):
        return open(self.root / path, "rb") if self.fs is None else self.fs.open(f"{self.root}/{path}", "rb")

    def dates(self, table: str) -> list[date]:
        found = {re.search(r"date=(\d{4}-\d{2}-\d{2})", p) for p in self._glob(f"{table}/date=*")}
        return sorted(date.fromisoformat(m.group(1)) for m in found if m)

    def read(self, table: str, day: date, **kwargs) -> pd.DataFrame:
        paths = self._glob(f"{table}/date={day}/*.parquet") + self._glob(f"{table}/date={day}.parquet")
        frames = []
        for path in paths:
            with self._open(path) as f:
                frames.append(pd.read_parquet(f, **kwargs))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def files(self, pattern: str) -> list[str]:
        return self._glob(pattern)

    def read_file(self, path: str, **kwargs) -> pd.DataFrame:
        with self._open(path) as f:
            return pd.read_parquet(f, **kwargs)

    def exists(self, path: str) -> bool:
        return bool(self._glob(path))

    def write(self, df: pd.DataFrame, path: str) -> None:
        (self.out / path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.out / path, index=False, compression="zstd")
        self.written.append(path)

    def upload(self, message: str) -> None:
        if self.fs is not None and self.written:
            self.api.upload_folder(folder_path=str(self.out), repo_id=self.repo_id, repo_type="dataset",
                                   allow_patterns=self.written, commit_message=message)


# ---------------------------------------------------------------- one expiry cycle

def rate_on(archive: Archive, day: date) -> float:
    stored = archive.read("rates/sofr", day)
    return float(stored["sofr"].iloc[0]) if not stored.empty else risk_free_asof(day)


def symbols_on(archive: Archive, day: date, track: Track) -> list[str]:
    if track.symbols is not None:
        return list(track.symbols)
    members = archive.read("universe", day)
    if members.empty:  # Older or partial snapshots: fall back to every stored stock and ETF
        log.warning("No S&P 500 membership stored for %s, using every stored symbol.", day)
        stored = archive.read("underlying", day, columns=["symbol"])["symbol"]
        members = pd.DataFrame({"symbol": [s for s in stored if s not in {"SPX", "NDX", "RUT", "VIX"}]})
    return list(dict.fromkeys(ETFS + members["symbol"].tolist()))


def forecast_cycle(archive: Archive, track: Track, friday: date, forecast_date: date, workers: int) -> pd.DataFrame:
    symbols = symbols_on(archive, forecast_date, track)
    chains = archive.read("chains", forecast_date, columns=CHAIN_COLUMNS,
                          filters=[("expiration", "in", [friday, friday - timedelta(days=1)]),
                                   ("underlying_symbol", "in", symbols)])
    underlying = archive.read("underlying", forecast_date, columns=["symbol", "iv30"]).drop_duplicates("symbol")
    r = rate_on(archive, forecast_date)

    by_symbol = dict(tuple(chains.groupby("underlying_symbol"))) if not chains.empty else {}
    jobs = [(s, by_symbol[s], friday, forecast_date, r) for s in symbols if s in by_symbol]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_fit_symbol_args, jobs, chunksize=8))
    rows += [{"symbol": s, "forecast_date": forecast_date, "expiration": friday, "rate": r, "status": "no_chain"}
             for s in symbols if s not in by_symbol]

    df = pd.DataFrame(rows).merge(underlying, on="symbol", how="left")
    df["iv30"] = df["iv30"] / 100  # CBOE quotes IV30 in percent
    return df


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit implied distributions for the backtest forecast table.")
    parser.add_argument("--data", type=Path, help="Local snapshot folder (default: the Hugging Face dataset).")
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO") or DEFAULT_REPO_ID)
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS), choices=list(TRACKS))
    parser.add_argument("--force", action="store_true", help="Recompute expiries that already have a file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel fits.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    archive = Archive(local=args.data) if args.data else Archive(repo_id=args.repo_id, token=os.environ.get("HF_TOKEN"))
    snapshot_dates = archive.dates("underlying")
    if not snapshot_dates:
        log.error("No snapshots found.")
        return 1
    first, latest = snapshot_dates[0], snapshot_dates[-1]
    log.info("Snapshots from %s to %s (%d days).", first, latest, len(snapshot_dates))

    for name in args.tracks:
        track = TRACKS[name]
        # Cycles whose forecast date is already settled: friday - target_dte between first and latest
        for friday in expiry_fridays(first + timedelta(days=track.target_dte),
                                     latest + timedelta(days=track.target_dte), track.monthly):
            path = f"forecasts/{name}/{friday}.parquet"
            if not args.force and archive.exists(path):
                continue
            forecast_date = pick_forecast_date(friday, snapshot_dates, track)
            if forecast_date is None:
                log.warning("%s %s: no snapshot within %d-%d days before expiry, skipped.", name, friday,
                            track.target_dte, track.target_dte + track.tolerance)
                continue
            df = forecast_cycle(archive, track, friday, forecast_date, args.workers)
            df.insert(0, "track", name)
            archive.write(df, path)
            log.info("%s %s: forecasts from %s, %d ok / %d symbols. %s", name, friday, forecast_date,
                     (df["status"] == "ok").sum(), len(df), df["status"].value_counts().to_dict())

    if not archive.written:
        log.info("Nothing new to forecast.")
    elif args.data:
        log.info("Wrote %d files under %s", len(archive.written), archive.out / "forecasts")
    else:
        archive.upload(f"Forecasts up to {latest}")
        log.info("Uploaded %d files to %s", len(archive.written), args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
