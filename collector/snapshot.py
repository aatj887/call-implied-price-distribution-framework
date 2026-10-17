"""
Daily option-chain snapshot collector.

Downloads the full delayed CBOE chain for every symbol in collector/watchlist.toml (plus the S&P 500
members when enabled), stores one Parquet file per table and day, and uploads the new files to a
private Hugging Face Dataset repo:

    chains/date=2026-10-02/part-213012.parquet       <- every contract of every symbol
    underlying/date=2026-10-02/part-213012.parquet   <- one row per symbol (spot, OHLC, volume, IV30)
    rates/sofr/date=2026-10-02.parquet               <- latest SOFR observation
    universe/date=2026-10-02.parquet                 <- S&P 500 members that day (avoids survivorship bias)

Few large files instead of one file per symbol keeps the repo far below Hugging Face's
recommended 100k files. A later run on the same day (the backup cron) only fetches the symbols
missing from the underlying table and writes them as an extra part file.

Run from the project root:
    python -m collector.snapshot --dry-run --out ./tmp_snap      # local only
    HF_TOKEN=... python -m collector.snapshot                     # upload to the dataset repo
"""
import argparse
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from collector.universe import load_universe
from src.data_sources.cboe import fetch_chain_raw, normalize_chain, quote_date
from src.data_sources.risk_free import load_sofr_history

DEFAULT_REPO_ID = "aatj887/option-chain-snapshots"
REFERENCE_SYMBOL = "SPY"  # Its quote date tells whether today was a trading session
UNDERLYING_FIELDS = ["current_price", "bid", "ask", "open", "high", "low", "close",
                     "prev_day_close", "volume", "iv30", "last_trade_time"]

# Explicit column types so every daily file has the same schema, whatever subset of symbols it holds
STRING_COLUMNS = {"contract_symbol", "option_type", "tick", "last_trade_time", "underlying_symbol",
                  "quote_time", "symbol"}
DATE_COLUMNS = {"expiration", "quote_date"}

log = logging.getLogger("collector")


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in STRING_COLUMNS:
            df[col] = df[col].astype("string")
        elif col == "dte":
            df[col] = df[col].astype("int32")
        elif col not in DATE_COLUMNS and col != "snapshot_ts_utc":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


class Store:
    """
    Where snapshots already live: the Hugging Face dataset, or the local output folder on a dry run.
    """
    def __init__(self, out: Path, repo_id: str | None = None, token: str | None = None):
        self.out, self.repo_id, self.fs = out, repo_id, None
        if repo_id:
            from huggingface_hub import HfApi, HfFileSystem
            self.api = HfApi(token=token)
            # Private on purpose: CBOE's website terms do not allow redistributing its quotes
            self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
            self.fs = HfFileSystem(token=token)

    def stored_symbols(self, day: str) -> set[str]:
        if self.fs is None:
            files = [str(p) for p in self.out.glob(f"underlying/date={day}/*.parquet")]
            opener = open
        else:
            files = self.fs.glob(f"datasets/{self.repo_id}/underlying/date={day}/*.parquet")
            opener = self.fs.open
        symbols = set()
        for path in files:
            with opener(path, "rb") as f:
                symbols |= set(pd.read_parquet(f, columns=["symbol"])["symbol"])
        return symbols

    def upload(self, written: list[Path], message: str) -> None:
        self.api.upload_folder(
            folder_path=str(self.out), repo_id=self.repo_id, repo_type="dataset",
            allow_patterns=[p.as_posix() for p in written], commit_message=message,
        )


def write_parquet(df: pd.DataFrame, out: Path, path: Path) -> Path:
    (out / path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / path, index=False, compression="zstd")
    return path


def fetch_symbol(symbol: str, day: date, raw: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Downloads and normalizes one symbol. Raises if its quotes are not from the session being stored.
    """
    raw = raw or fetch_chain_raw(symbol)
    if quote_date(raw) != day:
        raise ValueError(f"quotes are from {quote_date(raw)}, not {day}")
    chain = normalize_chain(raw)
    underlying = {"symbol": symbol, **{k: raw["data"].get(k) for k in UNDERLYING_FIELDS}}
    return chain, underlying


def fetch_all(symbols: list[str], day: date, workers: int, cached: dict) -> tuple[list, list, list[str]]:
    """
    Fetches symbols in parallel, then retries the failures once, one at a time, after a pause
    (CBOE's CDN occasionally throttles bursts).
    """
    chains, underlyings, failed = [], [], []

    def attempt(symbol):
        try:
            return symbol, fetch_symbol(symbol, day, cached.pop(symbol, None)), None
        except Exception as e:
            return symbol, None, e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(attempt, symbols))

    retry = [s for s, res, err in results if err is not None]
    if retry:
        log.warning("%d symbols failed, retrying them sequentially: %s", len(retry), retry[:20])
        time.sleep(10)
        results = [r for r in results if r[2] is None] + [attempt(s) for s in retry]

    for symbol, res, err in results:
        if err is not None:
            log.error("%s: failed (%s)", symbol, err)
            failed.append(symbol)
        else:
            chains.append(res[0])
            underlyings.append(res[1])
    return chains, underlyings, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot CBOE option chains to Parquet / Hugging Face.")
    parser.add_argument("--symbols", nargs="*", help="Override the watchlist (and the S&P 500 list).")
    parser.add_argument("--out", type=Path, help="Local output folder (default: a temporary folder).")
    parser.add_argument("--dry-run", action="store_true", help="Write files locally without uploading.")
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO") or DEFAULT_REPO_ID)
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads (CBOE rate-limits bursts).")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Store the last session's quotes even if it was not today (weekends, holidays, tests).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.symbols:
        symbols, constituents = [s.upper() for s in args.symbols], None
    else:
        symbols, constituents = load_universe()
    out = args.out or Path(tempfile.mkdtemp(prefix="snapshots_"))
    store = Store(out) if args.dry_run else Store(out, args.repo_id, os.environ.get("HF_TOKEN"))

    # Which session do the quotes belong to? Skip weekends and market holidays.
    reference = fetch_chain_raw(REFERENCE_SYMBOL)
    session = quote_date(reference)
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    if session != today_et and not args.allow_stale:
        log.warning("Latest quotes are from %s, not %s: market closed today. Nothing to store.", session, today_et)
        return 0
    day = session.isoformat()

    done = store.stored_symbols(day)
    todo = [s for s in symbols if s not in done]
    if not todo:
        log.info("All %d symbols are already stored for %s.", len(symbols), day)
        return 0
    log.info("Snapshotting %d symbols for %s (%d already stored).", len(todo), day, len(done))

    started = time.time()
    chains, underlyings, failed = fetch_all(todo, session, args.workers, {REFERENCE_SYMBOL: reference})
    log.info("Fetched %d/%d symbols in %.0fs.", len(chains), len(todo), time.time() - started)
    if not chains:
        # On a backup run the leftovers are usually symbols that fail every day (stale or missing chains)
        if done:
            log.warning("Nothing new to store; %d symbols still failing: %s", len(failed), failed)
            return 0
        return 1

    snapshot_ts = datetime.now(timezone.utc)
    part = f"part-{snapshot_ts:%H%M%S}.parquet"
    written = []

    chain = pd.concat(chains, ignore_index=True)
    chain["snapshot_ts_utc"] = snapshot_ts
    written.append(write_parquet(enforce_schema(chain), out, Path(f"chains/date={day}/{part}")))

    underlying = pd.DataFrame(underlyings)
    underlying["quote_date"] = session
    underlying["snapshot_ts_utc"] = snapshot_ts
    written.append(write_parquet(enforce_schema(underlying), out, Path(f"underlying/date={day}/{part}")))

    if not done:  # First run of the day: also store the rate and the index membership
        try:
            sofr = load_sofr_history((session - pd.Timedelta(days=14)).isoformat(), day)
            rate = pd.DataFrame([{"quote_date": session, "sofr_date": sofr.index[-1].date(), "sofr": sofr.iloc[-1]}])
            written.append(write_parquet(rate, out, Path(f"rates/sofr/date={day}.parquet")))
        except Exception as e:
            log.error("SOFR for %s: failed (%s)", day, e)
        if constituents is not None:
            written.append(write_parquet(constituents.assign(quote_date=session), out, Path(f"universe/date={day}.parquet")))

    log.info("%d contracts across %d symbols (%.1f MB).", len(chain), len(underlyings),
             (out / written[0]).stat().st_size / 1e6)

    if args.dry_run:
        log.info("Wrote %d files to %s", len(written), out)
    else:
        store.upload(written, f"Snapshot {day}: {len(underlyings)} symbols")
        log.info("Uploaded %d files to %s", len(written), args.repo_id)

    if failed:
        log.warning("%d symbols failed: %s", len(failed), failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
