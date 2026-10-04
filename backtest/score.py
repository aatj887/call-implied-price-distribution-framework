"""
Scores the forecasts against what happened and prints a plain-language report.

Three questions, per track and for a few subgroups (ETFs, stocks, earnings in the window or not):
  1. When the forecast said "X% chance of ending higher", did it end higher about X% of the time?
  2. Where did prices land inside the forecast? Each tenth of the forecast should catch ~10% of
     outcomes, and the 10-90% range should contain the price ~80% of the time.
  3. Did the two-lognormal mixture beat a single lognormal (bell curve) fitted on the same quotes?
     Scored with the log score: the log of the probability density each forecast gave to the price
     that actually happened (higher is better).

Stocks move together, so one expiry's forecasts are not independent: comparisons are averaged per
expiry first and the "t" value is computed across expiries.

Run from the project root:
    python -m backtest.score --data ./tmp_snap [--out report.md]
    HF_TOKEN=... python -m backtest.score
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm

from backtest.forecasts import ETFS, Archive, mixture_cdf
from collector.snapshot import DEFAULT_REPO_ID
from src.modelling.minimization import mixture_pdf

KEYS = ["track", "symbol", "forecast_date", "expiration"]
PROB_BINS = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1]
MIN_EXPIRIES = 6  # Below this the report says it is too early to conclude


def load(archive: Archive) -> pd.DataFrame:
    """
    Forecasts joined with their outcomes, keeping only fitted forecasts with a settlement price.
    """
    def read_all(pattern):
        frames = [archive.read_file(p) for p in archive.files(pattern)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    forecasts, outcomes = read_all("forecasts/*/*.parquet"), read_all("outcomes/*/*.parquet")
    if forecasts.empty or outcomes.empty:
        return pd.DataFrame()
    outcomes = outcomes.drop(columns=["status"]).loc[outcomes["status"] == "ok"]
    df = forecasts[forecasts["status"] == "ok"].merge(outcomes, on=KEYS, how="inner")
    return df[df["settle"].notna()].reset_index(drop=True)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, for the mixture and the single lognormal ("_ln"):
      pit      : forecast probability of ending below the realised price (uniform if well calibrated)
      logscore : log of the forecast density at the realised price
    and went_up: whether the price ended above the spot price.
    """
    df = df.copy()
    theta = [df[c].to_numpy() for c in ["a1", "b1", "a2", "b2", "q"]]
    s = df["settle"].to_numpy()
    df["pit"] = mixture_cdf(s, *theta)
    df["pit_ln"] = norm.cdf((np.log(s) - df["ln_a"]) / df["ln_b"])
    with np.errstate(divide="ignore"):
        df["logscore"] = np.log(mixture_pdf(s, *theta))
        df["logscore_ln"] = np.log(lognorm.pdf(s, s=df["ln_b"], scale=np.exp(df["ln_a"])))
    df["went_up"] = df["settle"] > df["spot"]
    return df


def calibration_table(df: pd.DataFrame, prob: str = "p_up") -> pd.DataFrame:
    """
    Forecasts grouped by their stated chance of ending higher, against how often that happened.
    """
    bins = pd.cut(df[prob], PROB_BINS, include_lowest=True)
    t = df.groupby(bins, observed=True).agg(forecast=(prob, "mean"), happened=("went_up", "mean"),
                                            count=("went_up", "size"))
    t.index = [f"{iv.left:.0%}-{iv.right:.0%}" for iv in t.index]
    return t


def pit_shares(df: pd.DataFrame, col: str = "pit") -> pd.Series:
    """
    Share of outcomes in each tenth of the forecast distribution (0.10 each if well calibrated).
    """
    counts = pd.cut(df[col], np.linspace(0, 1, 11), include_lowest=True).value_counts(sort=False)
    counts.index = [f"{int(iv.left*100)}-{int(iv.right*100)}%" for iv in counts.index]
    return counts / counts.sum()


def logscore_comparison(df: pd.DataFrame) -> dict:
    """
    Mixture minus single-lognormal log score, averaged per expiry, then summarised across expiries.
    """
    finite = df[np.isfinite(df["logscore"]) & np.isfinite(df["logscore_ln"])]
    per_expiry = (finite["logscore"] - finite["logscore_ln"]).groupby(finite["expiration"]).mean()
    n = len(per_expiry)
    sd = per_expiry.std(ddof=1) if n > 1 else np.nan
    return {"expiries": n, "mixture_wins": int((per_expiry > 0).sum()), "mean_gain": float(per_expiry.mean()),
            "t": float(per_expiry.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
            "dropped": len(df) - len(finite)}


def section(df: pd.DataFrame, title: str) -> list[str]:
    n_exp = df["expiration"].nunique()
    lines = [f"### {title}: {len(df)} forecasts over {n_exp} expiries", ""]
    if n_exp < MIN_EXPIRIES:
        lines += [f"_Only {n_exp} expiries so far: treat everything below as anecdotal "
                  f"(wait for at least {MIN_EXPIRIES})._", ""]

    lines += ["**1. Did the chance of ending higher match reality?**", "",
              "| Forecast said | Model average | Ended higher | Bell curve said | Forecasts |", "|---|---|---|---|---|"]
    cal = calibration_table(df)
    cal_ln = df.groupby(pd.cut(df["p_up"], PROB_BINS, include_lowest=True), observed=True)["p_up_ln"].mean()
    for (label, r), ln in zip(cal.iterrows(), cal_ln):
        lines.append(f"| {label} | {r['forecast']:.0%} | {r['happened']:.0%} | {ln:.0%} | {int(r['count'])} |")
    lines.append("")

    shares, shares_ln = pit_shares(df), pit_shares(df, "pit_ln")
    lines += ["**2. Where did prices land inside the forecast?** (each tenth should get about 10%)", "",
              "| Part of forecast | Mixture | Bell curve |", "|---|---|---|"]
    lines += [f"| {k} | {v:.0%} | {shares_ln[k]:.0%} |" for k, v in shares.items()]
    inside = ((df["pit"] >= 0.1) & (df["pit"] <= 0.9)).mean()
    inside_ln = ((df["pit_ln"] >= 0.1) & (df["pit_ln"] <= 0.9)).mean()
    lines += ["", f"The 10-90% range held the price {inside:.0%} of the time (bell curve {inside_ln:.0%}; should be 80%). "
              f"Below the 10% level: {(df['pit'] < 0.1).mean():.0%}, above the 90% level: {(df['pit'] > 0.9).mean():.0%} "
              "(should be 10% each).", ""]

    c = logscore_comparison(df)
    verdict = ("clearly better" if c["t"] > 2 else "clearly worse" if c["t"] < -2 else "not clearly different")
    lines += ["**3. Mixture vs bell curve** (log score, higher is better)", "",
              f"The mixture scored better in {c['mixture_wins']} of {c['expiries']} expiries, by "
              f"{c['mean_gain']:+.3f} on average (t = {c['t']:.1f}): {verdict}."
              + (f" {c['dropped']} forecasts gave the outcome zero density and were left out." if c["dropped"] else ""), ""]
    return lines


def report(df: pd.DataFrame) -> str:
    lines = ["# Implied distribution backtest", ""]
    for track, t in df.groupby("track"):
        lines += [f"## {track}", ""] + section(t, "All symbols")
        groups = {"ETFs": t["symbol"].isin(ETFS), "Stocks": ~t["symbol"].isin(ETFS)}
        if "earnings_in_window" in t:
            groups["Stocks with earnings in the window"] = groups["Stocks"] & (t["earnings_in_window"] == True)  # noqa: E712
            groups["Stocks without earnings"] = groups["Stocks"] & (t["earnings_in_window"] == False)  # noqa: E712
        for name, mask in groups.items():
            if mask.any() and mask.sum() < len(t):
                lines += section(t[mask], name)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score backtest forecasts against realised prices.")
    parser.add_argument("--data", type=Path, help="Local snapshot folder (default: the Hugging Face dataset).")
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO") or DEFAULT_REPO_ID)
    parser.add_argument("--out", type=Path, help="Also save the report as Markdown, and the scored rows as CSV next to it.")
    args = parser.parse_args(argv)

    archive = Archive(local=args.data) if args.data else Archive(repo_id=args.repo_id, token=os.environ.get("HF_TOKEN"))
    df = load(archive)
    if df.empty:
        print("No forecasts with outcomes yet.")
        return 0
    df = add_scores(df)
    text = report(df)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        df.to_csv(args.out.with_suffix(".csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
