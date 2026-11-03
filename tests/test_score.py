from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.forecasts import Archive
from backtest.outcomes import settle
from backtest.score import add_scores, calibration_table, load, logscore_comparison, pit_shares, report


def synthetic_backtest(n_expiries=12, n_symbols=200, seed=0):
    """
    Forecasts whose outcomes really are drawn from the forecast mixture: a perfectly calibrated model.
    The single lognormal ("_ln") matches its mean and variance in log space but misses the shape.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_expiries):
        expiry = date(2027, 1, 15) + timedelta(weeks=4*e)
        for i in range(n_symbols):
            spot = 100.0
            a1, b1, a2, b2, q = np.log(spot) - 0.08, 0.12, np.log(spot) + 0.01, 0.04, 0.2
            first = rng.random() < q
            settle_price = np.exp(rng.normal(a1, b1) if first else rng.normal(a2, b2))
            mean = q*a1 + (1-q)*a2
            sd = np.sqrt(q*(b1**2 + a1**2) + (1-q)*(b2**2 + a2**2) - mean**2)
            rows.append(dict(track="monthly30", symbol=f"S{i}", forecast_date=expiry - timedelta(days=30),
                             expiration=expiry, spot=spot, settle=settle_price, a1=a1, b1=b1, a2=a2, b2=b2, q=q,
                             ln_a=mean, ln_b=sd))
    df = pd.DataFrame(rows)
    from backtest.forecasts import mixture_cdf
    from scipy.stats import norm
    df["p_up"] = 1 - mixture_cdf(df["spot"].to_numpy(), a1, b1, a2, b2, q)
    df["p_up_ln"] = 1 - norm.cdf((np.log(df["spot"]) - df["ln_a"]) / df["ln_b"])
    return df


def test_calibrated_forecasts_score_as_calibrated():
    df = add_scores(synthetic_backtest())
    assert pit_shares(df).to_numpy() == pytest.approx(np.full(10, 0.1), abs=0.015)
    cal = calibration_table(df)
    assert cal["happened"].iloc[0] == pytest.approx(cal["forecast"].iloc[0], abs=0.02)


def test_true_shape_beats_bell_curve():
    c = logscore_comparison(add_scores(synthetic_backtest()))
    assert c["expiries"] == 12
    assert c["mixture_wins"] >= 10 and c["t"] > 2


def test_report_mentions_each_question():
    text = report(add_scores(synthetic_backtest(n_expiries=3, n_symbols=20)))
    assert "Did the chance of ending higher match reality" in text
    assert "Only 3 expiries so far" in text
    assert "Mixture vs bell curve" in text


def test_settle_handles_split_and_missing_prices():
    forecasts = pd.DataFrame({"track": "monthly30", "symbol": ["NVDA", "GONE"], "forecast_date": date(2024, 5, 21),
                              "expiration": date(2024, 6, 21), "spot": [953.86, 10.0]})
    # Yahoo restates history after the 10:1 split, so the forecast-date close shows as 95.386
    closes = {"NVDA": pd.Series([95.386, 126.57], index=pd.to_datetime(["2024-05-21", "2024-06-21"]))}
    out = settle(forecasts, closes).set_index("symbol")
    assert out.loc["NVDA", "settle"] == pytest.approx(1265.7, rel=1e-4)
    assert out.loc["NVDA", "realized_return"] == pytest.approx(0.32691, rel=1e-3)
    assert out.loc["GONE", "status"] == "no_price"


def test_load_keeps_only_fitted_and_settled(tmp_path):
    archive = Archive(local=tmp_path)
    keys = dict(track="monthly30", forecast_date=date(2026, 10, 21), expiration=date(2026, 11, 20))
    archive.write(pd.DataFrame([{**keys, "symbol": "A", "status": "ok", "spot": 10.0},
                                {**keys, "symbol": "B", "status": "too_few_strikes"},
                                {**keys, "symbol": "C", "status": "ok", "spot": 10.0}]),
                  "forecasts/monthly30/2026-11-20.parquet")
    archive.write(pd.DataFrame([{**keys, "symbol": "A", "status": "ok", "settle": 11.0},
                                {**keys, "symbol": "B", "status": "ok", "settle": 11.0},
                                {**keys, "symbol": "C", "status": "no_price", "settle": np.nan}]),
                  "outcomes/monthly30/2026-11-20.parquet")
    assert load(archive)["symbol"].tolist() == ["A"]
