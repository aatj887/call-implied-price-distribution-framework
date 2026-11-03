from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.forecasts import TRACKS, expiry_fridays, fit_symbol, mixture_cdf, mixture_quantile, pick_forecast_date
from src.modelling.minimization import call_price_model, put_price_model


def test_monthly_expiries_are_third_fridays():
    assert expiry_fridays(date(2026, 10, 1), date(2026, 12, 31), monthly=True) == [
        date(2026, 10, 16), date(2026, 11, 20), date(2026, 12, 18)]


def test_weekly_expiries_are_every_friday():
    assert expiry_fridays(date(2026, 10, 3), date(2026, 10, 23), monthly=False) == [
        date(2026, 10, 9), date(2026, 10, 16), date(2026, 10, 23)]


def test_forecast_date_is_last_snapshot_at_least_target_days_before():
    track = TRACKS["monthly30"]
    snapshots = [date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21), date(2026, 10, 22)]
    # Nov 20 - 30 days = Oct 21: Oct 22 is too late
    assert pick_forecast_date(date(2026, 11, 20), snapshots, track) == date(2026, 10, 21)


def test_forecast_date_outside_tolerance_is_skipped():
    track = TRACKS["monthly30"]  # 30 + 5 days at most
    assert pick_forecast_date(date(2026, 11, 20), [date(2026, 10, 14)], track) is None
    assert pick_forecast_date(date(2026, 11, 20), [date(2026, 10, 16)], track) == date(2026, 10, 16)


def test_quantile_inverts_cdf():
    theta = (6.60, 0.05, 6.65, 0.02, 0.3)
    for p in (0.1, 0.5, 0.9):
        assert mixture_cdf(mixture_quantile(p, *theta), *theta) == pytest.approx(p, abs=1e-8)


def synthetic_chain(S0=100.0, r=0.04, dte=30, sigma=0.25):
    """
    Exact Black-Scholes quotes (one lognormal) with a 2-cent spread.
    """
    tau = dte / 365
    b = sigma * np.sqrt(tau)
    a = np.log(S0) + r*tau - 0.5*b*b
    strikes = np.arange(70, 131, 2.5)
    rows = []
    for kind, pricer in (("call", call_price_model), ("put", lambda X, *p: put_price_model(X, *p, S0))):
        mid = pricer(strikes, r, tau, a, b, a, b, 1.0)
        rows += [{"underlying_symbol": "TEST", "expiration": date(2026, 11, 20), "dte": dte, "option_type": kind,
                  "strike": k, "bid": m - 0.01, "ask": m + 0.01, "underlying_price": S0} for k, m in zip(strikes, mid)]
    return pd.DataFrame(rows), a, b


def test_fit_symbol_recovers_a_lognormal():
    chain, a, b = synthetic_chain()
    chain = chain[chain["bid"] > 0]
    row = fit_symbol("TEST", chain, date(2026, 11, 20), date(2026, 10, 21), 0.04)
    assert row["status"] == "ok"
    assert row["ln_a"] == pytest.approx(a, abs=1e-3)
    assert row["ln_b"] == pytest.approx(b, rel=1e-2)
    true_p_up = 1 - mixture_cdf(100.0, a, b, a, b, 1.0)
    assert row["p_up"] == pytest.approx(true_p_up, abs=0.01)
    assert row["rmse"] < 1e-3 and row["rmse_ln"] < 1e-3  # Exact quotes: both fit to optimiser tolerance


def test_holiday_friday_falls_back_to_thursday():
    chain, _, _ = synthetic_chain(dte=29)
    chain["expiration"] = date(2026, 11, 19)
    row = fit_symbol("TEST", chain[chain["bid"] > 0], date(2026, 11, 20), date(2026, 10, 21), 0.04)
    assert row["expiration"] == date(2026, 11, 19) and row["dte"] == 29 and row["status"] == "ok"


def test_thin_chains_are_flagged_not_fitted():
    chain, _, _ = synthetic_chain()
    row = fit_symbol("TEST", chain[chain["bid"] > 0].head(6), date(2026, 11, 20), date(2026, 10, 21), 0.04)
    assert row["status"] == "too_few_strikes"
    assert "a1" not in row
