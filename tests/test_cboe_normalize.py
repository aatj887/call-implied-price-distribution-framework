import json
from datetime import date
from pathlib import Path

import pytest

from src.data_sources.cboe import normalize_chain, quote_date, to_cboe_symbol

FIXTURE = Path(__file__).parent / "fixtures" / "cboe_spy_small.json"


@pytest.fixture
def raw():
    return json.loads(FIXTURE.read_text())


def test_to_cboe_symbol():
    assert to_cboe_symbol("spy") == "SPY"
    assert to_cboe_symbol("^SPX") == "_SPX"
    assert to_cboe_symbol("vix") == "_VIX"


def test_quote_date_is_last_session(raw):
    assert quote_date(raw) == date(2026, 10, 2)


def test_schema_matches_model_inputs(raw):
    chain = normalize_chain(raw)
    for col in ["strike", "bid", "ask", "option_type", "dte", "underlying_price", "expiration"]:
        assert col in chain.columns
    assert set(chain["option_type"]) == {"call", "put"}
    assert chain["underlying_price"].iloc[0] == pytest.approx(769.64)


def test_occ_symbol_parsing_and_dte(raw):
    chain = normalize_chain(raw).set_index("contract_symbol")
    row = chain.loc["SPY261016P00770000"]
    assert row["strike"] == 770.0
    assert row["option_type"] == "put"
    assert row["expiration"] == date(2026, 10, 16)
    assert row["dte"] == 14
    assert chain.loc["SPY261002C00505000", "dte"] == 0


def test_expired_contracts_dropped(raw):
    chain = normalize_chain(raw, asof=date(2026, 10, 5))
    assert (chain["dte"] >= 0).all()
    assert len(chain) == 4
