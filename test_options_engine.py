"""Tests for options chain parsing and contract selection.

Three bugs found on 2026-09-08, all of which made the options engine look like
it worked while producing unusable data:

1. The chain was requested with feed=OPRA only. This account has no OPRA
   agreement, so every request failed and the engine fell back to placeholder
   contracts with strike=None.
2. get_option_chain returns a dict keyed by OCC symbol, so iterating it yields
   strings. getattr(str, "strike_price") silently produced strike=None for
   every contract -- a chain that looked populated but priced nothing.
3. Put deltas are negative. Selecting "closest to 0.30" without taking
   magnitudes picked the delta nearest +0.30 on the number line, which for
   puts is the one closest to zero: the furthest OTM, nearly worthless
   contract. A correct -0.30 put scored as the worst possible match.
"""

import unittest
from types import SimpleNamespace

from logic.options_engine import (
    _parse_occ_symbol,
    _select_by_delta,
    _select_by_moneyness,
    filter_expirations_by_dte,
    select_call_contract,
    select_put_contract,
)


def _row(symbol="UNG261023C00011500", strike=11.5, opt_type="call", delta=0.31,
         expiration="2026-10-23", underlying="UNG"):
    return {
        "symbol": symbol, "underlying": underlying, "expiration": expiration,
        "strike": strike, "type": opt_type, "delta": delta,
    }


class TestParseOccSymbol(unittest.TestCase):
    def test_parses_a_real_call_symbol(self):
        p = _parse_occ_symbol("UNG261023C00011500")
        self.assertEqual(p["expiration"], "2026-10-23")
        self.assertEqual(p["strike"], 11.5)
        self.assertEqual(p["type"], "call")

    def test_parses_a_real_put_symbol(self):
        p = _parse_occ_symbol("UNG260916P00017000")
        self.assertEqual(p["expiration"], "2026-09-16")
        self.assertEqual(p["strike"], 17.0)
        self.assertEqual(p["type"], "put")

    def test_handles_fractional_strikes(self):
        self.assertEqual(_parse_occ_symbol("SPY260116C00612500")["strike"], 612.5)

    def test_handles_long_underlying(self):
        p = _parse_occ_symbol("GOOGL261218C00200000")
        self.assertEqual(p["strike"], 200.0)
        self.assertEqual(p["type"], "call")

    def test_rejects_a_non_occ_string(self):
        self.assertIsNone(_parse_occ_symbol("UNG"))
        self.assertIsNone(_parse_occ_symbol("not-a-symbol"))

    def test_rejects_an_impossible_date(self):
        self.assertIsNone(_parse_occ_symbol("UNG261345C00011500"))


class TestDeltaSelection(unittest.TestCase):
    def test_call_selection_picks_nearest_delta(self):
        chain = [
            _row(delta=0.10, strike=14.0),
            _row(delta=0.31, strike=11.5),
            _row(delta=0.72, strike=9.0),
        ]
        self.assertEqual(_select_by_delta(chain, 0.30, "call")["delta"], 0.31)

    def test_put_selection_uses_magnitude_not_sign(self):
        """The bug: -0.06 is numerically closer to +0.30 than -0.31 is."""
        chain = [
            _row(opt_type="put", delta=-0.06, strike=8.0),   # far OTM, near worthless
            _row(opt_type="put", delta=-0.31, strike=10.0),  # the real 0.30-delta put
            _row(opt_type="put", delta=-0.85, strike=14.0),
        ]
        chosen = _select_by_delta(chain, 0.30, "put")
        self.assertEqual(chosen["delta"], -0.31)
        self.assertEqual(chosen["strike"], 10.0)

    def test_put_and_call_land_symmetrically(self):
        chain = [
            _row(opt_type="call", delta=0.30, strike=11.5),
            _row(opt_type="put", delta=-0.30, strike=10.0),
            _row(opt_type="put", delta=-0.02, strike=6.0),
        ]
        call = _select_by_delta(chain, 0.30, "call")
        put = _select_by_delta(chain, 0.30, "put")
        self.assertAlmostEqual(abs(call["delta"]), abs(put["delta"]), places=6)

    def test_rows_without_delta_are_skipped(self):
        chain = [_row(delta=None), _row(delta=0.29, strike=11.0)]
        self.assertEqual(_select_by_delta(chain, 0.30, "call")["delta"], 0.29)

    def test_no_matching_type_returns_none(self):
        self.assertIsNone(_select_by_delta([_row(opt_type="call")], 0.30, "put"))


class TestMoneynessFallback(unittest.TestCase):
    """Used when the delayed feed omits greeks entirely."""

    def test_call_picks_a_strike_above_spot(self):
        chain = [_row(delta=None, strike=s) for s in (9.0, 10.6, 12.5)]
        chosen = _select_by_moneyness(chain, 0.30, "call", spot=10.0)
        self.assertEqual(chosen["strike"], 10.6)  # nearest 10.0 * 1.06
        self.assertIn("moneyness proxy", chosen["selection_basis"])

    def test_put_picks_a_strike_below_spot(self):
        chain = [_row(opt_type="put", delta=None, strike=s) for s in (7.0, 9.4, 11.0)]
        chosen = _select_by_moneyness(chain, 0.30, "put", spot=10.0)
        self.assertEqual(chosen["strike"], 9.4)  # nearest 10.0 * 0.94

    def test_unknown_spot_returns_none_rather_than_guessing(self):
        self.assertIsNone(_select_by_moneyness([_row()], 0.30, "call", spot=0.0))


class TestDteFilter(unittest.TestCase):
    def test_keeps_only_the_target_window(self):
        from datetime import datetime, timedelta

        today = datetime.utcnow().date()
        chain = [
            _row(expiration=(today + timedelta(days=d)).isoformat())
            for d in (5, 30, 90)
        ]
        kept = filter_expirations_by_dte(chain, min_dte=21, max_dte=45)
        self.assertEqual(len(kept), 1)

    def test_unparseable_expiration_is_dropped_not_crashed(self):
        self.assertEqual(filter_expirations_by_dte([_row(expiration="")]), [])


class TestSelectorsEndToEnd(unittest.TestCase):
    def test_selectors_prefer_real_greeks_when_present(self):
        chain = [
            _row(delta=0.31, strike=11.5),
            _row(opt_type="put", delta=-0.30, strike=10.0),
        ]
        self.assertEqual(select_call_contract(chain)["strike"], 11.5)
        self.assertEqual(select_put_contract(chain)["strike"], 10.0)

    def test_empty_chain_returns_a_safe_placeholder(self):
        out = select_call_contract([])
        self.assertEqual(out["type"], "call")
        self.assertIsNone(out["strike"])


if __name__ == "__main__":
    unittest.main()
