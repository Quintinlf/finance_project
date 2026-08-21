"""Regression test for the 2026-08-13 KOLD bracket rejection.

The order was rejected by Alpaca with:

    {"base_price":"29.805","code":42210000,
     "message":"take_profit.limit_price must be >= base_price + 0.01"}

Cause: signals are priced off yfinance daily bars, but Alpaca validates bracket
legs against its own live price. On a fast-moving instrument the 4% take-profit
computed from a stale bar can land *below* the broker's base price, and the
whole order is lost rather than merely mispriced.
"""

import unittest
from unittest.mock import patch

from logic.broker_client import AlpacaBrokerClient


class TestBracketPriceRepair(unittest.TestCase):
    def _client(self):
        return AlpacaBrokerClient(trading_client=object(), paper=True)

    def test_take_profit_below_live_price_is_lifted_above_it(self):
        """The exact KOLD case: TP computed off a stale bar, under base_price."""
        client = self._client()
        with patch.object(AlpacaBrokerClient, "_latest_price", return_value=29.805):
            tp, sl, note = client._repair_bracket_prices(
                symbol="KOLD", side="buy", take_profit_price=29.64, stop_loss_price=27.93
            )

        self.assertGreaterEqual(tp, 29.805 + 0.01)
        self.assertIsNotNone(note)
        self.assertIn("re-anchored", note)

    def test_stop_loss_above_live_price_is_pushed_below_it(self):
        client = self._client()
        with patch.object(AlpacaBrokerClient, "_latest_price", return_value=29.805):
            tp, sl, note = client._repair_bracket_prices(
                symbol="KOLD", side="buy", take_profit_price=31.50, stop_loss_price=30.20
            )

        self.assertLessEqual(sl, 29.805 - 0.01)

    def test_valid_bracket_is_left_untouched(self):
        client = self._client()
        with patch.object(AlpacaBrokerClient, "_latest_price", return_value=29.805):
            tp, sl, note = client._repair_bracket_prices(
                symbol="KOLD", side="buy", take_profit_price=31.00, stop_loss_price=29.21
            )

        self.assertEqual((tp, sl), (31.00, 29.21))
        self.assertIsNone(note)

    def test_intended_distance_is_preserved_when_repairing(self):
        """Re-anchoring should keep the strategy's risk/reward, not flatten it."""
        client = self._client()
        with patch.object(AlpacaBrokerClient, "_latest_price", return_value=30.00):
            tp, _, _ = client._repair_bracket_prices(
                symbol="KOLD", side="buy", take_profit_price=28.80, stop_loss_price=27.00
            )
        # Asked for $1.20 of distance from its own anchor; keep that distance.
        self.assertAlmostEqual(tp, 31.20, places=2)

    def test_unavailable_quote_leaves_prices_alone(self):
        """No live price is a reason to pass through, never to invent one."""
        client = self._client()
        with patch.object(AlpacaBrokerClient, "_latest_price", return_value=None):
            tp, sl, note = client._repair_bracket_prices(
                symbol="KOLD", side="buy", take_profit_price=29.64, stop_loss_price=27.93
            )

        self.assertEqual((tp, sl), (29.64, 27.93))
        self.assertIsNone(note)

    def test_short_side_repairs_in_the_mirror_direction(self):
        client = self._client()
        with patch.object(AlpacaBrokerClient, "_latest_price", return_value=30.00):
            tp, sl, _ = client._repair_bracket_prices(
                symbol="KOLD", side="sell", take_profit_price=30.50, stop_loss_price=29.50
            )

        self.assertLessEqual(tp, 30.00 - 0.01)
        self.assertGreaterEqual(sl, 30.00 + 0.01)


if __name__ == "__main__":
    unittest.main()
