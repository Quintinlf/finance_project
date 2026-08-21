"""Tests for the exposure deadlock fix.

Covers the two halves of the fix:
  * enforce_risk_limits sizes a BUY into the remaining exposure headroom
    instead of rejecting every order that would not fit whole.
  * trim_to_exposure_cap sells an over-cap book back under its limit so the
    strategy can trade again.

Regression target: from 2026-08-14 onward every run rejected every BUY with
"portfolio exposure limit reached" because BAC (opened 2026-03-24, never
harvested) held exposure at 36.6% against a 30% cap.
"""

import unittest
from unittest.mock import MagicMock

from logic.data_structures import OrderPlan, PositionState
from logic.exposure_manager import trim_to_exposure_cap
from logic.execution_engine import enforce_risk_limits
from logic.risk_config import PortfolioRiskConfig


def _flat(symbol="UNG"):
    return PositionState(
        symbol=symbol, quantity=0, avg_entry_price=0.0, side="flat", source="test"
    )


def _plan(symbol="UNG", qty=10):
    return OrderPlan(
        symbol=symbol,
        side="buy",
        quantity=qty,
        entry_type="market",
        limit_price=None,
        tp_price=None,
        sl_price=None,
        time_in_force="day",
        reason="test",
    )


def _enforce(*, exposure, equity=493.09, price=9.86, qty=10, cfg=None):
    return enforce_risk_limits(
        _plan(qty=qty),
        "buy",
        _flat(),
        cfg or PortfolioRiskConfig(),
        account_cash=equity,
        trades_today=0,
        daily_return=0.0,
        exposure=exposure,
        db_path=":memory:",
        current_price=price,
        account_equity=equity,
    )


class TestExposureHeadroomSizing(unittest.TestCase):
    def test_under_cap_with_room_sizes_into_headroom(self):
        """25% exposed against a 30% cap leaves 5% of equity to deploy."""
        allowed, plan, reason, outcome = _enforce(exposure=0.25)

        self.assertTrue(allowed, msg=reason)
        self.assertEqual(outcome, "execute")
        # 5% of $493.09 = $24.65 of room; at $9.86/share that is 2 shares.
        self.assertEqual(plan.quantity, 2)
        self.assertIn("exposure headroom", plan.reason)

    def test_position_cap_binds_when_it_is_the_tighter_limit(self):
        """With plenty of exposure room, max_position_size is what caps the size."""
        allowed, plan, reason, outcome = _enforce(exposure=0.0, qty=100)

        self.assertTrue(allowed, msg=reason)
        # 20% of $493.09 = $98.62 -> 10 shares at $9.86, not the 100 requested.
        self.assertEqual(plan.quantity, 10)
        self.assertIn("max_position_size", plan.reason)

    def test_at_or_over_cap_still_rejects(self):
        """The real August state: no headroom at all means no BUY."""
        allowed, plan, reason, outcome = _enforce(exposure=0.3655)

        self.assertFalse(allowed)
        self.assertEqual(outcome, "reject")
        self.assertIn("exposure limit reached", reason)
        self.assertIn("trim", reason)

    def test_headroom_too_small_for_one_share_skips_with_detail(self):
        """A sliver of room that cannot fund a share is a skip, not a crash."""
        allowed, plan, reason, outcome = _enforce(exposure=0.299, price=58.53)

        self.assertFalse(allowed)
        self.assertEqual(outcome, "skip")
        self.assertIn("position_below_minimum", reason)
        self.assertIn("exposure headroom", reason)


class _FakePosition:
    def __init__(self, symbol, qty, price):
        self.symbol = symbol
        self.qty = str(qty)
        self.current_price = str(price)
        self.market_value = str(qty * price)


class TestTrimToExposureCap(unittest.TestCase):
    def _broker(self, positions):
        trading_client = MagicMock()
        trading_client.get_all_positions.return_value = positions
        trading_client.get_orders.return_value = []
        broker = MagicMock()
        broker._trading_client = trading_client
        broker.place_market_order.return_value = MagicMock(id="order-1")
        return broker, trading_client

    def test_compliant_book_is_left_alone(self):
        broker, _ = self._broker([_FakePosition("BAC", 2, 64.08)])
        results = trim_to_exposure_cap(
            broker_client=broker,
            equity=1000.0,
            exposure=0.10,
            max_exposure=0.30,
            dry_run=False,
        )
        self.assertEqual(results, [])
        broker.place_market_order.assert_not_called()

    def test_over_cap_book_is_trimmed_largest_first(self):
        """The August book: BAC $128 + BNO $52 on $493 equity = 36.6%."""
        broker, _ = self._broker(
            [_FakePosition("BNO", 1, 52.11), _FakePosition("BAC", 2, 64.08)]
        )
        results = trim_to_exposure_cap(
            broker_client=broker,
            equity=493.09,
            exposure=0.3655,
            max_exposure=0.30,
            release_buffer=0.05,
            dry_run=False,
        )

        self.assertTrue(results)
        # Richest position goes first.
        self.assertEqual(results[0].symbol, "BAC")
        self.assertEqual(results[0].action, "trimmed")
        broker.place_market_order.assert_called_once_with(
            symbol="BAC", qty=1, side="sell", time_in_force="day"
        )

    def test_trim_frees_enough_to_clear_the_cap(self):
        broker, _ = self._broker([_FakePosition("BAC", 2, 64.08)])
        results = trim_to_exposure_cap(
            broker_client=broker,
            equity=493.09,
            exposure=0.3655,
            max_exposure=0.30,
            release_buffer=0.05,
            dry_run=False,
        )
        freed = sum(r.qty * 64.08 for r in results)
        new_exposure = (0.3655 * 493.09 - freed) / 493.09
        self.assertLessEqual(new_exposure, 0.30)

    def test_resting_protective_sells_are_cancelled_first(self):
        """Shares backing a resting sell are reserved; the trim would be rejected."""
        broker, trading_client = self._broker([_FakePosition("BAC", 2, 64.08)])
        resting = MagicMock()
        resting.side = "sell"
        resting.id = "resting-order"
        trading_client.get_orders.return_value = [resting]

        trim_to_exposure_cap(
            broker_client=broker,
            equity=493.09,
            exposure=0.3655,
            max_exposure=0.30,
            dry_run=False,
        )
        trading_client.cancel_order_by_id.assert_called_once_with("resting-order")

    def test_dry_run_submits_nothing(self):
        broker, trading_client = self._broker([_FakePosition("BAC", 2, 64.08)])
        results = trim_to_exposure_cap(
            broker_client=broker,
            equity=493.09,
            exposure=0.3655,
            max_exposure=0.30,
            dry_run=True,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].action, "dry_run")
        broker.place_market_order.assert_not_called()
        trading_client.cancel_order_by_id.assert_not_called()

    def test_broker_rejection_is_reported_not_raised(self):
        broker, _ = self._broker([_FakePosition("BAC", 2, 64.08)])
        broker.place_market_order.return_value = None
        broker.last_order_error = "market closed"

        results = trim_to_exposure_cap(
            broker_client=broker,
            equity=493.09,
            exposure=0.3655,
            max_exposure=0.30,
            dry_run=False,
        )
        self.assertEqual(results[0].action, "error")
        self.assertIn("market closed", results[0].detail)

    def test_simulation_broker_is_a_noop(self):
        broker = MagicMock(spec=[])  # no _trading_client attribute
        results = trim_to_exposure_cap(
            broker_client=broker,
            equity=493.09,
            exposure=0.90,
            max_exposure=0.30,
            dry_run=False,
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
