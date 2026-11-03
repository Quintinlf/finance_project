"""Paper sizing may buy a fraction of a share. Whole-share mode stays whole."""

import unittest

from logic.data_structures import ExecutionConfig, PositionState, Signal
from logic.execution_engine import _size_quantity, build_order_plan


def _signal():
    return Signal(symbol="GLD", signal_type="buy", confidence=0.8, prob_profit=0.7, meta={})


def _flat():
    return PositionState(symbol="GLD", quantity=0, avg_entry_price=0, side="flat", source="paper")


class TestFractionalPaperSizing(unittest.TestCase):
    def test_whole_shares_stay_zero_when_one_share_does_not_fit(self):
        qty = _size_quantity(0.4, max_shares=0.4, price=200.0, fractional=False)
        self.assertEqual(qty, 0)

    def test_paper_fraction_clears_a_dollar_notional(self):
        qty = _size_quantity(0.4, max_shares=0.4, price=200.0, fractional=True)
        self.assertGreater(qty, 0)
        self.assertLess(qty, 1)
        self.assertGreaterEqual(qty * 200.0, 1.0)

    def test_paper_order_plan_uses_a_fraction_under_the_cap(self):
        config = ExecutionConfig(
            execution_mode="paper",
            dry_run=True,
            fractional_shares=True,
            base_risk_pct=100.0,
            max_position_pct_of_equity=20.0,
        )
        plan = build_order_plan(
            _signal(),
            _flat(),
            config,
            account_cash=500.0,
            current_price=200.0,
            account_equity=500.0,
        )
        self.assertGreater(plan.quantity, 0)
        self.assertLess(plan.quantity, 1)
        self.assertLessEqual(plan.quantity * 200.0, 500.0 * 0.20 + 1e-6)

    def test_closing_a_fractional_position_sells_all_of_it(self):
        config = ExecutionConfig(execution_mode="paper", dry_run=True, fractional_shares=True)
        held = PositionState(
            symbol="GLD", quantity=0.25, avg_entry_price=200.0, side="long", source="paper"
        )
        signal = Signal(symbol="GLD", signal_type="sell", confidence=0.8, prob_profit=0.3, meta={})
        plan = build_order_plan(
            signal, held, config, account_cash=500.0, current_price=200.0, account_equity=500.0
        )
        self.assertAlmostEqual(plan.quantity, 0.25)


if __name__ == "__main__":
    unittest.main()
