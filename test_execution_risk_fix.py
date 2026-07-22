"""Tests for unified equity resolution and risk enforcement consistency."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from logic.account_equity import resolve_account_equity
from logic.data_structures import ExecutionConfig, OrderPlan, PositionState
from logic.execution_engine import build_order_plan, enforce_risk_limits
from logic.risk_config import PortfolioRiskConfig


class _StubBroker:
    def __init__(self, summary: dict) -> None:
        self._summary = summary

    def get_account_summary(self) -> dict:
        return dict(self._summary)


class AccountEquityTests(unittest.TestCase):
    def test_prefers_live_broker_equity_over_stale_snapshot(self) -> None:
        broker = _StubBroker(
            {
                "cash": 50000.0,
                "portfolio_value": 100000.0,
                "account_id": "paper-default",
            }
        )
        ctx = resolve_account_equity(
            broker_client=broker,  # type: ignore[arg-type]
            db_path=None,
            fallback_cash=470.0,
        )
        self.assertEqual(ctx.equity_source, "broker_portfolio_value")
        self.assertEqual(ctx.equity, 100000.0)


class RiskEnforcementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.risk_cfg = PortfolioRiskConfig(max_position_size=0.20)
        self.position = PositionState(
            symbol="NVDA",
            quantity=0,
            avg_entry_price=0.0,
            side="flat",
            source="paper",
        )

    def test_allows_one_share_with_live_equity(self) -> None:
        plan = OrderPlan(
            symbol="NVDA",
            side="buy",
            quantity=1,
            entry_type="market",
            limit_price=None,
            tp_price=200.0,
            sl_price=180.0,
            time_in_force="day",
            reason="test",
        )
        allowed, updated, reason, outcome = enforce_risk_limits(
            order_plan=plan,
            action="buy",
            position_state=self.position,
            risk_cfg=self.risk_cfg,
            account_cash=50000.0,
            trades_today=0,
            daily_return=0.0,
            exposure=0.1,
            db_path="trade_logs/trading.db",
            current_price=192.53,
            account_equity=100000.0,
        )
        self.assertTrue(allowed)
        self.assertEqual(outcome, "execute")
        self.assertIsNone(reason)
        self.assertEqual(updated.quantity, 1)

    def test_skips_instead_of_rejecting_when_below_one_share(self) -> None:
        plan = OrderPlan(
            symbol="GOOGL",
            side="buy",
            quantity=1,
            entry_type="market",
            limit_price=None,
            tp_price=350.0,
            sl_price=320.0,
            time_in_force="day",
            reason="test",
        )
        allowed, updated, reason, outcome = enforce_risk_limits(
            order_plan=plan,
            action="buy",
            position_state=PositionState(
                symbol="GOOGL",
                quantity=0,
                avg_entry_price=0.0,
                side="flat",
                source="paper",
            ),
            risk_cfg=self.risk_cfg,
            account_cash=25.5,
            trades_today=0,
            daily_return=None,
            exposure=0.0,
            db_path="trade_logs/trading.db",
            current_price=337.39,
            account_equity=470.0,
        )
        self.assertFalse(allowed)
        self.assertEqual(outcome, "skip")
        self.assertIn("position_below_minimum", reason or "")

    def test_planner_respects_cap_before_enforcement(self) -> None:
        config = ExecutionConfig(execution_mode="paper", max_position_pct_of_equity=20.0)
        plan = build_order_plan(
            signal=MagicMock(
                symbol="NVDA",
                signal_type="buy",
                confidence=0.9,
                meta={},
            ),
            position_state=self.position,
            config=config,
            account_cash=25.5,
            current_price=192.53,
            account_equity=100000.0,
            max_position_fraction=0.20,
        )
        self.assertGreaterEqual(plan.quantity, 1)


if __name__ == "__main__":
    unittest.main()
