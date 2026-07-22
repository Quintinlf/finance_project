"""Tests for the position exit reconciler (Mar-24 unmanaged-position guardrail)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List

from logic.position_reconciler import (
    _has_protective_sell,
    reconcile_position_exits,
)


class _Position(SimpleNamespace):
    pass


class _Order(SimpleNamespace):
    pass


class _FakeTradingClient:
    """Minimal stand-in for an Alpaca TradingClient."""

    def __init__(self, positions: List[Any], open_orders: List[Any]) -> None:
        self._positions = positions
        self._open_orders = open_orders
        self.submitted: List[Any] = []

    def get_all_positions(self):
        return self._positions

    def get_orders(self, filter=None):  # noqa: A002 - mirrors alpaca-py signature
        return self._open_orders

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id=f"ORDER-{len(self.submitted)}")


class _FakeBroker:
    def __init__(self, trading_client) -> None:
        self._trading_client = trading_client


class HasProtectiveSellTests(unittest.TestCase):
    def test_detects_open_sell_limit(self) -> None:
        orders = [_Order(side="sell", type="limit", order_class="simple")]
        self.assertTrue(_has_protective_sell(orders))

    def test_detects_oco_order(self) -> None:
        orders = [_Order(side="sell", type="market", order_class="oco")]
        self.assertTrue(_has_protective_sell(orders))

    def test_ignores_buy_orders(self) -> None:
        orders = [_Order(side="buy", type="limit", order_class="simple")]
        self.assertFalse(_has_protective_sell(orders))

    def test_empty_is_unprotected(self) -> None:
        self.assertFalse(_has_protective_sell([]))


class ReconcileTests(unittest.TestCase):
    def test_dry_run_stop_only_anchored_to_current_price(self) -> None:
        positions = [_Position(symbol="BAC", qty="2", avg_entry_price="47.38", current_price="61.25")]
        client = _FakeTradingClient(positions=positions, open_orders=[])
        broker = _FakeBroker(client)

        results = reconcile_position_exits(
            broker_client=broker, tp_pct=0.04, sl_pct=0.02, dry_run=True, verbose=False
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].action, "dry_run")
        # Default is stop-only: SL anchored to CURRENT price 61.25, no TP that
        # would cap the upside or force-sell the winner.
        self.assertIn("60.02", results[0].detail)   # 61.25 * 0.98
        self.assertNotIn("63.7", results[0].detail)  # no take-profit leg
        self.assertEqual(client.submitted, [])       # nothing submitted in dry-run

    def test_dry_run_oco_includes_take_profit(self) -> None:
        positions = [_Position(symbol="BAC", qty="2", avg_entry_price="47.38", current_price="61.25")]
        client = _FakeTradingClient(positions=positions, open_orders=[])
        broker = _FakeBroker(client)

        results = reconcile_position_exits(
            broker_client=broker, tp_pct=0.04, sl_pct=0.02, dry_run=True, verbose=False,
            exit_style="oco",
        )

        self.assertEqual(results[0].action, "dry_run")
        self.assertIn("63.7", results[0].detail)    # 61.25 * 1.04 take-profit
        self.assertIn("60.02", results[0].detail)   # 61.25 * 0.98 stop-loss

    def test_live_submits_exit_when_unprotected(self) -> None:
        positions = [_Position(symbol="TSLA", qty="1", avg_entry_price="379.74", current_price="379.08")]
        client = _FakeTradingClient(positions=positions, open_orders=[])
        broker = _FakeBroker(client)

        results = reconcile_position_exits(
            broker_client=broker, tp_pct=0.04, sl_pct=0.02, dry_run=False, verbose=False
        )

        self.assertEqual(results[0].action, "attached")
        self.assertEqual(len(client.submitted), 1)

    def test_skips_when_already_protected(self) -> None:
        positions = [_Position(symbol="BAC", qty="2", avg_entry_price="47.38", current_price="61.25")]
        open_orders = [_Order(symbol="BAC", side="sell", type="limit", order_class="oco")]
        client = _FakeTradingClient(positions=positions, open_orders=open_orders)
        broker = _FakeBroker(client)

        results = reconcile_position_exits(
            broker_client=broker, tp_pct=0.04, sl_pct=0.02, dry_run=False, verbose=False
        )

        self.assertEqual(results[0].action, "already_protected")
        self.assertEqual(client.submitted, [])

    def test_skips_short_positions(self) -> None:
        positions = [_Position(symbol="XYZ", qty="-5", avg_entry_price="10.0", current_price="9.0")]
        client = _FakeTradingClient(positions=positions, open_orders=[])
        broker = _FakeBroker(client)

        results = reconcile_position_exits(
            broker_client=broker, tp_pct=0.04, sl_pct=0.02, dry_run=False, verbose=False
        )

        self.assertEqual(results[0].action, "skipped")
        self.assertEqual(client.submitted, [])

    def test_noop_without_trading_client(self) -> None:
        broker = SimpleNamespace()  # no _trading_client (e.g. PaperBrokerClient sim)
        results = reconcile_position_exits(
            broker_client=broker, tp_pct=0.04, sl_pct=0.02, dry_run=True, verbose=False
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
