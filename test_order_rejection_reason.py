"""The broker's rejection reason must reach the decision log.

Seventeen buy attempts in Mar-Apr 2026 failed and every one recorded the same
useless string, "Order submission returned None", because APIError was caught
and discarded at the submitter. These tests pin the reason to the record.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from logic.broker_client import AlpacaBrokerClient


class FakeAPIError(Exception):
    """Stands in for alpaca.common.exceptions.APIError."""


class TestSubmitterRaisesWithReason(unittest.TestCase):
    def test_market_order_raises_with_broker_message(self):
        from logic import alpaca_exercises

        client = MagicMock()
        client.submit_order.side_effect = alpaca_exercises.APIError("insufficient buying power")

        with self.assertRaises(alpaca_exercises.OrderSubmissionError) as ctx:
            alpaca_exercises.place_market_order(
                client, symbol="TSLA", qty=1, side="buy"
            )
        self.assertIn("insufficient buying power", ctx.exception.message)
        self.assertEqual(ctx.exception.symbol, "TSLA")
        self.assertEqual(ctx.exception.order_class, "market")

    def test_bracket_order_raises_with_broker_message(self):
        from logic import alpaca_exercises

        client = MagicMock()
        client.submit_order.side_effect = alpaca_exercises.APIError("wash trade detected")

        with self.assertRaises(alpaca_exercises.OrderSubmissionError) as ctx:
            alpaca_exercises.place_bracket_order(
                client, symbol="BAC", qty=2, side="buy",
                take_profit_price=50.0, stop_loss_price=45.0,
            )
        self.assertIn("wash trade", ctx.exception.message)
        self.assertEqual(ctx.exception.order_class, "bracket")

    def test_successful_market_order_still_returns_order(self):
        from logic import alpaca_exercises

        client = MagicMock()
        sentinel = object()
        client.submit_order.return_value = sentinel
        result = alpaca_exercises.place_market_order(client, symbol="BAC", qty=1, side="buy")
        self.assertIs(result, sentinel)


class TestBrokerClientCapturesReason(unittest.TestCase):
    def test_market_rejection_sets_last_order_error(self):
        from logic import alpaca_exercises

        trading_client = MagicMock()
        trading_client.submit_order.side_effect = alpaca_exercises.APIError(
            "insufficient buying power"
        )
        broker = AlpacaBrokerClient(trading_client, paper=True)

        order = broker.place_market_order(symbol="TSLA", qty=1, side="buy")
        self.assertIsNone(order)
        self.assertIn("insufficient buying power", broker.last_order_error)

    def test_bracket_rejection_sets_last_order_error(self):
        from logic import alpaca_exercises

        trading_client = MagicMock()
        trading_client.submit_order.side_effect = alpaca_exercises.APIError("asset not tradable")
        broker = AlpacaBrokerClient(trading_client, paper=True)

        order = broker.place_bracket_order(
            symbol="WEAT", qty=5, side="buy",
            take_profit_price=6.0, stop_loss_price=5.0,
        )
        self.assertIsNone(order)
        self.assertIn("asset not tradable", broker.last_order_error)

    def test_last_error_is_cleared_on_a_successful_submission(self):
        from logic import alpaca_exercises

        trading_client = MagicMock()
        broker = AlpacaBrokerClient(trading_client, paper=True)

        trading_client.submit_order.side_effect = alpaca_exercises.APIError("nope")
        broker.place_market_order(symbol="BAC", qty=1, side="buy")
        self.assertIsNotNone(broker.last_order_error)

        # A later success must not leave the stale failure hanging around.
        good = MagicMock()
        good.id = "order-123"
        good.symbol = "BAC"
        good.qty = "1"
        good.status = "accepted"
        trading_client.submit_order.side_effect = None
        trading_client.submit_order.return_value = good

        order = broker.place_market_order(symbol="BAC", qty=1, side="buy")
        self.assertIsNotNone(order)
        self.assertIsNone(broker.last_order_error)


class TestReasonReachesDecisionRecord(unittest.TestCase):
    def test_execution_result_carries_broker_reason(self):
        from logic.data_structures import ExecutionConfig, OrderPlan
        from logic.execution_engine import execute_order_plan

        broker = MagicMock()
        broker.place_bracket_order.return_value = None
        broker.last_order_error = "insufficient buying power"

        plan = OrderPlan(
            symbol="TSLA", side="buy", quantity=1, entry_type="market",
            tp_price=400.0, sl_price=370.0,
        )
        config = ExecutionConfig(execution_mode="paper", dry_run=False)

        result = execute_order_plan(
            order_plan=plan, config=config, broker_client=broker, sim_portfolio={}
        )
        self.assertFalse(result["executed"])
        self.assertIn("insufficient buying power", result["error_message"])

    def test_missing_reason_falls_back_to_explicit_message(self):
        from logic.data_structures import ExecutionConfig, OrderPlan
        from logic.execution_engine import execute_order_plan

        broker = MagicMock()
        broker.place_bracket_order.return_value = None
        broker.last_order_error = None

        plan = OrderPlan(
            symbol="TSLA", side="buy", quantity=1, entry_type="market",
            tp_price=400.0, sl_price=370.0,
        )
        config = ExecutionConfig(execution_mode="paper", dry_run=False)

        result = execute_order_plan(
            order_plan=plan, config=config, broker_client=broker, sim_portfolio={}
        )
        self.assertFalse(result["executed"])
        self.assertIn("no reason reported", result["error_message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
