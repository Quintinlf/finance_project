"""The silent in-memory fallback must never masquerade as a real broker again.

Missing credentials used to produce a PaperBrokerClient seeded with $100,000,
with no log line and no exception. A scheduled CI job ran green for months on
that fake book while the Alpaca account saw nothing.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from logic.broker_client import (
    AlpacaBrokerClient,
    BrokerConnectionError,
    PaperBrokerClient,
    create_broker_client,
)


class TestSimulationMode(unittest.TestCase):
    def test_simulation_always_returns_the_in_memory_broker(self):
        client = create_broker_client(execution_mode="simulation", initial_cash=500.0)
        self.assertIsInstance(client, PaperBrokerClient)
        self.assertEqual(client.get_account_summary()["cash"], 500.0)


class TestStrictMode(unittest.TestCase):
    def _fail_creds(self):
        return patch(
            "logic.alpaca_exercises.load_alpaca_creds",
            side_effect=ValueError("Missing Alpaca credentials"),
        )

    def test_strict_raises_instead_of_faking(self):
        with self._fail_creds():
            with self.assertRaises(BrokerConnectionError) as ctx:
                create_broker_client(execution_mode="paper", strict=True)
        self.assertIn("Missing Alpaca credentials", str(ctx.exception))
        self.assertIn("repository secrets", str(ctx.exception).lower())

    def test_non_strict_falls_back_but_says_so_loudly(self):
        with self._fail_creds():
            with self.assertLogs("root", level="ERROR") as logs:
                client = create_broker_client(execution_mode="paper", strict=False)
        self.assertIsInstance(client, PaperBrokerClient)
        combined = "\n".join(logs.output)
        self.assertIn("BROKER CONNECTION FAILED", combined)
        self.assertIn("NOTHING FROM THIS RUN WILL REACH ALPACA", combined)

    def test_ci_defaults_to_strict(self):
        # This is the case that burned months of scheduled runs.
        with self._fail_creds():
            with patch.dict("os.environ", {"CI": "true", "STRICT_BROKER": ""}, clear=False):
                with self.assertRaises(BrokerConnectionError):
                    create_broker_client(execution_mode="paper")

    def test_local_defaults_to_lenient(self):
        with self._fail_creds():
            with patch.dict("os.environ", {"CI": "", "STRICT_BROKER": ""}, clear=False):
                with self.assertLogs("root", level="ERROR"):
                    client = create_broker_client(execution_mode="paper")
        self.assertIsInstance(client, PaperBrokerClient)

    def test_strict_broker_env_overrides_ci_detection(self):
        with self._fail_creds():
            with patch.dict("os.environ", {"CI": "true", "STRICT_BROKER": "false"}, clear=False):
                with self.assertLogs("root", level="ERROR"):
                    client = create_broker_client(execution_mode="paper")
        self.assertIsInstance(client, PaperBrokerClient)

        with self._fail_creds():
            with patch.dict("os.environ", {"CI": "", "STRICT_BROKER": "true"}, clear=False):
                with self.assertRaises(BrokerConnectionError):
                    create_broker_client(execution_mode="paper")


class TestCredentialsAreExercised(unittest.TestCase):
    def test_connection_is_verified_with_a_real_api_call(self):
        # Constructing a TradingClient does no I/O, so bad credentials would
        # otherwise stay hidden until the first order of the day.
        with patch("logic.alpaca_exercises.load_alpaca_creds", return_value=object()):
            with patch("logic.alpaca_exercises.connect_trading_client") as connect:
                with patch.object(
                    AlpacaBrokerClient,
                    "get_account_summary",
                    side_effect=RuntimeError("401 unauthorized"),
                ):
                    with self.assertRaises(BrokerConnectionError) as ctx:
                        create_broker_client(execution_mode="paper", strict=True)
        self.assertIn("401 unauthorized", str(ctx.exception))
        connect.assert_called_once()

    def test_healthy_connection_returns_the_alpaca_client(self):
        with patch("logic.alpaca_exercises.load_alpaca_creds", return_value=object()):
            with patch("logic.alpaca_exercises.connect_trading_client"):
                with patch.object(
                    AlpacaBrokerClient, "get_account_summary", return_value={"cash": 490.48}
                ):
                    client = create_broker_client(execution_mode="paper", strict=True)
        self.assertIsInstance(client, AlpacaBrokerClient)


if __name__ == "__main__":
    unittest.main(verbosity=2)
