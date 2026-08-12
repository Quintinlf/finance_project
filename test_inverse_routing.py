"""Tests for routing bearish signals into long inverse-fund positions."""

from __future__ import annotations

import unittest

from logic.data_structures import ExecutionConfig, PositionState, Signal
from logic.execution_engine import resolve_exit_pcts
from logic.inverse_routing import format_routing_table, route_signals
from logic.universe import (
    INVERSE_PROXIES,
    NO_INVERSE_COVERAGE,
    coverage_report,
    inverse_for,
    inverse_symbols,
    resolve_inverse,
    scale_exits_for_leverage,
)


def fake_prices(symbols):
    """Stand-in price lookup so tests never touch the network."""
    return {s: 25.0 for s in symbols}


def no_prices(symbols):
    return {s: None for s in symbols}


def make_signal(symbol: str, signal_type: str, confidence=0.8, prob_profit=0.2) -> Signal:
    return Signal(
        symbol=symbol,
        signal_type=signal_type,
        confidence=confidence,
        prob_profit=prob_profit,
        meta={},
    )


def position(symbol: str, quantity: float) -> PositionState:
    side = "long" if quantity > 0 else "short" if quantity < 0 else "flat"
    return PositionState(
        symbol=symbol, quantity=quantity,
        avg_entry_price=10.0 if quantity else 0.0,
        side=side, source="paper",
    )


class TestProxyTable(unittest.TestCase):
    def test_every_proxy_declares_its_underlying_and_leverage(self):
        for underlying, proxy in INVERSE_PROXIES.items():
            self.assertEqual(proxy.underlying_symbol, underlying)
            self.assertGreater(proxy.leverage, 0)
            self.assertIn(proxy.instrument, {"etf", "etn"})

    def test_agriculture_is_declared_uncovered_not_silently_missing(self):
        # The default scope is agriculture and none of it is routable. That gap
        # must be explicit, or a bearish ag day looks like a bug.
        for symbol in ("WEAT", "CORN", "SOYB", "CANE", "DBA"):
            self.assertIsNone(inverse_for(symbol))
            self.assertIn(symbol, NO_INVERSE_COVERAGE)

    def test_no_symbol_is_both_mapped_and_declared_uncovered(self):
        overlap = set(INVERSE_PROXIES) & set(NO_INVERSE_COVERAGE)
        self.assertEqual(overlap, set())

    def test_inverse_symbols_are_deduplicated(self):
        symbols = inverse_symbols("energy")
        self.assertEqual(len(symbols), len(set(symbols)))
        # USO, BNO and UGA all proxy through SCO.
        self.assertIn("SCO", symbols)

    def test_coverage_report_names_both_outcomes(self):
        report = coverage_report("all")
        self.assertIn("NO COVERAGE", report)
        self.assertIn("SCO", report)


class TestLeverageScaling(unittest.TestCase):
    def test_double_leverage_halves_the_bands(self):
        tp, sl = scale_exits_for_leverage(0.04, 0.02, 2.0)
        self.assertAlmostEqual(tp, 0.02)
        self.assertAlmostEqual(sl, 0.01)

    def test_unit_leverage_is_a_no_op(self):
        tp, sl = scale_exits_for_leverage(0.04, 0.02, 1.0)
        self.assertAlmostEqual(tp, 0.04)
        self.assertAlmostEqual(sl, 0.02)

    def test_resolve_exit_pcts_applies_the_divisor(self):
        signal = make_signal("GLL", "buy")
        signal.meta["exit_leverage_divisor"] = 2.0
        config = ExecutionConfig(execution_mode="simulation", tp_pct=0.04, sl_pct=0.02)
        tp, sl = resolve_exit_pcts(signal, config)
        self.assertAlmostEqual(tp, 0.02)
        self.assertAlmostEqual(sl, 0.01)
        self.assertEqual(signal.meta["exit_leverage_applied"]["divisor"], 2.0)

    def test_signals_without_a_divisor_are_untouched(self):
        signal = make_signal("WEAT", "buy")
        config = ExecutionConfig(execution_mode="simulation", tp_pct=0.04, sl_pct=0.02)
        tp, sl = resolve_exit_pcts(signal, config)
        self.assertAlmostEqual(tp, 0.04)
        self.assertAlmostEqual(sl, 0.02)
        self.assertNotIn("exit_leverage_applied", signal.meta)


class TestResolveInverse(unittest.TestCase):
    def test_unpriceable_proxy_is_rejected(self):
        proxy, reason = resolve_inverse("SLV", price_lookup=no_prices)
        self.assertIsNone(proxy)
        self.assertIn("no current price", reason)

    def test_uncovered_symbol_reports_its_reason(self):
        proxy, reason = resolve_inverse("WEAT", price_lookup=fake_prices)
        self.assertIsNone(proxy)
        self.assertIn("wheat", reason)

    def test_lookup_failure_is_caught(self):
        def boom(symbols):
            raise RuntimeError("network down")

        proxy, reason = resolve_inverse("SLV", price_lookup=boom)
        self.assertIsNone(proxy)
        self.assertIn("network down", reason)


class TestRouting(unittest.TestCase):
    def _route(self, signals, states=None):
        return route_signals(
            signals, states or {}, price_lookup=fake_prices, verbose=False
        )

    def test_sell_while_flat_becomes_a_buy_of_the_proxy(self):
        result = self._route([make_signal("SLV", "sell", prob_profit=0.0005)])
        self.assertEqual(len(result.signals), 1)
        routed = result.signals[0]
        self.assertEqual(routed.symbol, "ZSL")
        self.assertEqual(routed.signal_type, "buy")
        # P(proxy up) is P(underlying down).
        self.assertAlmostEqual(routed.prob_profit, 0.9995)
        self.assertEqual(routed.meta["inverse_routing"]["routed_from"], "SLV")
        self.assertEqual(routed.meta["exit_leverage_divisor"], 2.0)

    def test_sell_while_holding_the_underlying_is_left_alone(self):
        result = self._route(
            [make_signal("SLV", "sell")], {"SLV": position("SLV", 3)}
        )
        self.assertEqual(result.signals[0].symbol, "SLV")
        self.assertEqual(result.signals[0].signal_type, "sell")
        self.assertEqual(result.outcomes[0].action, "unchanged")

    def test_buy_while_holding_the_proxy_closes_the_proxy(self):
        result = self._route(
            [make_signal("SLV", "buy", prob_profit=0.9)], {"ZSL": position("ZSL", 4)}
        )
        routed = result.signals[0]
        self.assertEqual(routed.symbol, "ZSL")
        self.assertEqual(routed.signal_type, "sell")
        self.assertEqual(result.outcomes[0].action, "routed_exit")

    def test_plain_buy_while_flat_passes_through(self):
        result = self._route([make_signal("SLV", "buy", prob_profit=0.9)])
        self.assertEqual(result.signals[0].symbol, "SLV")
        self.assertEqual(result.signals[0].signal_type, "buy")

    def test_holds_pass_through_untouched(self):
        signal = make_signal("WEAT", "hold", prob_profit=0.5)
        result = self._route([signal])
        self.assertIs(result.signals[0], signal)
        self.assertEqual(result.outcomes, [])

    def test_uncovered_sell_is_dropped_but_reported(self):
        result = self._route([make_signal("WEAT", "sell")])
        self.assertEqual(result.signals, [])
        self.assertEqual(len(result.outcomes), 1)
        self.assertEqual(result.outcomes[0].action, "unroutable")
        self.assertIn("wheat", result.outcomes[0].reason)

    def test_original_signal_is_not_mutated(self):
        original = make_signal("SLV", "sell", prob_profit=0.1)
        self._route([original])
        self.assertEqual(original.symbol, "SLV")
        self.assertEqual(original.signal_type, "sell")
        self.assertAlmostEqual(original.prob_profit, 0.1)
        self.assertNotIn("inverse_routing", original.meta)

    def test_extra_symbols_lists_proxies_for_position_lookup(self):
        result = self._route([
            make_signal("SLV", "sell"),
            make_signal("USO", "sell"),
            make_signal("BNO", "sell"),  # also SCO
        ])
        self.assertCountEqual(result.extra_symbols, ["ZSL", "SCO"])

    def test_mixed_batch_routes_each_signal_independently(self):
        result = self._route(
            [
                make_signal("SLV", "sell"),    # -> BUY ZSL
                make_signal("WEAT", "sell"),   # -> unroutable
                make_signal("IAU", "buy"),     # -> passes through
                make_signal("CORN", "hold"),   # -> untouched
            ]
        )
        symbols = [(s.symbol, s.signal_type) for s in result.signals]
        self.assertIn(("ZSL", "buy"), symbols)
        self.assertIn(("IAU", "buy"), symbols)
        self.assertIn(("CORN", "hold"), symbols)
        self.assertNotIn("WEAT", [s for s, _ in symbols])

    def test_summary_and_table_render(self):
        result = self._route([make_signal("SLV", "sell"), make_signal("WEAT", "sell")])
        self.assertIn("routed_entry", result.summary())
        self.assertIn("unroutable", result.summary())
        table = format_routing_table(result.outcomes)
        self.assertIn("ZSL", table)
        self.assertIn("(no directional signals to route)", format_routing_table([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
