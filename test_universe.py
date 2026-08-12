"""Tests for the tradeable-universe definition and affordability screening.

The affordability tests use an injected price lookup so they assert the sizing
arithmetic rather than the network.
"""

import pytest

from logic.universe import (
    ALL_ASSETS,
    COMMODITIES,
    EQUITIES,
    THEMES,
    Asset,
    assets_for_theme,
    build_tradeable_universe,
    format_affordability_table,
    get_assets,
    get_symbols,
    lookup,
    screen_affordability,
)


# --- Universe definition ---------------------------------------------------


def test_symbols_are_unique_across_the_universe():
    symbols = [a.symbol for a in ALL_ASSETS]
    assert len(symbols) == len(set(symbols))


def test_every_asset_declares_at_least_one_known_theme():
    for asset in ALL_ASSETS:
        assert asset.themes, f"{asset.symbol} has no themes"
        assert set(asset.themes) <= set(THEMES)


def test_unknown_theme_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown theme"):
        Asset("ZZZ", "Bogus", "equity", "technology", "nothing", ("not_a_real_theme",))


def test_scopes_partition_sensibly():
    assert set(get_symbols("equities")) == {a.symbol for a in EQUITIES}
    assert set(get_symbols("commodities")) == {a.symbol for a in COMMODITIES}
    assert set(get_symbols("all")) == set(get_symbols("equities")) | set(get_symbols("commodities"))
    assert set(get_symbols("agriculture")) < set(get_symbols("commodities"))


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown universe scope"):
        get_assets("crypto")


def test_lookup_is_case_insensitive_and_returns_none_for_strangers():
    assert lookup("weat").symbol == "WEAT"
    assert lookup(" corn ").symbol == "CORN"
    assert lookup("TSLA") is None


def test_grain_theme_maps_to_the_war_sensitive_ag_funds():
    """The war->grains join the news analyst will rely on."""
    grains = {a.symbol for a in assets_for_theme("grains")}
    assert {"WEAT", "CORN", "SOYB"} <= grains


def test_geopolitical_theme_spans_ags_energy_and_safe_havens():
    hit = {a.sector for a in assets_for_theme("geopolitical_supply_shock")}
    assert {"agriculture", "energy", "precious_metals"} <= hit


def test_assets_for_unknown_theme_raises():
    with pytest.raises(ValueError, match="Unknown theme"):
        assets_for_theme("alien_invasion")


# --- Affordability ---------------------------------------------------------


def _fixed_prices(mapping):
    return lambda symbols: {s: mapping.get(s) for s in symbols}


def test_cheap_symbol_is_affordable_and_expensive_one_is_not():
    verdicts = {
        v.symbol: v
        for v in screen_affordability(
            ["WEAT", "GLD"],
            account_equity=487.44,
            max_position_fraction=0.20,
            price_lookup=_fixed_prices({"WEAT": 24.26, "GLD": 371.71}),
        )
    }
    # Cap is 487.44 * 0.20 = $97.49
    assert verdicts["WEAT"].affordable
    assert verdicts["WEAT"].max_shares == 4  # 97.488 / 24.26
    assert not verdicts["GLD"].affordable
    assert "cap is" in verdicts["GLD"].reason


def test_share_priced_exactly_at_the_cap_is_affordable():
    (verdict,) = screen_affordability(
        ["X"],
        account_equity=1000.0,
        max_position_fraction=0.10,
        price_lookup=_fixed_prices({"X": 100.0}),
    )
    assert verdict.affordable and verdict.max_shares == 1


def test_share_one_cent_over_the_cap_is_not():
    (verdict,) = screen_affordability(
        ["X"],
        account_equity=1000.0,
        max_position_fraction=0.10,
        price_lookup=_fixed_prices({"X": 100.01}),
    )
    assert not verdict.affordable and verdict.max_shares == 0


def test_missing_price_is_unaffordable_not_an_error():
    (verdict,) = screen_affordability(
        ["DELISTED"],
        account_equity=10_000.0,
        max_position_fraction=0.20,
        price_lookup=_fixed_prices({}),
    )
    assert not verdict.affordable
    assert verdict.price is None
    assert verdict.reason == "no price available"


def test_zero_equity_makes_everything_unaffordable():
    verdicts = screen_affordability(
        ["WEAT", "CORN"],
        account_equity=0.0,
        max_position_fraction=0.20,
        price_lookup=_fixed_prices({"WEAT": 24.26, "CORN": 17.93}),
    )
    assert not any(v.affordable for v in verdicts)


def test_every_screened_symbol_gets_a_verdict():
    symbols = ["WEAT", "CORN", "GLD", "NOPE"]
    verdicts = screen_affordability(
        symbols,
        account_equity=500.0,
        max_position_fraction=0.20,
        price_lookup=_fixed_prices({"WEAT": 24.26, "CORN": 17.93, "GLD": 371.71}),
    )
    assert [v.symbol for v in verdicts] == symbols


# --- Tradeable universe assembly -------------------------------------------


def test_tiny_account_still_gets_a_tradeable_commodity_universe():
    """The whole point: $487 can't buy any equity but can buy grains."""
    prices = {
        "AAPL": 328.0, "MSFT": 397.0, "AMZN": 248.0, "NVDA": 207.0, "GOOGL": 347.0,
        "WEAT": 24.26, "CORN": 17.93, "SOYB": 25.24, "CANE": 9.84, "DBA": 27.79,
        "UNG": 10.11, "BNO": 47.83, "USO": 122.12, "UGA": 111.72, "XLE": 58.79,
        "SLV": 52.46, "IAU": 76.19, "GLD": 371.71, "PPLT": 14.76, "PALL": 22.83,
        "CPER": 39.64, "DBB": 25.19, "DBC": 28.88, "GSG": 31.18,
    }
    symbols, verdicts = build_tradeable_universe(
        scope="all",
        account_equity=487.44,
        max_position_fraction=0.20,
        price_lookup=_fixed_prices(prices),
    )

    assert symbols, "a $487 account should still be able to trade something"
    assert not ({"AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"} & set(symbols))
    assert {"WEAT", "CORN", "UNG", "CANE"} <= set(symbols)
    assert {"GLD", "USO", "UGA"}.isdisjoint(symbols)
    assert len(verdicts) == len(get_symbols("all"))


def test_held_symbols_survive_the_screen_so_they_stay_sellable():
    symbols, _ = build_tradeable_universe(
        scope="equities",
        account_equity=487.44,
        max_position_fraction=0.20,
        always_include=["bac", "TSLA"],
        price_lookup=_fixed_prices({s: 400.0 for s in get_symbols("equities")}),
    )
    assert set(symbols) == {"BAC", "TSLA"}


def test_held_symbol_is_not_duplicated_when_also_affordable():
    symbols, _ = build_tradeable_universe(
        scope="agriculture",
        account_equity=10_000.0,
        max_position_fraction=0.20,
        always_include=["WEAT"],
        price_lookup=_fixed_prices({s: 20.0 for s in get_symbols("agriculture")}),
    )
    assert symbols.count("WEAT") == 1


def test_funded_account_can_trade_the_whole_universe():
    symbols, _ = build_tradeable_universe(
        scope="all",
        account_equity=10_000.0,
        max_position_fraction=0.20,
        price_lookup=_fixed_prices({s: 400.0 for s in get_symbols("all")}),
    )
    assert set(symbols) == set(get_symbols("all"))


def test_table_lists_affordable_first_and_handles_empty():
    verdicts = screen_affordability(
        ["GLD", "WEAT"],
        account_equity=500.0,
        max_position_fraction=0.20,
        price_lookup=_fixed_prices({"GLD": 371.71, "WEAT": 24.26}),
    )
    table = format_affordability_table(verdicts)
    assert table.index("WEAT") < table.index("GLD")
    assert format_affordability_table([]) == "(no symbols screened)"
