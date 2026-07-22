from dataclasses import dataclass


@dataclass
class PortfolioRiskConfig:
    max_portfolio_exposure: float = 0.30  # max % of cash that can be invested
    max_position_size: float = 0.20       # max fraction of equity per trade (aligned with planner 20%)
    max_trades_per_day: int = 3           # cap number of trades per cycle
    daily_loss_limit: float = 0.02        # stop trading if daily PnL drops below this
    paper_trading_mode: bool = True       # safety switch


def load_risk_config() -> PortfolioRiskConfig:
    """Return the default portfolio risk configuration.

    This can be extended later to load from a file or environment.
    """
    return PortfolioRiskConfig()
