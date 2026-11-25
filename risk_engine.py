"""Compatibility layer for the original single-position risk-engine interface.

The V2 analysis uses agent_engine.simulate_agent_based_portfolio directly.
"""

import numpy as np
from dataclasses import replace

from agent_engine import generate_borrower_population, simulate_agent_based_portfolio
from config import ModelConfig


def estimate_slippage(notional_usd: float, liquidity_usd: float, config: ModelConfig) -> float:
    if notional_usd <= 0:
        return 0.0
    ratio = notional_usd / max(float(liquidity_usd), 1.0)
    value = config.slippage_coefficient * ratio ** config.slippage_exponent
    return float(min(value, config.max_slippage))


def simulate_dynamic_lending_position(
    price_path: np.ndarray,
    ltv: float,
    config: ModelConfig,
    oracle_anchor_price: float | None = None,
) -> dict:
    """Legacy one-borrower wrapper; prefer the portfolio agent engine in V2."""
    # The wrapper simulates ONE path for ONE borrower; the agent engine requires
    # config.n_sims to equal the number of price paths, so it must be 1 here.
    local_config = replace(config, n_borrower_agents=1, n_sims=1)
    population = generate_borrower_population(local_config)
    result = simulate_agent_based_portfolio(
        np.asarray(price_path, dtype=float).reshape(-1, 1),
        ltv,
        local_config,
        population,
        oracle_anchor_price=oracle_anchor_price,
    )
    bad_debt = float(result["Bad_Debt_Scenarios"][0])
    return {
        "LTV": float(ltv),
        "Initial_Debt": float(result["Initial_Debt"]),
        "Liquidation_Events": int(result["Liquidation_Events_Scenarios"][0]),
        "Minimum_Health_Factor": float(result["Minimum_Health_Factor_Scenarios"][0]),
        "Bad_Debt": bad_debt,
        "Total_Slippage_Cost": float(result["Slippage_Cost_Scenarios"][0]),
    }
