"""Rule-based agent layer for crypto-collateralized lending stress tests.

Borrower agents hold heterogeneous ETH collateral sizes and use different
fractions of the protocol's maximum LTV. A protocol monitor evaluates each
position using a lagged Chainlink-style oracle. Liquidator agents execute only
when collateral-sale proceeds after the shared pool-impact estimate cover the
repayment and configured execution cost.
"""

import numpy as np

from config import ModelConfig


def generate_borrower_population(config: ModelConfig) -> dict:
    """Generate one reproducible heterogeneous borrower population.

    Sizes are lognormal with mean normalized to 1 ETH. Utilization is sampled
    uniformly within the configured interval. Profiles stay fixed across LTV
    candidates so candidate comparisons use common agents.
    """
    rng = np.random.default_rng(config.seed + 1)
    sigma = float(config.borrower_size_sigma)
    raw_sizes = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=config.n_borrower_agents)
    scale = float(getattr(config, "borrower_collateral_scale", 1.0))
    collateral_eth = (raw_sizes / np.mean(raw_sizes)) * scale
    utilization = rng.uniform(
        config.min_borrow_utilization,
        config.max_borrow_utilization,
        size=config.n_borrower_agents,
    )
    return {
        "collateral_eth": collateral_eth.astype(float),
        "borrow_utilization": utilization.astype(float),
    }


def _estimate_slippage_array(
    notional_usd: np.ndarray,
    config: ModelConfig,
    effective_liquidity: float | np.ndarray | None = None,
) -> np.ndarray:
    """Estimate slippage given aggregate liquidation notional vs pool depth.

    Parameters
    ----------
    notional_usd : array of aggregate sale notional per MC scenario.
    config : ModelConfig.
    effective_liquidity : override pool depth in USD (used by Liquidity Crunch mechanism).
                          Can be a scalar float or a 1D array of shape (n_sims,).
                          If None, uses config.dex_liquidity_usd (baseline snapshot).
    """
    if effective_liquidity is not None:
        liquidity = np.maximum(np.asarray(effective_liquidity, dtype=float), 1.0)
    else:
        liquidity = max(float(config.dex_liquidity_usd), 1.0)
    ratio = np.maximum(notional_usd, 0.0) / liquidity
    slip = config.slippage_coefficient * np.power(ratio, config.slippage_exponent)
    return np.clip(slip, 0.0, config.max_slippage)


def _lagged_oracle_price(
    price_paths: np.ndarray,
    step: int,
    delay_steps: float,
    oracle_anchor_price: float,
) -> np.ndarray:
    """Interpolate in log-price space for fractional oracle delays."""
    n_steps = price_paths.shape[0] - 1
    effective_index = max(0.0, min(float(step) - delay_steps, float(n_steps)))
    lo = int(np.floor(effective_index))
    hi = min(lo + 1, n_steps)
    weight = effective_index - lo
    log_p0 = np.log(np.maximum(price_paths[lo], 1e-12))
    log_p1 = np.log(np.maximum(price_paths[hi], 1e-12))
    simulated_lagged = np.exp((1.0 - weight) * log_p0 + weight * log_p1)
    market_anchor = float(price_paths[0, 0])
    return simulated_lagged * (float(oracle_anchor_price) / market_anchor)


def simulate_agent_based_portfolio(
    price_paths: np.ndarray,
    ltv: float,
    config: ModelConfig,
    borrower_population: dict,
    oracle_anchor_price: float | None = None,
    flex_price_paths: np.ndarray | None = None,
) -> dict:
    """Simulate borrower/protocol/liquidator interactions for all MC paths.

    Asset architecture:
      - Collateral: ETH (quantity maintained in Collateral_ETH)
      - Debt: FLEX (quantity maintained in Debt_FLEX)
      - Numeraire: USD (used to value collateral, debt, Health Factor, and shortfall)

    Valuation formulas:
      Collateral_USD(t) = Collateral_ETH(t) * ETH_USD_Oracle_Price(t)
      Debt_USD(t) = Debt_FLEX(t) * FLEX_USD_Price(t)
      HF(t) = [Collateral_USD(t) * Liquidation_Threshold] / Debt_USD(t)

    FLEX price modes:
      - Mode A (pegged): FLEX/USD = 1.0 constant throughout horizon.
      - Mode B (scenario): flex_price_paths provided or flex_scenario_shock_pct applied.

    Metric definitions (all per Monte Carlo scenario, portfolio level):
      * Health_Factor_Breach: some borrower has oracle HF < 1 at some step.
      * Liquidation_Flag: at least one liquidation was EXECUTED (subset of HF breach).
      * Bad_Debt = Realized_Shortfall + Closeout_Shortfall (USD).
          - Realized: debt left on borrowers whose collateral was fully seized.
          - Closeout: shortfall on borrowers still holding collateral at horizon end.
      * Peak_Liquidation_Slippage: highest pool-impact applied to an interim batch.
    """
    config.validate()
    paths = np.asarray(price_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2:
        raise ValueError("price_paths phải có shape (n_steps + 1, n_sims).")
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise ValueError("price_paths chứa giá không hợp lệ.")

    n_rows, n_sims = paths.shape
    if n_sims != config.n_sims:
        raise ValueError(
            f"Số price paths ({n_sims}) không khớp config.n_sims ({config.n_sims})."
        )
    n_steps = n_rows - 1
    collateral_profile = np.asarray(borrower_population["collateral_eth"], dtype=float)
    utilization = np.asarray(borrower_population["borrow_utilization"], dtype=float)
    if collateral_profile.ndim != 1 or utilization.shape != collateral_profile.shape:
        raise ValueError("Borrower population không hợp lệ.")
    if collateral_profile.size != config.n_borrower_agents:
        raise ValueError("Số borrower agents không khớp config.n_borrower_agents.")

    s0 = float(paths[0, 0])
    if not np.allclose(paths[0], s0):
        raise ValueError("Tất cả price paths phải có cùng giá khởi tạo.")
    oracle_anchor = float(oracle_anchor_price if oracle_anchor_price is not None else s0)
    if oracle_anchor <= 0:
        raise ValueError("oracle_anchor_price phải > 0.")

    # ------------------------------------------------------------------
    # FLEX/USD Price Path Setup
    # ------------------------------------------------------------------
    if flex_price_paths is not None:
        flex_prices = np.asarray(flex_price_paths, dtype=float)
        if flex_prices.shape == (n_rows,):
            flex_prices = np.repeat(flex_prices[:, None], n_sims, axis=1)
        elif flex_prices.shape != (n_rows, n_sims):
            raise ValueError(f"flex_price_paths shape ({flex_prices.shape}) không khớp (n_steps + 1, n_sims).")
    elif config.flex_price_mode == "scenario" and config.flex_scenario_shock_pct != 0.0:
        base_flex_p = float(config.flex_initial_price_usd)
        shocked_flex_p = base_flex_p * (1.0 + float(config.flex_scenario_shock_pct))
        flex_prices = np.full((n_rows, n_sims), shocked_flex_p)
        flex_prices[0, :] = base_flex_p
    else:
        # Mode A: Pegged assumption (FLEX = 1 USD or flex_initial_price_usd)
        flex_prices = np.full((n_rows, n_sims), float(config.flex_initial_price_usd))

    initial_flex_price = float(flex_prices[0, 0])
    if initial_flex_price <= 0:
        raise ValueError("initial_flex_price phải > 0.")

    # ------------------------------------------------------------------
    # Initial Portfolios (Borrower agents x Monte Carlo scenarios)
    # ------------------------------------------------------------------
    collateral_eth = np.repeat(collateral_profile[:, None], n_sims, axis=1)
    initial_debt_usd_per_agent = (collateral_profile * s0 * float(ltv) * utilization)[:, None]
    initial_debt_usd = float(np.sum(initial_debt_usd_per_agent))

    # Debt maintained in FLEX tokens: Initial_Debt_FLEX = Initial_Debt_USD / Initial_FLEX_Price
    debt_flex = np.repeat(initial_debt_usd_per_agent / initial_flex_price, n_sims, axis=1)
    initial_debt_flex = float(np.sum(debt_flex[:, 0]))

    initial_oracle_hf = (
        collateral_eth * oracle_anchor * config.liquidation_threshold
        / np.maximum(debt_flex * initial_flex_price, 1e-10)
    )
    min_hf = initial_oracle_hf.copy()
    liquidation_events = np.zeros((collateral_profile.size, n_sims), dtype=np.int32)
    slippage_cost = np.zeros(n_sims, dtype=float)
    peak_liquidation_slippage = np.zeros(n_sims, dtype=float)
    delay_steps = max(0.0, config.oracle_delay_hours * config.steps_per_day / 24.0)
    dt_years = 1.0 / (365.0 * config.steps_per_day)
    interest_factor = np.exp(config.borrow_apr * dt_years)
    eps = 1e-10

    # ------------------------------------------------------------------
    # Endogenous Liquidity Crunch state
    # ------------------------------------------------------------------
    haircut_enabled = float(getattr(config, "liquidity_haircut_pct", 0.0)) > 0.0
    haircut_pct = float(getattr(config, "liquidity_haircut_pct", 0.0))
    haircut_trigger = float(getattr(config, "liquidity_haircut_trigger_pct", 0.15))
    haircut_recovery = int(getattr(config, "liquidity_haircut_recovery_steps", 0))
    # Vector of effective liquidity per MC scenario
    effective_liquidity_usd = np.full(n_sims, float(config.dex_liquidity_usd), dtype=float)
    has_haircut_triggered = np.zeros(n_sims, dtype=bool)
    steps_since_haircut = np.full(n_sims, -1, dtype=np.int32)
    rolling_window_steps = max(1, int(config.steps_per_day))
    effective_liquidity_history: list[float] = [float(config.dex_liquidity_usd)]

    # ------------------------------------------------------------------
    # Step-by-Step Simulation Loop
    # ------------------------------------------------------------------
    for step in range(1, n_rows):
        market_price = paths[step]
        oracle_price = _lagged_oracle_price(paths, step, delay_steps, oracle_anchor)
        flex_p = flex_prices[step]

        # ------------------------------------------------------------------
        # Endogenous Liquidity Crunch: update effective pool depth per path
        # Evaluates 24-hour rolling drop as well as cumulative drop from S0.
        # If ETH drops >= liquidity_haircut_trigger_pct, LPs withdraw liquidity
        # to avoid impermanent loss, slashing pool depth by haircut_pct.
        # ------------------------------------------------------------------
        if haircut_enabled:
            lookback_step = max(0, step - rolling_window_steps)
            lookback_price = paths[lookback_step]
            drop_24h = (lookback_price - market_price) / np.maximum(lookback_price, eps)
            drop_from_s0 = (paths[0] - market_price) / np.maximum(paths[0], eps)
            path_drop = np.maximum(drop_24h, drop_from_s0)

            new_triggers = (path_drop >= haircut_trigger) & ~has_haircut_triggered
            if np.any(new_triggers):
                has_haircut_triggered |= new_triggers
                effective_liquidity_usd = np.where(
                    has_haircut_triggered,
                    float(config.dex_liquidity_usd) * (1.0 - haircut_pct),
                    float(config.dex_liquidity_usd),
                )
                steps_since_haircut[new_triggers] = 0

            if haircut_recovery > 0:
                recovering = has_haircut_triggered & (steps_since_haircut >= 0)
                steps_since_haircut[recovering] += 1
                recovered = steps_since_haircut >= haircut_recovery
                if np.any(recovered):
                    effective_liquidity_usd[recovered] = float(config.dex_liquidity_usd)
                    has_haircut_triggered[recovered] = False
                    steps_since_haircut[recovered] = -1

        effective_liquidity_history.append(float(np.mean(effective_liquidity_usd)))

        # Accrue interest in FLEX debt tokens
        debt_flex *= interest_factor

        # USD valuations
        debt_usd = debt_flex * flex_p[None, :]
        collateral_usd = collateral_eth * oracle_price[None, :]

        # Health Factor evaluation
        hf = (collateral_usd * config.liquidation_threshold) / np.maximum(debt_usd, eps)
        min_hf = np.minimum(min_hf, hf)

        # State B: Eligible for liquidation
        eligible = (hf < 1.0) & (debt_flex > eps) & (collateral_eth > eps)
        if not np.any(eligible):
            continue

        # Close factor limits the repayable FLEX debt
        repay_requested_flex = debt_flex * config.close_factor
        repay_requested_usd = repay_requested_flex * flex_p[None, :]

        # Collateral seized in USD and converted to ETH
        seize_requested_usd = repay_requested_usd * (1.0 + config.liquidation_bonus)
        seize_requested_eth = seize_requested_usd / np.maximum(oracle_price[None, :], eps)

        # Collateral actually available to seize
        seize_amount_eth = np.minimum(collateral_eth, seize_requested_eth)
        # Cap repayment if collateral was fully seized
        repay_actual_usd = np.minimum(
            repay_requested_usd,
            seize_amount_eth * oracle_price[None, :] / (1.0 + config.liquidation_bonus),
        )
        repay_actual_flex = repay_actual_usd / np.maximum(flex_p[None, :], eps)
        sale_notional_usd = seize_amount_eth * market_price[None, :]

        # Liquidator agents assess aggregate order against shared liquidity pool
        preliminary_flow = np.sum(np.where(eligible, sale_notional_usd, 0.0), axis=0)
        preliminary_slippage = _estimate_slippage_array(preliminary_flow, config, effective_liquidity_usd)
        preliminary_profit = (
            sale_notional_usd * (1.0 - preliminary_slippage[None, :])
            - repay_actual_usd
            - config.liquidation_gas_cost_usd
        )
        candidates = eligible & (preliminary_profit > 0.0) & (repay_actual_usd > eps)

        # Reprice the executable flow once and execute profitable liquidations
        executable_flow = np.sum(np.where(candidates, sale_notional_usd, 0.0), axis=0)
        step_slippage = _estimate_slippage_array(executable_flow, config, effective_liquidity_usd)
        final_profit = (
            sale_notional_usd * (1.0 - step_slippage[None, :])
            - repay_actual_usd
            - config.liquidation_gas_cost_usd
        )
        execute = candidates & (final_profit > 0.0)

        # State C: Executed liquidations
        executed_seize_eth = np.where(execute, seize_amount_eth, 0.0)
        executed_repay_flex = np.where(execute, repay_actual_flex, 0.0)

        collateral_eth = np.maximum(collateral_eth - executed_seize_eth, 0.0)
        debt_flex = np.maximum(debt_flex - executed_repay_flex, 0.0)
        liquidation_events += execute.astype(np.int32)

        executed_notional = np.sum(np.where(execute, sale_notional_usd, 0.0), axis=0)
        slippage_cost += executed_notional * step_slippage
        peak_liquidation_slippage = np.maximum(
            peak_liquidation_slippage, np.where(np.any(execute, axis=0), step_slippage, 0.0)
        )

    # ------------------------------------------------------------------
    # Terminal Closeout (Horizon Step)
    # ------------------------------------------------------------------
    final_market_price = paths[-1]
    final_flex_price = flex_prices[-1]

    terminal_debt_usd = debt_flex * final_flex_price[None, :]
    terminal_notional_by_agent = collateral_eth * final_market_price[None, :]
    terminal_flow = np.sum(terminal_notional_by_agent, axis=0)
    terminal_slippage = _estimate_slippage_array(terminal_flow, config, effective_liquidity_usd)
    terminal_recovery = terminal_notional_by_agent * (1.0 - terminal_slippage[None, :])

    slippage_cost += terminal_flow * terminal_slippage
    bad_debt_by_agent = np.maximum(terminal_debt_usd - terminal_recovery, 0.0)
    bad_debt_scenarios = np.sum(bad_debt_by_agent, axis=0)

    collateral_exhausted = collateral_eth <= eps
    realized_shortfall = np.sum(np.where(collateral_exhausted, bad_debt_by_agent, 0.0), axis=0)
    closeout_shortfall = np.sum(np.where(~collateral_exhausted, bad_debt_by_agent, 0.0), axis=0)

    liquidation_events_per_scenario = np.sum(liquidation_events, axis=0)
    liquidation_flag = liquidation_events_per_scenario > 0
    minimum_hf_per_scenario = np.min(min_hf, axis=0)

    # Borrower-level summary rates across population
    borrower_hf_breach_rate = float(np.mean(np.any(min_hf < 1.0, axis=1)))
    borrower_liquidation_rate = float(np.mean(np.any(liquidation_events > 0, axis=1)))
    borrower_bad_debt_rate = float(np.mean(np.any(bad_debt_by_agent > eps, axis=1)))

    return {
        "Initial_Debt": initial_debt_usd,
        "Initial_Debt_USD": initial_debt_usd,
        "Initial_Debt_FLEX": initial_debt_flex,
        "Terminal_Collateral_ETH": collateral_eth,
        "Terminal_Debt_FLEX": debt_flex,
        "Bad_Debt_Scenarios": bad_debt_scenarios,
        "Realized_Shortfall_Scenarios": realized_shortfall,
        "Closeout_Shortfall_Scenarios": closeout_shortfall,
        "Peak_Liquidation_Slippage_Scenarios": peak_liquidation_slippage,
        "Liquidation_Flag_Scenarios": liquidation_flag,
        "Health_Factor_Breach_Scenarios": minimum_hf_per_scenario < 1.0,
        "Liquidation_Events_Scenarios": liquidation_events_per_scenario,
        "Minimum_Health_Factor_Scenarios": minimum_hf_per_scenario,
        "Slippage_Cost_Scenarios": slippage_cost,
        "Borrower_Agents": int(collateral_profile.size),
        "Borrower_HF_Breach_Rate": borrower_hf_breach_rate,
        "Borrower_Liquidation_Rate": borrower_liquidation_rate,
        "Borrower_Bad_Debt_Rate": borrower_bad_debt_rate,
        "FLEX_Price_Mode": config.flex_price_mode,
        "FLEX_Initial_Price": initial_flex_price,
        "FLEX_Price_Source": getattr(config, "flex_price_source", "assumed"),
        # Liquidity Crunch diagnostics
        "Liquidity_Haircut_Active": haircut_enabled,
        "Liquidity_Haircut_Pct": haircut_pct if haircut_enabled else 0.0,
        "Liquidity_Crunch_Steps": int(np.sum(has_haircut_triggered)),
        "Liquidity_Crunch_Scenario_Rate": float(np.mean(has_haircut_triggered)) if haircut_enabled else 0.0,
        "Liquidity_Crunch_Scenarios": has_haircut_triggered,
        "Effective_Liquidity_Final_USD": float(np.mean(effective_liquidity_usd)),
        "Effective_Liquidity_Scenarios": effective_liquidity_usd,
        "Effective_Liquidity_History": effective_liquidity_history,
    }
