import math

import numpy as np
import pandas as pd
from dataclasses import replace

from agent_engine import generate_borrower_population, simulate_agent_based_portfolio
from config import ModelConfig
from data_engine import fetch_historical_data, fetch_realtime_price, calibrate_merton_parameters
from onchain_engine import fetch_chainlink_oracle_data, fetch_uniswap_liquidity
from simulation import simulate_merton_jump_diffusion


def _ltv_grid(config: ModelConfig) -> list[float]:
    n = int(round((config.max_ltv - config.min_ltv) / config.ltv_step))
    return [round(config.min_ltv + i * config.ltv_step, 10) for i in range(n + 1)]


def _cvar_upper_tail(losses: np.ndarray, alpha: float = 0.99) -> float:
    """Empirical Expected Shortfall using exactly the upper (1-alpha) tail mass."""
    values = np.sort(np.asarray(losses, dtype=float))
    if values.size == 0:
        return 0.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha phải thuộc (0, 1).")
    tail_mass = (1.0 - alpha) * values.size
    if tail_mass <= 0:
        return float(values[-1])
    whole = int(np.floor(tail_mass))
    fraction = tail_mass - whole
    total = float(np.sum(values[-whole:])) if whole else 0.0
    if fraction > 1e-12:
        total += fraction * float(values[-whole - 1])
    return total / tail_mass


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a simulated event probability.

    The interval measures finite Monte Carlo sampling error only. It does not
    include uncertainty in calibrated parameters or model assumptions.
    """
    if trials <= 0:
        raise ValueError("trials phải > 0.")
    if successes < 0 or successes > trials:
        raise ValueError("successes phải nằm trong [0, trials].")
    p_hat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p_hat + z2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(p_hat * (1.0 - p_hat) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


# ---------------------------------------------------------------------------
# LTV feasibility and selection (Phase 1 of the roadmap)
#
# Core logic, in plain words:
#   for each LTV on the grid (ascending):
#       simulate -> metrics
#       feasible = ALL active limits hold
#           (probabilities are judged on the Wilson 95% UPPER bound)
#   optimal LTV = the largest feasible LTV; if none is feasible -> None
#                 (the model never falls back to a "closest" LTV).
# ---------------------------------------------------------------------------
_PROBABILITY_CONSTRAINTS = ("P(HF Breach)", "P(Liquidation)", "P(Bad Debt)")


def min_simulations_for_limit(limit: float, z: float = 1.959963984540054) -> float:
    """Smallest n_sims for which a Wilson upper bound can pass `limit`.

    Even with ZERO observed events the Wilson upper bound is z^2/(n+z^2), so a
    limit p is reachable only if n >= z^2 (1-p)/p. Below that, no LTV can ever
    be feasible when limits are judged on the upper bound.
    """
    if limit <= 0:
        return math.inf
    if limit >= 1:
        return 0.0
    return math.ceil(z * z * (1.0 - limit) / limit)


def evaluate_feasibility(
    row: dict,
    config: ModelConfig,
    use_ci_upper_bound: bool | None = None,
) -> list[tuple[str, float, float]]:
    """Return the violated constraints of one LTV candidate as (name, value, limit).

    An empty list means the candidate is feasible. Probability limits use the
    Wilson 95% upper bound when `use_ci_upper_bound` is True (default from
    config), otherwise the point estimate. CVaR99/InitialDebt is always the
    point estimate (a CI for the tail mean is not computed).
    """
    use_ci = config.use_ci_upper_bound if use_ci_upper_bound is None else use_ci_upper_bound

    def prob(name: str) -> float:
        return float(row[f"{name} CI95 High"] if use_ci else row[name])

    checks = []
    if config.max_hf_breach_probability is not None:
        checks.append(("P(HF Breach)", prob("P(HF Breach)"), config.max_hf_breach_probability))
    checks.extend([
        ("P(Liquidation)", prob("P(Liquidation)"), config.max_liquidation_probability),
        ("P(Bad Debt)", prob("P(Bad Debt)"), config.max_bad_debt_probability),
        ("CVaR99/InitialDebt", float(row["CVaR99/InitialDebt"]), config.max_cvar_ratio),
    ])
    return [(name, value, limit) for name, value, limit in checks if value > limit]


def format_violations(violations: list[tuple[str, float, float]], use_ci: bool) -> str:
    if not violations:
        return ""
    parts = []
    for name, value, limit in violations:
        tag = " (cận trên CI95)" if use_ci and name in _PROBABILITY_CONSTRAINTS else ""
        parts.append(f"{name}{tag} = {value:.2%} > {limit:.2%}")
    return "; ".join(parts)


def apply_feasibility(results_df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Add Feasible, Feasible (Point Est.) and Violated Constraints columns."""
    out = results_df.copy()
    feasible, feasible_point, reasons = [], [], []
    for row in out.to_dict("records"):
        violations = evaluate_feasibility(row, config)
        feasible.append(not violations)
        feasible_point.append(not evaluate_feasibility(row, config, use_ci_upper_bound=False))
        reasons.append(format_violations(violations, config.use_ci_upper_bound))
    out["Feasible (Point Est.)"] = feasible_point
    out["Feasible"] = feasible
    out["Violated Constraints"] = reasons
    return out


def select_optimal_ltv(rate_df: pd.DataFrame) -> dict:
    """Pick the largest feasible LTV for ONE APR scenario and record the evidence.

    Returns a status instead of guessing: if no LTV is feasible the optimum is
    None and `status == "NO_FEASIBLE_LTV"`. Also reports the next grid LTV that
    was rejected and why, whether the optimum sits on the grid's upper edge
    (a censored result), and any non-monotone pattern (a feasible LTV above an
    infeasible one), which would indicate Monte Carlo noise or model artefacts.
    """
    ordered = rate_df.sort_values("LTV").reset_index(drop=True)
    feasible = ordered[ordered["Feasible"]]
    if feasible.empty:
        lowest = ordered.iloc[0]
        return {
            "status": "NO_FEASIBLE_LTV",
            "optimal_ltv": None,
            "next_ltv": None,
            "next_ltv_violations": None,
            "lowest_tested_ltv": float(lowest["LTV"]),
            "lowest_tested_ltv_violations": str(lowest["Violated Constraints"]),
            "at_grid_upper_bound": False,
            "infeasible_ltv_below_optimal": [],
        }

    optimal = float(feasible["LTV"].max())
    higher = ordered[ordered["LTV"] > optimal + 1e-9]
    below_infeasible = ordered[(ordered["LTV"] < optimal - 1e-9) & (~ordered["Feasible"])]
    next_row = higher.iloc[0] if not higher.empty else None
    return {
        "status": "OPTIMAL_FOUND",
        "optimal_ltv": optimal,
        "next_ltv": float(next_row["LTV"]) if next_row is not None else None,
        "next_ltv_violations": str(next_row["Violated Constraints"]) if next_row is not None else None,
        "lowest_tested_ltv": float(ordered.iloc[0]["LTV"]),
        "lowest_tested_ltv_violations": None,
        "at_grid_upper_bound": next_row is None,
        "infeasible_ltv_below_optimal": [float(x) for x in below_infeasible["LTV"]],
    }


def liquidation_limit_sensitivity(
    rate_df: pd.DataFrame,
    config: ModelConfig,
    limits: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20),
) -> list[dict]:
    """Independent sensitivity: vary max_liquidation_probability holding other limits fixed.

    Answers: 'If the protocol tolerates higher liquidation risk, by how much can
    the optimal LTV increase?'
    """
    out = []
    for limit in limits:
        cfg = replace(config, max_liquidation_probability=float(limit))
        decision = select_optimal_ltv(apply_feasibility(rate_df, cfg))
        out.append({
            "limit": float(limit),
            "liquidation_limit": float(limit),  # Backward compatibility
            "optimal_ltv": decision["optimal_ltv"],
            "status": decision["status"],
            "next_ltv": decision["next_ltv"],
            "next_ltv_violations": decision["next_ltv_violations"],
        })
    return out


def hf_breach_limit_sensitivity(
    rate_df: pd.DataFrame,
    config: ModelConfig,
    limits: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20),
) -> list[dict]:
    """Independent sensitivity: vary max_hf_breach_probability holding other limits fixed.

    Evaluates protocol tolerance for borrowers entering unhealthy health factor territory.
    """
    out = []
    for limit in limits:
        cfg = replace(config, max_hf_breach_probability=float(limit))
        decision = select_optimal_ltv(apply_feasibility(rate_df, cfg))
        out.append({
            "limit": float(limit),
            "hf_breach_limit": float(limit),
            "optimal_ltv": decision["optimal_ltv"],
            "status": decision["status"],
            "next_ltv": decision["next_ltv"],
            "next_ltv_violations": decision["next_ltv_violations"],
        })
    return out


def bad_debt_limit_sensitivity(
    rate_df: pd.DataFrame,
    config: ModelConfig,
    limits: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05),
) -> list[dict]:
    """Independent sensitivity: vary max_bad_debt_probability holding other limits fixed.

    Evaluates protocol solvency constraint sensitivity.
    """
    out = []
    for limit in limits:
        cfg = replace(config, max_bad_debt_probability=float(limit))
        decision = select_optimal_ltv(apply_feasibility(rate_df, cfg))
        out.append({
            "limit": float(limit),
            "bad_debt_limit": float(limit),
            "optimal_ltv": decision["optimal_ltv"],
            "status": decision["status"],
            "next_ltv": decision["next_ltv"],
            "next_ltv_violations": decision["next_ltv_violations"],
        })
    return out


def cvar_ratio_limit_sensitivity(
    rate_df: pd.DataFrame,
    config: ModelConfig,
    limits: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05),
) -> list[dict]:
    """Independent sensitivity: vary max_cvar_ratio (CVaR99 / Initial Debt) holding other limits fixed.

    Evaluates tail loss tolerance sensitivity.
    """
    out = []
    for limit in limits:
        cfg = replace(config, max_cvar_ratio=float(limit))
        decision = select_optimal_ltv(apply_feasibility(rate_df, cfg))
        out.append({
            "limit": float(limit),
            "cvar_limit": float(limit),
            "optimal_ltv": decision["optimal_ltv"],
            "status": decision["status"],
            "next_ltv": decision["next_ltv"],
            "next_ltv_violations": decision["next_ltv_violations"],
        })
    return out


def comprehensive_risk_sensitivity(rate_df: pd.DataFrame, config: ModelConfig) -> dict:
    """Return all 4 independent sensitivities for complete governance risk profiling."""
    return {
        "liquidation": liquidation_limit_sensitivity(rate_df, config),
        "hf_breach": hf_breach_limit_sensitivity(rate_df, config),
        "bad_debt": bad_debt_limit_sensitivity(rate_df, config),
        "cvar_ratio": cvar_ratio_limit_sensitivity(rate_df, config),
    }


_DECISION_COLUMNS = [
    "LTV", "P(HF Breach)", "P(Liquidation)", "P(Bad Debt)",
    "Expected Loss", "CVaR99", "CVaR99/InitialDebt", "Feasible",
]


def format_decision_table(rate_df: pd.DataFrame, decision: dict) -> str:
    """Decision table: chosen LTV and the next LTV that was rejected."""
    ordered = rate_df.sort_values("LTV").reset_index(drop=True)
    rows = []
    if decision["optimal_ltv"] is not None:
        row = ordered[np.isclose(ordered["LTV"], decision["optimal_ltv"])].iloc[0]
        rows.append(("Optimal", row))
    if decision["next_ltv"] is not None:
        row = ordered[np.isclose(ordered["LTV"], decision["next_ltv"])].iloc[0]
        rows.append(("Next (bị loại)", row))
    if not rows:
        row = ordered.iloc[0]
        rows.append(("Thấp nhất đã thử", row))
    table = pd.DataFrame([{"Vai trò": role, **{c: r[c] for c in _DECISION_COLUMNS}} for role, r in rows])
    pct = lambda x: f"{x:.2%}"
    return table.to_string(index=False, formatters={
        "LTV": pct, "P(HF Breach)": pct, "P(Liquidation)": pct, "P(Bad Debt)": pct,
        "CVaR99/InitialDebt": pct,
        "Expected Loss": lambda x: f"${x:,.2f}", "CVaR99": lambda x: f"${x:,.2f}",
    })


def run_v2_onchain_analysis_with_inputs(config: ModelConfig | None = None):
    config = config or ModelConfig()
    config.validate()
    print("=== BƯỚC 1: DỮ LIỆU THỊ TRƯỜNG VÀ ON-CHAIN ===")

    current_price = fetch_realtime_price("ETHUSDT")
    print(f"Binance ETH/USDT: ${current_price:,.2f}")

    oracle_data = fetch_chainlink_oracle_data()
    # Use a conservative historical update-interval proxy. Latest oracle age is
    # kept separately as an observed freshness metric, not confused with lag.
    config.oracle_delay_hours = oracle_data["oracle_p90_update_interval_hours"]
    price_basis_gap = abs(current_price - oracle_data["oracle_price"]) / oracle_data["oracle_price"]
    print(f"Chainlink ETH/USD: ${oracle_data['oracle_price']:,.2f}")
    print(f"Oracle age now: {oracle_data['oracle_age_hours']:.4f} giờ")
    print(f"P90 update-interval proxy: {config.oracle_delay_hours:.4f} giờ")
    if price_basis_gap > 0.01:
        print(f"CẢNH BÁO: Binance/Chainlink price basis gap = {price_basis_gap:.2%}.")

    dex_data = fetch_uniswap_liquidity(oracle_data["oracle_price"])
    config.dex_liquidity_usd = float(dex_data["liquidity_usd"])
    print(f"Uniswap V3 pool-balance TVL proxy: ${config.dex_liquidity_usd:,.2f}")
    print("Lưu ý: TVL proxy không phải độ sâu thanh khoản có thể khớp lệnh tại mọi mức giá.")

    print("\n=== BƯỚC 2: HIỆU CHỈNH MERTON TỪ LỊCH SỬ BINANCE ===")
    historical_df = fetch_historical_data("ETHUSDT", days=1095)
    merton_params = calibrate_merton_parameters(historical_df, jump_threshold_sigma=3.0)
    print(f"mu diffusion log (annual): {merton_params['mu']:.2%}")
    print(f"sigma diffusion (annual): {merton_params['sigma']:.2%}")
    print(f"downside jump intensity: {merton_params['lam']:.3f}/năm")
    print(f"negative jump count: {merton_params['jump_count']}")
    print(f"negative jump mean: {merton_params['mu_j']:.2%}")
    print(f"negative jump std: {merton_params['sigma_j']:.2%}")
    if merton_params["jump_count"] < 5:
        print("CẢNH BÁO: Số downside jumps ít; tham số jump có độ bất định cao.")

    print(f"\n=== BƯỚC 3: MONTE CARLO - {config.n_sims:,} PATHS ===")
    price_paths = simulate_merton_jump_diffusion(current_price, merton_params, config)
    borrower_population = generate_borrower_population(config)
    print(f"Borrower agents: {config.n_borrower_agents}")
    print(f"Average collateral size: {np.mean(borrower_population['collateral_eth']):.3f} ETH")
    print(
        "Borrow utilization range: "
        f"{borrower_population['borrow_utilization'].min():.1%}–"
        f"{borrower_population['borrow_utilization'].max():.1%}"
    )

    results = []
    for apr in sorted(set(float(x) for x in config.borrow_apr_scenarios)):
        rate_config = replace(config, borrow_apr=apr)
        print(f"\n--- KỊCH BẢN BORROW APR: {apr:.0%} ---")
        for ltv in _ltv_grid(config):
            agent_result = simulate_agent_based_portfolio(
                price_paths=price_paths,
                ltv=ltv,
                config=rate_config,
                borrower_population=borrower_population,
                oracle_anchor_price=oracle_data["oracle_price"],
            )
            bad_debts = agent_result["Bad_Debt_Scenarios"]
            p_hf_breach = float(np.mean(agent_result["Health_Factor_Breach_Scenarios"]))
            p_liquidation = float(np.mean(agent_result["Liquidation_Flag_Scenarios"]))
            p_bad_debt = float(np.mean(bad_debts > 1e-8))
            hf_ci = _wilson_interval(
                int(np.sum(agent_result["Health_Factor_Breach_Scenarios"])), config.n_sims
            )
            liquidation_ci = _wilson_interval(
                int(np.sum(agent_result["Liquidation_Flag_Scenarios"])), config.n_sims
            )
            bad_debt_ci = _wilson_interval(int(np.sum(bad_debts > 1e-8)), config.n_sims)
            expected_loss = float(np.mean(bad_debts))
            realized = agent_result["Realized_Shortfall_Scenarios"]
            closeout = agent_result["Closeout_Shortfall_Scenarios"]
            closeout_share = float(np.mean(closeout)) / expected_loss if expected_loss > 0 else np.nan
            expected_slippage_cost = float(np.mean(agent_result["Slippage_Cost_Scenarios"]))
            mean_minimum_hf = float(np.mean(agent_result["Minimum_Health_Factor_Scenarios"]))
            var_99 = float(np.quantile(bad_debts, 0.99))
            cvar_99 = _cvar_upper_tail(bad_debts, alpha=0.99)
            initial_debt = float(agent_result["Initial_Debt"])
            cvar_ratio = cvar_99 / initial_debt if initial_debt > 0 else np.inf

            row = {
                "LTV": ltv,
                "Borrow APR": apr,
                "P(HF Breach)": p_hf_breach,
                "P(HF Breach) CI95 Low": hf_ci[0],
                "P(HF Breach) CI95 High": hf_ci[1],
                "P(Liquidation)": p_liquidation,
                "P(Liquidation) CI95 Low": liquidation_ci[0],
                "P(Liquidation) CI95 High": liquidation_ci[1],
                "P(Bad Debt)": p_bad_debt,
                "P(Bad Debt) CI95 Low": bad_debt_ci[0],
                "P(Bad Debt) CI95 High": bad_debt_ci[1],
                "Expected Loss": expected_loss,
                "P(Realized Shortfall)": float(np.mean(realized > 1e-8)),
                "Expected Realized Loss": float(np.mean(realized)),
                "Closeout Share of Expected Loss": closeout_share,
                "Peak Liquidation Slippage": float(np.max(agent_result["Peak_Liquidation_Slippage_Scenarios"])),
                "Expected Loss/InitialDebt": expected_loss / initial_debt if initial_debt else np.inf,
                "Expected Slippage Cost": expected_slippage_cost,
                "Mean Minimum HF": mean_minimum_hf,
                "VaR99": var_99,
                "CVaR99": cvar_99,
                "CVaR99/InitialDebt": cvar_ratio,
                "Average Liquidation Events": float(np.mean(agent_result["Liquidation_Events_Scenarios"])),
                "Borrower HF Breach Rate": agent_result.get("Borrower_HF_Breach_Rate", 0.0),
                "Borrower Liquidation Rate": agent_result.get("Borrower_Liquidation_Rate", 0.0),
                "Borrower Bad Debt Rate": agent_result.get("Borrower_Bad_Debt_Rate", 0.0),
            }
            results.append(row)
            print(f"Hoàn thành LTV {ltv:.0%} · APR {apr:.0%}")


    # --- LTV decision: apply ALL limits, then pick the largest feasible LTV per APR ---
    df = pd.DataFrame(results).sort_values(["Borrow APR", "LTV"]).reset_index(drop=True)
    df = apply_feasibility(df, config)
    apr_list = sorted(set(float(x) for x in config.borrow_apr_scenarios))
    optimal_ltv_by_apr, decision_by_apr, limit_sensitivity_by_apr = {}, {}, {}
    comprehensive_sensitivity_by_apr = {}
    for apr in apr_list:
        rate_df = df[np.isclose(df["Borrow APR"], apr)]
        decision = select_optimal_ltv(rate_df)
        if decision["optimal_ltv"] is not None:
            best = rate_df[np.isclose(rate_df["LTV"], decision["optimal_ltv"])].iloc[0]
            share = float(best["Closeout Share of Expected Loss"])
            decision["closeout_share_at_optimal"] = None if np.isnan(share) else share
            decision["peak_liquidation_slippage_at_optimal"] = float(best["Peak Liquidation Slippage"])
            decision["binding_constraints"] = decision["next_ltv_violations"] if decision["next_ltv"] is not None else "Biên trên của lưới (max_ltv)"
        else:
            decision["binding_constraints"] = decision["lowest_tested_ltv_violations"]
        key = f"{apr:.4f}"
        decision_by_apr[key] = decision
        optimal_ltv_by_apr[key] = decision["optimal_ltv"]
        limit_sensitivity_by_apr[key] = liquidation_limit_sensitivity(rate_df, config)
        comprehensive_sensitivity_by_apr[key] = comprehensive_risk_sensitivity(rate_df, config)
    baseline_apr = min(apr_list)
    optimal_ltv = optimal_ltv_by_apr[f"{baseline_apr:.4f}"]

    input_data = {
        "asset_structure": {
            "collateral_asset": "ETH",
            "debt_asset": "FLEX",
            "numeraire": "USD",
            "flex_price_mode": config.flex_price_mode,
            "flex_initial_price_usd": config.flex_initial_price_usd,
            "flex_scenario_shock_pct": config.flex_scenario_shock_pct,
            "flex_price_source": getattr(config, "flex_price_source", "assumed"),
            "mode_description": (
                "Mode A (Pegged Debt Assumption): 1 FLEX = $1.00 cố định"
                if config.flex_price_mode == "pegged"
                else f"Mode B (FLEX Stress Scenario): shock {config.flex_scenario_shock_pct:+.1%}"
            ),
        },
        "liquidity_crunch": {
            "haircut_active": float(getattr(config, "liquidity_haircut_pct", 0.0)) > 0.0,
            "haircut_pct": float(getattr(config, "liquidity_haircut_pct", 0.0)),
            "trigger_pct": float(getattr(config, "liquidity_haircut_trigger_pct", 0.15)),
            "recovery_steps": int(getattr(config, "liquidity_haircut_recovery_steps", 0)),
            "baseline_liquidity_usd": float(config.dex_liquidity_usd),
        },
        "market": {
            "binance_eth_price": current_price,
            "binance_chainlink_basis_gap": price_basis_gap,
        },
        "chainlink": oracle_data,
        "uniswap": dex_data,
        "merton": merton_params,
        "agents": {
            "n_borrower_agents": config.n_borrower_agents,
            "borrower_size_sigma": config.borrower_size_sigma,
            "min_borrow_utilization": config.min_borrow_utilization,
            "max_borrow_utilization": config.max_borrow_utilization,
        },
        "rate_sensitivity": {
            "borrow_apr_scenarios": sorted(set(float(x) for x in config.borrow_apr_scenarios)),
            "optimal_ltv_by_apr": optimal_ltv_by_apr,
            "rates_are_illustrative_scenarios": True,
        },
        "simulation": {
            "paths": config.n_sims,
            "horizon_days": config.horizon_days,
            "steps_per_day": config.steps_per_day,
            "ltv_min": config.min_ltv,
            "ltv_max": config.max_ltv,
            "ltv_step": config.ltv_step,
            "borrow_apr_scenarios": sorted(set(float(x) for x in config.borrow_apr_scenarios)),
            "oracle_delay_hours_used": config.oracle_delay_hours,
            "max_bad_debt_probability": config.max_bad_debt_probability,
            "max_cvar_ratio": config.max_cvar_ratio,
            "max_liquidation_probability": config.max_liquidation_probability,
            "max_hf_breach_probability": config.max_hf_breach_probability,
            "use_ci_upper_bound": config.use_ci_upper_bound,
        },
        "decision": {
            "baseline_apr": baseline_apr,
            "by_apr": decision_by_apr,
            "liquidation_limit_sensitivity_by_apr": limit_sensitivity_by_apr,
            "comprehensive_sensitivity_by_apr": comprehensive_sensitivity_by_apr,
            "limits_are_study_assumptions": True,
        },
    }

    print("\n=== BẢNG RỦI RO ===")
    percentage_columns = [
        "LTV", "Borrow APR", "P(HF Breach)", "P(HF Breach) CI95 Low", "P(HF Breach) CI95 High",
        "P(Liquidation)", "P(Liquidation) CI95 Low", "P(Liquidation) CI95 High",
        "P(Bad Debt)", "P(Bad Debt) CI95 Low", "P(Bad Debt) CI95 High",
        "Expected Loss/InitialDebt", "CVaR99/InitialDebt",
    ]
    formatters = {column: (lambda x: f"{x:.2%}") for column in percentage_columns}
    formatters.update({
        "Expected Loss": lambda x: f"${x:,.2f}",
        "Expected Slippage Cost": lambda x: f"${x:,.2f}",
        "Mean Minimum HF": lambda x: f"{x:.3f}",
        "VaR99": lambda x: f"${x:,.2f}",
        "CVaR99": lambda x: f"${x:,.2f}",
        "Average Liquidation Events": lambda x: f"{x:.2f}",
        "P(Realized Shortfall)": lambda x: f"{x:.2%}",
        "Expected Realized Loss": lambda x: f"${x:,.2f}",
        "Closeout Share of Expected Loss": lambda x: "n/a" if np.isnan(x) else f"{x:.1%}",
        "Peak Liquidation Slippage": lambda x: f"{x:.2%}",
    })
    print(df.drop(columns=["Violated Constraints"]).to_string(index=False, formatters=formatters))

    print("\n=== GIỚI HẠN RỦI RO (GIẢ ĐỊNH QUẢN TRỊ RỦI RO, KHÔNG PHẢI DỮ LIỆU QUAN SÁT) ===")
    basis = "cận trên Wilson CI95" if config.use_ci_upper_bound else "ước lượng điểm"
    print(f"Các xác suất được so với giới hạn bằng: {basis}")
    if config.max_hf_breach_probability is not None:
        print(f"  P(HF Breach)         <= {config.max_hf_breach_probability:.2%}")
    print(f"  P(Liquidation)       <= {config.max_liquidation_probability:.2%}")
    print(f"  P(Bad Debt)          <= {config.max_bad_debt_probability:.2%}  (terminal close-out shortfall)")
    print(f"  CVaR99/InitialDebt   <= {config.max_cvar_ratio:.2%}  (ước lượng điểm)")
    if config.use_ci_upper_bound:
        limits_in_use = [config.max_bad_debt_probability, config.max_liquidation_probability]
        if config.max_hf_breach_probability is not None:
            limits_in_use.append(config.max_hf_breach_probability)
        needed = max(min_simulations_for_limit(x) for x in limits_in_use)
        if config.n_sims < needed:
            print(f"CẢNH BÁO: n_sims={config.n_sims:,} < {needed:,.0f}: ngay cả khi 0 sự kiện, "
                  "cận trên Wilson vẫn vượt giới hạn nhỏ nhất ⇒ sẽ không có LTV khả thi.")

    for apr in apr_list:
        key = f"{apr:.4f}"
        decision = decision_by_apr[key]
        rate_df = df[np.isclose(df["Borrow APR"], apr)]
        print(f"\n=== QUYẾT ĐỊNH LTV · APR {apr:.0%} ===")
        print(format_decision_table(rate_df, decision))
        if decision["status"] == "NO_FEASIBLE_LTV":
            print(f"Trạng thái: NO_FEASIBLE_LTV — không có LTV nào trong "
                  f"[{config.min_ltv:.0%}, {config.max_ltv:.0%}] thỏa đồng thời các giới hạn.")
            print(f"Ngay cả LTV thấp nhất ({decision['lowest_tested_ltv']:.0%}) vi phạm: "
                  f"{decision['lowest_tested_ltv_violations']}")
            continue
        print(f"Trạng thái: OPTIMAL_FOUND — Optimal LTV = {decision['optimal_ltv']:.0%}")
        best = rate_df[np.isclose(rate_df["LTV"], decision["optimal_ltv"])].iloc[0]
        share = decision.get("closeout_share_at_optimal")
        if share is None:
            print("Cấu thành tổn thất tại Optimal: không ghi nhận Bad Debt trong các kịch bản mô phỏng.")
        else:
            print(f"Cấu thành Expected Loss tại Optimal: thanh lý thực tế {1 - share:.0%} · "
                  f"giả định đóng vị thế cuối kỳ {share:.0%}")
            if share > 0.5:
                print("LƯU Ý: Bad Debt chủ yếu đến từ giả định bán nốt tài sản cuối kỳ (thường do liquidator "
                      "không thực thi, ví dụ gas cao), không phải từ thanh lý đã xảy ra.")
        peak_slip = decision["peak_liquidation_slippage_at_optimal"]
        print(f"Slippage thanh lý cao nhất ghi nhận tại Optimal: {peak_slip:.2%} "
              f"(liquidation bonus {config.liquidation_bonus:.0%}).")
        if peak_slip < 0.5 * config.liquidation_bonus:
            print("LƯU Ý: slippage còn rất thấp so với bonus nên kênh thanh khoản dùng chung chưa làm thanh lý "
                  "hết lời ở quy mô danh mục này; thay đổi thanh khoản Uniswap sẽ ít tác động đến Optimal LTV.")
        if decision["next_ltv"] is not None:
            print(f"LTV kế tiếp {decision['next_ltv']:.0%} bị loại vì: {decision['next_ltv_violations']}")
        if decision["at_grid_upper_bound"]:
            print(f"CẢNH BÁO: Optimal LTV nằm đúng biên trên của lưới ({config.max_ltv:.0%}); "
                  "kết quả bị chặn bởi lưới tìm kiếm, cần mở rộng max_ltv hoặc siết giới hạn.")
        if decision["infeasible_ltv_below_optimal"]:
            lows = ", ".join(f"{x:.0%}" for x in decision["infeasible_ltv_below_optimal"])
            print(f"CẢNH BÁO: có LTV không khả thi nằm dưới mức tối ưu ({lows}); "
                  "rủi ro không đơn điệu theo LTV — kiểm tra nhiễu Monte Carlo/seed.")

    print(f"\n=== ĐỘ NHẠY CỦA OPTIMAL LTV THEO CÁC NGƯỠNG RỦI RO ĐỘC LẬP · APR {baseline_apr:.0%} ===")
    comp_sens = comprehensive_sensitivity_by_apr[f"{baseline_apr:.4f}"]

    print("--- 1. Độ nhạy theo P(Liquidation) (các giới hạn khác giữ nguyên) ---")
    for item in comp_sens["liquidation"]:
        label = f"{item['optimal_ltv']:.0%}" if item["optimal_ltv"] is not None else "không có LTV khả thi"
        nxt_info = f" (kế tiếp {item['next_ltv']:.0%} bị loại: {item['next_ltv_violations']})" if item["next_ltv"] is not None else ""
        print(f"  P(Liquidation) <= {item['limit']:>5.1%}  ->  Optimal LTV = {label}{nxt_info}")

    print("\n--- 2. Độ nhạy theo P(HF Breach) (các giới hạn khác giữ nguyên) ---")
    for item in comp_sens["hf_breach"]:
        label = f"{item['optimal_ltv']:.0%}" if item["optimal_ltv"] is not None else "không có LTV khả thi"
        nxt_info = f" (kế tiếp {item['next_ltv']:.0%} bị loại: {item['next_ltv_violations']})" if item["next_ltv"] is not None else ""
        print(f"  P(HF Breach)   <= {item['limit']:>5.1%}  ->  Optimal LTV = {label}{nxt_info}")

    print("\n--- 3. Độ nhạy theo P(Bad Debt) (các giới hạn khác giữ nguyên) ---")
    for item in comp_sens["bad_debt"]:
        label = f"{item['optimal_ltv']:.0%}" if item["optimal_ltv"] is not None else "không có LTV khả thi"
        nxt_info = f" (kế tiếp {item['next_ltv']:.0%} bị loại: {item['next_ltv_violations']})" if item["next_ltv"] is not None else ""
        print(f"  P(Bad Debt)    <= {item['limit']:>5.1%}  ->  Optimal LTV = {label}{nxt_info}")

    print("\n--- 4. Độ nhạy theo CVaR99 / Initial Debt (các giới hạn khác giữ nguyên) ---")
    for item in comp_sens["cvar_ratio"]:
        label = f"{item['optimal_ltv']:.0%}" if item["optimal_ltv"] is not None else "không có LTV khả thi"
        nxt_info = f" (kế tiếp {item['next_ltv']:.0%} bị loại: {item['next_ltv_violations']})" if item["next_ltv"] is not None else ""
        print(f"  CVaR99/Debt    <= {item['limit']:>5.1%}  ->  Optimal LTV = {label}{nxt_info}")

    base_decision = decision_by_apr[f"{baseline_apr:.4f}"]
    print(f"\nGiới hạn rủi ro đang trực tiếp quyết định / chặn Optimal LTV tại APR {baseline_apr:.0%}:")
    print(f"  -> {base_decision['binding_constraints']}")

    return df, optimal_ltv, input_data


def run_v1_6_onchain_analysis_with_inputs(config: ModelConfig | None = None):
    """Backward-compatible alias used by the existing Streamlit dashboard."""
    return run_v2_onchain_analysis_with_inputs(config=config)


def run_v1_6_onchain_analysis():
    df, optimal_ltv, _ = run_v2_onchain_analysis_with_inputs()
    return df, optimal_ltv


if __name__ == "__main__":
    run_v2_onchain_analysis_with_inputs()
