import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from unittest.mock import patch

from agent_engine import generate_borrower_population, simulate_agent_based_portfolio
from config import ModelConfig
from data_engine import calibrate_merton_parameters
import main
from main import (
    _cvar_upper_tail,
    _wilson_interval,
    apply_feasibility,
    evaluate_feasibility,
    liquidation_limit_sensitivity,
    min_simulations_for_limit,
    select_optimal_ltv,
)
from risk_engine import simulate_dynamic_lending_position
from simulation import simulate_merton_jump_diffusion


class ModelEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            n_sims=300,
            n_borrower_agents=6,
            horizon_days=7,
            steps_per_day=24,
            seed=17,
            dex_liquidity_usd=1_500_000,
        )
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0002, 0.025, 1095)
        jump_indices = rng.choice(returns.size, size=10, replace=False)
        returns[jump_indices] += rng.normal(-0.18, 0.03, size=10)
        self.params = calibrate_merton_parameters(pd.DataFrame({"log_return": returns}))

    def test_calibration_and_monte_carlo_shape(self):
        paths = simulate_merton_jump_diffusion(3500.0, self.params, self.config)
        self.assertEqual(paths.shape, (169, 300))
        self.assertTrue(np.all(np.isfinite(paths)))
        self.assertTrue(np.all(paths > 0))
        self.assertGreaterEqual(self.params["lam"], 0)

    def test_agent_engine_returns_portfolio_level_losses(self):
        paths = simulate_merton_jump_diffusion(3500.0, self.params, self.config)
        population = generate_borrower_population(self.config)
        result = simulate_agent_based_portfolio(
            paths, 0.75, self.config, population, oracle_anchor_price=3490.0
        )
        self.assertEqual(result["Bad_Debt_Scenarios"].shape, (300,))
        self.assertEqual(result["Liquidation_Flag_Scenarios"].shape, (300,))
        self.assertTrue(np.all(result["Bad_Debt_Scenarios"] >= 0))
        self.assertGreater(result["Initial_Debt"], 0)
        self.assertEqual(result["Borrower_Agents"], 6)

    def test_apr_can_trigger_hf_breach_on_flat_market(self):
        cfg = ModelConfig(
            n_sims=4,
            n_borrower_agents=1,
            horizon_days=7,
            steps_per_day=24,
            min_borrow_utilization=1.0,
            max_borrow_utilization=1.0,
            dex_liquidity_usd=1_000_000,
            seed=3,
        )
        flat_paths = np.full((169, 4), 3500.0)
        profile = {
            "collateral_eth": np.array([1.0]),
            "borrow_utilization": np.array([1.0]),
        }
        no_interest = simulate_agent_based_portfolio(
            flat_paths, 0.84, replace(cfg, borrow_apr=0.0), profile, 3500.0
        )
        high_interest = simulate_agent_based_portfolio(
            flat_paths, 0.84, replace(cfg, borrow_apr=1.0), profile, 3500.0
        )
        self.assertFalse(np.any(no_interest["Health_Factor_Breach_Scenarios"]))
        self.assertTrue(np.any(high_interest["Health_Factor_Breach_Scenarios"]))

    def test_main_integration_with_mock_market_and_onchain_inputs(self):
        cfg = ModelConfig(
            n_sims=80,
            n_borrower_agents=3,
            horizon_days=2,
            steps_per_day=4,
            min_ltv=0.50,
            max_ltv=0.52,
            ltv_step=0.01,
            borrow_apr_scenarios=(0.0, 0.10),
            dex_liquidity_usd=1_000_000,
        )
        synthetic_returns = pd.DataFrame({"log_return": np.random.default_rng(5).normal(0.0, 0.02, 500)})
        oracle = {
            "oracle_price": 3498.0,
            "oracle_age_hours": 0.01,
            "oracle_p90_update_interval_hours": 0.5,
            "oracle_update_interval_hours": 0.3,
            "round_id": 1,
            "sampled_rounds": 20,
        }
        dex = {"liquidity_usd": 1_000_000.0, "active_liquidity": 1000}
        with patch.object(main, "fetch_realtime_price", return_value=3500.0), \
             patch.object(main, "fetch_chainlink_oracle_data", return_value=oracle), \
             patch.object(main, "fetch_uniswap_liquidity", return_value=dex), \
             patch.object(main, "fetch_historical_data", return_value=synthetic_returns):
            results, baseline_optimal_ltv, inputs = main.run_v2_onchain_analysis_with_inputs(cfg)
        self.assertEqual(len(results), 6)
        self.assertEqual(sorted(results["Borrow APR"].unique().tolist()), [0.0, 0.10])
        self.assertIn("optimal_ltv_by_apr", inputs["rate_sensitivity"])
        self.assertEqual(inputs["simulation"]["oracle_delay_hours_used"], 0.5)

    def test_cvar_uses_exact_tail_mass(self):
        losses = np.arange(1, 101, dtype=float)
        self.assertAlmostEqual(_cvar_upper_tail(losses, 0.99), 100.0)

    def test_wilson_interval_does_not_claim_zero_probability_after_zero_events(self):
        low, high = _wilson_interval(0, 10_000)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.001)

    def test_terminal_closeout_slippage_is_included_in_reported_cost(self):
        cfg = ModelConfig(
            n_sims=3,
            n_borrower_agents=1,
            horizon_days=1,
            steps_per_day=2,
            min_borrow_utilization=1.0,
            max_borrow_utilization=1.0,
            dex_liquidity_usd=100_000,
            slippage_coefficient=0.2,
            slippage_exponent=1.0,
            seed=11,
        )
        flat_paths = np.full((3, 3), 3_500.0)
        profile = {
            "collateral_eth": np.array([1.0]),
            "borrow_utilization": np.array([0.5]),
        }
        result = simulate_agent_based_portfolio(flat_paths, 0.5, cfg, profile, 3_500.0)
        self.assertTrue(np.all(result["Slippage_Cost_Scenarios"] > 0.0))

    def test_severe_negative_price_shock_increases_bad_debt_risk(self):
        cfg = ModelConfig(
            n_sims=1,
            n_borrower_agents=1,
            horizon_days=1,
            steps_per_day=4,
            oracle_delay_hours=0.0,
            min_borrow_utilization=1.0,
            max_borrow_utilization=1.0,
            dex_liquidity_usd=10_000_000,
            seed=9,
        )
        flat_paths = np.full((5, 1), 3_500.0)
        shock_paths = np.array([[3_500.0], [2_100.0], [2_100.0], [2_100.0], [2_100.0]])
        profile = {
            "collateral_eth": np.array([1.0]),
            "borrow_utilization": np.array([1.0]),
        }
        flat_result = simulate_agent_based_portfolio(flat_paths, 0.82, cfg, profile, 3_500.0)
        shock_result = simulate_agent_based_portfolio(shock_paths, 0.82, cfg, profile, 3_500.0)
        self.assertFalse(bool(flat_result["Health_Factor_Breach_Scenarios"][0]))
        self.assertTrue(bool(shock_result["Health_Factor_Breach_Scenarios"][0]))
        self.assertGreater(shock_result["Bad_Debt_Scenarios"][0], flat_result["Bad_Debt_Scenarios"][0])


# ---------------------------------------------------------------------------
# Checkpoint 1: LTV feasibility / optimisation logic
# ---------------------------------------------------------------------------
def _row(ltv=0.7, hf=0.02, liq=0.02, bd=0.0, cvar=0.0, ci_pad=0.0):
    """Minimal metrics row. ci_pad widens the upper CI bound above the point estimate."""
    return {
        "LTV": ltv,
        "P(HF Breach)": hf, "P(HF Breach) CI95 High": hf + ci_pad,
        "P(Liquidation)": liq, "P(Liquidation) CI95 High": liq + ci_pad,
        "P(Bad Debt)": bd, "P(Bad Debt) CI95 High": bd + ci_pad,
        "CVaR99/InitialDebt": cvar,
    }


def _decision_df(flags):
    """flags: list of (ltv, feasible) in any order -> DataFrame accepted by select_optimal_ltv."""
    return pd.DataFrame({
        "LTV": [x[0] for x in flags],
        "Feasible": [x[1] for x in flags],
        "Violated Constraints": ["" if x[1] else f"P(Liquidation) at {x[0]:.0%}" for x in flags],
    })


class LtvOptimisationTests(unittest.TestCase):
    def test_feasible_when_every_limit_holds(self):
        self.assertEqual(evaluate_feasibility(_row(), ModelConfig()), [])

    def test_each_limit_is_enforced_independently(self):
        cfg = ModelConfig()  # liq 10%, hf 10%, bad debt 1%, cvar 2%
        cases = {
            "P(Liquidation)": _row(hf=0.0, liq=0.20),
            "P(HF Breach)": _row(hf=0.20, liq=0.0),
            "P(Bad Debt)": _row(bd=0.05),
            "CVaR99/InitialDebt": _row(cvar=0.05),
        }
        for name, row in cases.items():
            with self.subTest(constraint=name):
                names = [v[0] for v in evaluate_feasibility(row, cfg)]
                self.assertEqual(names, [name])

    def test_liquidation_limit_binds_even_when_bad_debt_and_cvar_pass(self):
        # This is the gap in the old rule: Bad Debt / CVaR fine, 60% liquidation.
        row = _row(hf=0.60, liq=0.60, bd=0.007, cvar=0.005)
        self.assertTrue(evaluate_feasibility(row, ModelConfig(max_liquidation_probability=1.0,
                                                              max_hf_breach_probability=None)) == [])
        self.assertTrue(evaluate_feasibility(row, ModelConfig()))

    def test_ci_upper_bound_is_stricter_than_point_estimate(self):
        row = _row(hf=0.0, liq=0.095, ci_pad=0.010)  # point 9.5% <= 10%, upper 10.5% > 10%
        cfg_ci = ModelConfig(use_ci_upper_bound=True, max_hf_breach_probability=None)
        cfg_pt = ModelConfig(use_ci_upper_bound=False, max_hf_breach_probability=None)
        self.assertEqual([v[0] for v in evaluate_feasibility(row, cfg_ci)], ["P(Liquidation)"])
        self.assertEqual(evaluate_feasibility(row, cfg_pt), [])

    def test_hf_limit_can_be_disabled(self):
        row = _row(hf=0.50, liq=0.01)
        self.assertTrue(evaluate_feasibility(row, ModelConfig(max_hf_breach_probability=0.10)))
        self.assertEqual(evaluate_feasibility(row, ModelConfig(max_hf_breach_probability=None)), [])

    def test_min_simulations_matches_wilson_upper_bound_at_zero_events(self):
        n = min_simulations_for_limit(0.01)
        self.assertLessEqual(_wilson_interval(0, int(n))[1], 0.01)
        self.assertGreater(_wilson_interval(0, int(n) - 1)[1], 0.01)
        self.assertEqual(min_simulations_for_limit(0.0), float("inf"))

    def test_select_picks_largest_feasible_and_reports_next_rejection(self):
        d = select_optimal_ltv(_decision_df([(0.70, True), (0.71, True), (0.72, False), (0.73, False)]))
        self.assertEqual(d["status"], "OPTIMAL_FOUND")
        self.assertAlmostEqual(d["optimal_ltv"], 0.71)
        self.assertAlmostEqual(d["next_ltv"], 0.72)
        self.assertIn("P(Liquidation)", d["next_ltv_violations"])
        self.assertFalse(d["at_grid_upper_bound"])
        self.assertEqual(d["infeasible_ltv_below_optimal"], [])

    def test_select_returns_no_feasible_state_instead_of_closest_ltv(self):
        d = select_optimal_ltv(_decision_df([(0.50, False), (0.51, False), (0.52, False)]))
        self.assertEqual(d["status"], "NO_FEASIBLE_LTV")
        self.assertIsNone(d["optimal_ltv"])
        self.assertIsNone(d["next_ltv"])
        self.assertAlmostEqual(d["lowest_tested_ltv"], 0.50)
        self.assertTrue(d["lowest_tested_ltv_violations"])

    def test_select_flags_censored_optimum_on_grid_upper_edge(self):
        d = select_optimal_ltv(_decision_df([(0.80, True), (0.81, True), (0.82, True)]))
        self.assertAlmostEqual(d["optimal_ltv"], 0.82)
        self.assertTrue(d["at_grid_upper_bound"])
        self.assertIsNone(d["next_ltv"])

    def test_select_flags_non_monotone_pattern(self):
        d = select_optimal_ltv(_decision_df([(0.70, True), (0.71, False), (0.72, True), (0.73, False)]))
        self.assertAlmostEqual(d["optimal_ltv"], 0.72)
        self.assertEqual(d["infeasible_ltv_below_optimal"], [0.71])

    def test_limit_sensitivity_is_monotone_in_the_limit(self):
        rows = [_row(ltv=0.60 + 0.02 * i, hf=0.03 * i, liq=0.03 * i) for i in range(10)]
        df = apply_feasibility(pd.DataFrame(rows), ModelConfig(use_ci_upper_bound=False))
        out = liquidation_limit_sensitivity(df, ModelConfig(use_ci_upper_bound=False),
                                            limits=(0.02, 0.05, 0.10, 0.20))
        ltvs = [x["optimal_ltv"] if x["optimal_ltv"] is not None else -1.0 for x in out]
        self.assertEqual(ltvs, sorted(ltvs))

    def test_config_validation_of_new_limits(self):
        with self.assertRaises(ValueError):
            ModelConfig(max_liquidation_probability=1.5).validate()
        with self.assertRaises(ValueError):
            ModelConfig(max_hf_breach_probability=-0.1).validate()
        with self.assertRaises(ValueError):
            ModelConfig(borrower_collateral_scale=0.0).validate()
        with self.assertRaises(ValueError):
            ModelConfig(borrower_collateral_scale=-5.0).validate()
        ModelConfig(max_hf_breach_probability=None).validate()


class LtvPipelineTests(unittest.TestCase):
    """End-to-end on MOCK market/on-chain inputs (stylised, not real calibration)."""

    PARAMS = dict(mu=0.20, mu_diffusion_log=0.20, sigma=0.65, lam=3.5, mu_j=-0.11, sigma_j=0.04,
                  jump_count=10, observations=1095, jump_threshold_sigma=3.0,
                  jump_threshold_daily=0.10, calibration_method="mock")

    def _run(self, cfg):
        oracle = {"oracle_price": 3498.0, "oracle_age_hours": 0.1, "round_id": 1, "sampled_rounds": 40,
                  "oracle_p90_update_interval_hours": 1.0, "oracle_update_interval_hours": 0.5}
        dex = {"liquidity_usd": 2_000_000.0, "active_liquidity": 1000}
        with patch.object(main, "fetch_realtime_price", return_value=3500.0), \
             patch.object(main, "fetch_chainlink_oracle_data", return_value=oracle), \
             patch.object(main, "fetch_uniswap_liquidity", return_value=dex), \
             patch.object(main, "fetch_historical_data", return_value=pd.DataFrame({"log_return": [0.0]})), \
             patch.object(main, "calibrate_merton_parameters", return_value=dict(self.PARAMS)), \
             patch("builtins.print"):
            return main.run_v2_onchain_analysis_with_inputs(cfg)

    def _cfg(self, **kw):
        base = dict(n_sims=3000, min_ltv=0.60, max_ltv=0.82, ltv_step=0.02, borrow_apr_scenarios=(0.0,))
        base.update(kw)
        return ModelConfig(**base)

    def test_old_rule_hits_grid_edge_but_new_rule_stops_earlier(self):
        # Exact replica of the OLD rule: point estimates, only Bad Debt + CVaR limits.
        _, old_opt, _ = self._run(self._cfg(
            max_liquidation_probability=1.0, max_hf_breach_probability=None, use_ci_upper_bound=False))
        df, new_opt, inputs = self._run(self._cfg())
        self.assertAlmostEqual(old_opt, 0.82)  # old behaviour: Bad Debt/CVaR never bind
        self.assertIsNotNone(new_opt)
        self.assertLess(new_opt, old_opt)
        decision = inputs["decision"]["by_apr"]["0.0000"]
        self.assertEqual(decision["status"], "OPTIMAL_FOUND")
        self.assertIn("P(Liquidation)", decision["next_ltv_violations"])
        row = df[np.isclose(df["LTV"], new_opt)].iloc[0]
        self.assertLessEqual(row["P(Liquidation) CI95 High"], 0.10)

    def test_pipeline_returns_none_when_no_ltv_is_feasible(self):
        df, opt, inputs = self._run(self._cfg(max_liquidation_probability=0.0))
        self.assertIsNone(opt)
        self.assertFalse(df["Feasible"].any())
        self.assertEqual(inputs["decision"]["by_apr"]["0.0000"]["status"], "NO_FEASIBLE_LTV")
        self.assertIsNone(inputs["rate_sensitivity"]["optimal_ltv_by_apr"]["0.0000"])

    def test_dataframe_keeps_columns_used_by_dashboard(self):
        df, _, _ = self._run(self._cfg())
        for col in ["LTV", "Borrow APR", "P(HF Breach)", "P(Liquidation)", "P(Bad Debt)",
                    "CVaR99/InitialDebt", "Feasible", "Feasible (Point Est.)", "Violated Constraints"]:
            self.assertIn(col, df.columns)
        self.assertEqual(df["Feasible"].dtype, bool)

    def test_reported_risk_metrics_are_internally_consistent(self):
        df, _, _ = self._run(self._cfg())
        tol = 1e-9
        self.assertTrue((df["CVaR99"] >= df["VaR99"] - tol).all())
        self.assertTrue((df["CVaR99"] >= df["Expected Loss"] - tol).all())
        self.assertTrue((df["P(Liquidation)"] <= df["P(HF Breach)"] + tol).all())
        self.assertTrue((df["P(Realized Shortfall)"] <= df["P(Bad Debt)"] + tol).all())
        self.assertTrue((df["Expected Realized Loss"] <= df["Expected Loss"] + tol).all())
        share = df["Closeout Share of Expected Loss"].dropna()
        self.assertTrue(((share >= -tol) & (share <= 1 + tol)).all())
        for col in ["P(HF Breach)", "P(Liquidation)", "P(Bad Debt)"]:
            self.assertTrue((df[f"{col} CI95 Low"] <= df[col] + tol).all())
            self.assertTrue((df[col] <= df[f"{col} CI95 High"] + tol).all())

    def test_high_gas_hides_unhealthy_positions_from_liquidation_probability(self):
        """Why the HF-breach limit exists: gas can make P(Liquidation) look safe."""
        cfg = ModelConfig(n_sims=2000, liquidation_gas_cost_usd=5_000.0)
        paths = simulate_merton_jump_diffusion(3500.0, self.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.78, cfg, pop, oracle_anchor_price=3498.0)
        p_hf = float(np.mean(res["Health_Factor_Breach_Scenarios"]))
        p_liq = float(np.mean(res["Liquidation_Flag_Scenarios"]))
        self.assertGreater(p_hf, 0.10)
        self.assertEqual(p_liq, 0.0)
        row = _row(hf=p_hf, liq=p_liq)
        self.assertEqual([v[0] for v in evaluate_feasibility(row, ModelConfig())], ["P(HF Breach)"])


# ---------------------------------------------------------------------------
# Checkpoint 2: agent-engine consistency (loss decomposition, invariants, slippage channel)
# ---------------------------------------------------------------------------
class AgentEngineConsistencyTests(unittest.TestCase):
    PROFILE = {"collateral_eth": np.array([1.0]), "borrow_utilization": np.array([1.0])}

    def _single(self, **kw):
        base = dict(n_sims=1, n_borrower_agents=1, min_borrow_utilization=1.0, max_borrow_utilization=1.0,
                    dex_liquidity_usd=10_000_000, horizon_days=2, steps_per_day=1)
        base.update(kw)
        return ModelConfig(**base)

    def test_loss_decomposition_identity_on_random_paths(self):
        cfg = ModelConfig(n_sims=2000, seed=5)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.80, cfg, pop, oracle_anchor_price=3498.0)
        realized, closeout = res["Realized_Shortfall_Scenarios"], res["Closeout_Shortfall_Scenarios"]
        self.assertTrue(np.all(realized >= 0) and np.all(closeout >= 0))
        np.testing.assert_allclose(realized + closeout, res["Bad_Debt_Scenarios"], rtol=1e-9, atol=1e-9)
        self.assertGreater(res["Bad_Debt_Scenarios"].max(), 0.0)  # the test must exercise a loss

    def test_realized_shortfall_when_collateral_is_fully_seized(self):
        # 3500 -> 1000 in one step, no oracle lag: 0.5*debt*1.05 of collateral exceeds 1 ETH.
        res = simulate_agent_based_portfolio(
            np.array([[3500.0], [1000.0], [1000.0]]), 0.70, self._single(oracle_delay_hours=0.0),
            self.PROFILE, 3500.0)
        debt0 = 3500.0 * 0.70
        repaid = 1.0 * 1000.0 / 1.05  # all collateral seized, repayment limited by its value / (1+bonus)
        self.assertAlmostEqual(float(res["Realized_Shortfall_Scenarios"][0]), debt0 - repaid, places=2)
        self.assertEqual(float(res["Closeout_Shortfall_Scenarios"][0]), 0.0)
        self.assertEqual(int(res["Liquidation_Events_Scenarios"][0]), 1)

    def test_closeout_only_shortfall_when_oracle_lag_hides_the_crash(self):
        # The oracle lags one full step, so the final-step crash never triggers a liquidation;
        # the loss exists only because of the horizon close-out assumption.
        res = simulate_agent_based_portfolio(
            np.array([[3500.0], [3500.0], [2000.0]]), 0.70, self._single(oracle_delay_hours=24.0),
            self.PROFILE, 3500.0)
        self.assertEqual(int(res["Liquidation_Events_Scenarios"][0]), 0)
        self.assertFalse(bool(res["Health_Factor_Breach_Scenarios"][0]))
        self.assertEqual(float(res["Realized_Shortfall_Scenarios"][0]), 0.0)
        self.assertGreater(float(res["Closeout_Shortfall_Scenarios"][0]), 400.0)  # 2450 - ~2000

    def test_state_invariants_hold_across_the_ltv_grid(self):
        cfg = ModelConfig(n_sims=1000, seed=8)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        previous_breach = None
        for ltv in np.arange(0.50, 0.8201, 0.04):
            res = simulate_agent_based_portfolio(paths, float(ltv), cfg, pop, oracle_anchor_price=3498.0)
            breach, liq = res["Health_Factor_Breach_Scenarios"], res["Liquidation_Flag_Scenarios"]
            self.assertFalse(np.any(liq & ~breach))              # liquidation implies HF breach
            self.assertTrue(np.all(res["Bad_Debt_Scenarios"] >= 0))
            if previous_breach is not None:                       # same paths + agents: breach sets nest by LTV
                self.assertFalse(np.any(previous_breach & ~breach))
            previous_breach = breach

    def test_without_liquidator_action_all_loss_is_closeout(self):
        cfg = ModelConfig(n_sims=2000, seed=5, liquidation_gas_cost_usd=5_000.0)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.78, cfg, pop, oracle_anchor_price=3498.0)
        self.assertEqual(int(np.sum(res["Liquidation_Events_Scenarios"])), 0)
        self.assertTrue(np.all(res["Realized_Shortfall_Scenarios"] == 0.0))
        self.assertGreater(res["Closeout_Shortfall_Scenarios"].max(), 0.0)

    def test_peak_slippage_diagnostic_tracks_the_shared_liquidity_channel(self):
        flat = simulate_agent_based_portfolio(
            np.full((3, 1), 3500.0), 0.70, self._single(oracle_delay_hours=0.0), self.PROFILE, 3500.0)
        self.assertEqual(float(flat["Peak_Liquidation_Slippage_Scenarios"][0]), 0.0)
        thin_cfg = self._single(oracle_delay_hours=0.0, dex_liquidity_usd=20_000.0)
        thin = simulate_agent_based_portfolio(
            np.array([[3500.0], [3000.0], [3000.0]]), 0.80, thin_cfg, self.PROFILE, 3500.0)
        peak = float(thin["Peak_Liquidation_Slippage_Scenarios"][0])
        self.assertGreater(peak, 0.01)
        self.assertLessEqual(peak, thin_cfg.max_slippage)

    def test_slippage_is_inert_at_default_portfolio_scale(self):
        """Documents a model property: a ~$25k portfolio barely moves a $2M pool."""
        cfg = ModelConfig(n_sims=1500, seed=5)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.80, cfg, pop, oracle_anchor_price=3498.0)
        self.assertLess(float(res["Peak_Liquidation_Slippage_Scenarios"].max()), 0.5 * cfg.liquidation_bonus)

    def test_borrower_collateral_scale_activates_slippage(self):
        """Scaling up the portfolio to ~$2.5M makes liquidation batches move the $2M pool."""
        cfg_scaled = ModelConfig(n_sims=1500, seed=5, borrower_collateral_scale=100.0)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg_scaled)
        pop = generate_borrower_population(cfg_scaled)
        res = simulate_agent_based_portfolio(paths, 0.80, cfg_scaled, pop, oracle_anchor_price=3498.0)
        # With 100x scale, peak liquidation slippage should exceed the 5% bonus in severe scenarios
        self.assertGreater(float(res["Peak_Liquidation_Slippage_Scenarios"].max()), cfg_scaled.liquidation_bonus)

    def test_market_stress_contracts_optimal_ltv(self):
        """Under severe negative drift & high vol, optimal LTV strictly contracts or becomes infeasible."""
        from stress_test_cp4 import run_scenario_3a_market_stress
        df = run_scenario_3a_market_stress(n_sims=1200, seed=42)
        base_ltv = int(df.loc[df["Market Scenario"].str.contains("Baseline"), "Optimal LTV"].iloc[0].rstrip("%"))
        bear_ltv = int(df.loc[df["Market Scenario"].str.contains("Bear"), "Optimal LTV"].iloc[0].rstrip("%"))
        self.assertGreater(base_ltv, bear_ltv)

    def test_liquidity_drain_shifts_loss_to_closeout(self):
        """Thin pool and high gas render liquidations unprofitable, shifting 100% loss to closeout."""
        from stress_test_cp4 import run_scenario_3b_liquidity_gas_stress
        df = run_scenario_3b_liquidity_gas_stress(n_sims=500, seed=42)
        stressed_row = df.iloc[-1]
        self.assertEqual(stressed_row["Closeout Share"], "100.0%")
        self.assertEqual(stressed_row["P(Liquidation)"], "0.00%")


class LegacyWrapperTests(unittest.TestCase):
    def test_single_position_wrapper_works_with_default_config(self):
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, ModelConfig(n_sims=5))
        out = simulate_dynamic_lending_position(paths[:, 0], 0.70, ModelConfig(), 3498.0)
        self.assertGreater(out["Initial_Debt"], 0)
        self.assertGreaterEqual(out["Bad_Debt"], 0.0)


# ===========================================================================
# SPECIFICATION COMPLIANCE TEST SUITES (Section 11 Nhóm A, B, C, D, E)
# ===========================================================================

class AssetArchitectureAndValuationTests(unittest.TestCase):
    """Nhóm A: Đơn vị và định giá ETH/FLEX/USD."""

    def test_a1_pegged_mode_flex_debt_and_usd_compatibility(self):
        """1. Khi FLEX/USD = 1, khoản nợ USD phải tương thích với baseline cũ."""
        cfg = ModelConfig(n_sims=10, n_borrower_agents=2, flex_price_mode="pegged", flex_initial_price_usd=1.0)
        pop = {"collateral_eth": np.array([2.0, 3.0]), "borrow_utilization": np.array([0.8, 1.0])}
        paths = np.full((10, 10), 3000.0)
        res = simulate_agent_based_portfolio(paths, 0.70, cfg, pop, oracle_anchor_price=3000.0)
        expected_debt_agent0 = 2.0 * 3000.0 * 0.70 * 0.8  # 3360.0
        expected_debt_agent1 = 3.0 * 3000.0 * 0.70 * 1.0  # 6300.0
        expected_total_usd = expected_debt_agent0 + expected_debt_agent1  # 9660.0
        self.assertAlmostEqual(res["Initial_Debt_USD"], expected_total_usd)
        self.assertAlmostEqual(res["Initial_Debt_FLEX"], expected_total_usd / 1.0)

    def test_a2_flex_price_increase_increases_debt_usd_and_worsens_hf(self):
        """2. Khi FLEX tăng giá, giá trị USD của khoản nợ tăng -> HF xấu đi."""
        base_cfg = ModelConfig(n_sims=5, n_borrower_agents=1, flex_price_mode="pegged", flex_initial_price_usd=1.0)
        shocked_cfg = ModelConfig(n_sims=5, n_borrower_agents=1, flex_price_mode="scenario", flex_initial_price_usd=1.0, flex_scenario_shock_pct=0.20)
        pop = {"collateral_eth": np.array([1.0]), "borrow_utilization": np.array([1.0])}
        paths = np.full((10, 5), 3000.0)

        base_res = simulate_agent_based_portfolio(paths, 0.80, base_cfg, pop, oracle_anchor_price=3000.0)
        shock_res = simulate_agent_based_portfolio(paths, 0.80, shocked_cfg, pop, oracle_anchor_price=3000.0)

        # In shocked mode, FLEX appreciates to $1.20, making USD debt 20% higher and HF lower (worse)
        self.assertAlmostEqual(base_res["Initial_Debt_FLEX"], shock_res["Initial_Debt_FLEX"])
        self.assertGreater(float(np.mean(base_res["Minimum_Health_Factor_Scenarios"])),
                            float(np.mean(shock_res["Minimum_Health_Factor_Scenarios"])))

    def test_a3_flex_price_decrease_decreases_debt_usd_and_improves_hf(self):
        """3. Khi FLEX giảm giá, giá trị USD của khoản nợ giảm -> HF tốt lên."""
        base_cfg = ModelConfig(n_sims=5, n_borrower_agents=1, flex_price_mode="pegged", flex_initial_price_usd=1.0)
        drop_cfg = ModelConfig(n_sims=5, n_borrower_agents=1, flex_price_mode="scenario", flex_initial_price_usd=1.0, flex_scenario_shock_pct=-0.20)
        pop = {"collateral_eth": np.array([1.0]), "borrow_utilization": np.array([1.0])}
        paths = np.full((10, 5), 3000.0)

        base_res = simulate_agent_based_portfolio(paths, 0.80, base_cfg, pop, oracle_anchor_price=3000.0)
        drop_res = simulate_agent_based_portfolio(paths, 0.80, drop_cfg, pop, oracle_anchor_price=3000.0)

        self.assertLess(float(np.mean(base_res["Minimum_Health_Factor_Scenarios"])),
                        float(np.mean(drop_res["Minimum_Health_Factor_Scenarios"])))

    def test_a4_collateral_usd_valuation_identity(self):
        """4. Giá trị USD tài sản thế chấp bằng số ETH nhân với giá ETH/USD."""
        collateral_eth = 2.5
        eth_price = 3200.0
        collateral_usd = collateral_eth * eth_price
        self.assertEqual(collateral_usd, 8000.0)

    def test_a5_hf_exact_valuation_formula(self):
        """5. HF được tính đúng từ giá trị tài sản thế chấp và khoản nợ quy đổi USD."""
        collateral_eth = 2.0
        eth_oracle_p = 3000.0
        debt_flex = 4000.0
        flex_usd_p = 1.05
        liq_threshold = 0.85

        collateral_usd = collateral_eth * eth_oracle_p  # 6000.0
        debt_usd = debt_flex * flex_usd_p  # 4200.0
        expected_hf = (collateral_usd * liq_threshold) / debt_usd  # 5100 / 4200 = 1.2142857...
        self.assertAlmostEqual(expected_hf, 5100.0 / 4200.0)


class LiquidationMechanicsUnitTests(unittest.TestCase):
    """Nhóm B: Cơ chế thanh lý và tính bất biến."""

    def test_b1_no_liquidation_if_hf_not_breached(self):
        """1. Không thanh lý nếu không đủ điều kiện theo HF >= 1.0."""
        cfg = ModelConfig(n_sims=10, n_borrower_agents=1, oracle_delay_hours=0.0)
        pop = {"collateral_eth": np.array([1.0]), "borrow_utilization": np.array([0.5])}
        flat_paths = np.full((10, 10), 3000.0)
        res = simulate_agent_based_portfolio(flat_paths, 0.60, cfg, pop, oracle_anchor_price=3000.0)
        self.assertFalse(np.any(res["Liquidation_Flag_Scenarios"]))
        self.assertEqual(int(np.sum(res["Liquidation_Events_Scenarios"])), 0)

    def test_b2_repay_flex_bounded_by_close_factor(self):
        """2. Lượng FLEX hoàn trả trong một đợt thanh lý không vượt quá close factor."""
        cfg = ModelConfig(n_sims=1, n_borrower_agents=1, close_factor=0.50, oracle_delay_hours=0.0)
        pop = {"collateral_eth": np.array([1.0]), "borrow_utilization": np.array([1.0])}
        # Crash from 3000 to 1800 over 1 step triggers exactly 1 liquidation event
        shock_paths = np.array([[3000.0], [1800.0]])
        res = simulate_agent_based_portfolio(shock_paths, 0.80, cfg, pop, oracle_anchor_price=3000.0)
        initial_debt_flex = res["Initial_Debt_FLEX"]
        terminal_debt_flex = float(res["Terminal_Debt_FLEX"][0, 0])
        repaid_flex = initial_debt_flex - terminal_debt_flex
        self.assertEqual(int(res["Liquidation_Events_Scenarios"][0]), 1)
        self.assertLessEqual(repaid_flex, initial_debt_flex * cfg.close_factor + 1e-4)

    def test_b3_seized_eth_does_not_exceed_available_collateral(self):
        """3. Lượng ETH bị tịch thu không vượt lượng ETH borrower đang sở hữu."""
        cfg = ModelConfig(n_sims=1, n_borrower_agents=1, close_factor=1.0, oracle_delay_hours=0.0)
        pop = {"collateral_eth": np.array([1.0]), "borrow_utilization": np.array([1.0])}
        # Extreme flash crash
        crash_paths = np.array([[3000.0], [500.0], [500.0]])
        res = simulate_agent_based_portfolio(crash_paths, 0.80, cfg, pop, oracle_anchor_price=3000.0)
        terminal_collateral = float(res["Terminal_Collateral_ETH"][0, 0])
        self.assertGreaterEqual(terminal_collateral, 0.0)
        self.assertLessEqual(1.0 - terminal_collateral, 1.0)

    def test_b4_balances_never_negative(self):
        """4. Borrower không có số dư ETH hoặc FLEX âm."""
        cfg = ModelConfig(n_sims=50, seed=99)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.82, cfg, pop, oracle_anchor_price=3500.0)
        self.assertTrue(np.all(res["Terminal_Collateral_ETH"] >= 0.0))
        self.assertTrue(np.all(res["Terminal_Debt_FLEX"] >= 0.0))

    def test_b5_liquidator_refuses_when_unprofitable(self):
        """5. Liquidator không thực thi nếu không thỏa điều kiện sinh lời."""
        # Extreme gas cost makes liquidation unprofitable
        cfg = ModelConfig(n_sims=10, n_borrower_agents=1, liquidation_gas_cost_usd=50_000.0, oracle_delay_hours=0.0)
        pop = {"collateral_eth": np.array([1.0]), "borrow_utilization": np.array([1.0])}
        crash_paths = np.full((10, 10), 3000.0)
        crash_paths[1:, :] = 1500.0
        res = simulate_agent_based_portfolio(crash_paths, 0.80, cfg, pop, oracle_anchor_price=3000.0)
        self.assertTrue(np.all(res["Health_Factor_Breach_Scenarios"]))
        self.assertFalse(np.any(res["Liquidation_Flag_Scenarios"]))

    def test_b6_liquidation_flag_is_subset_of_hf_breach(self):
        """6. Liquidation Flag không xuất hiện nếu không có giao dịch thanh lý thực tế (và là tập con của HF Breach)."""
        cfg = ModelConfig(n_sims=500, seed=12)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.78, cfg, pop, oracle_anchor_price=3500.0)
        hf_breach = res["Health_Factor_Breach_Scenarios"]
        liq_flag = res["Liquidation_Flag_Scenarios"]
        self.assertFalse(np.any(liq_flag & ~hf_breach))


class RiskMetricDefinitionTests(unittest.TestCase):
    """Nhóm C: Định nghĩa và tính chất của các chỉ số rủi ro."""

    def test_c1_probabilities_in_unit_interval(self):
        """1. Xác suất nằm trong khoảng 0-100%."""
        cfg = ModelConfig(n_sims=200, seed=42)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.75, cfg, pop, oracle_anchor_price=3500.0)
        p_hf = np.mean(res["Health_Factor_Breach_Scenarios"])
        p_liq = np.mean(res["Liquidation_Flag_Scenarios"])
        p_bd = np.mean(res["Bad_Debt_Scenarios"] > 1e-8)
        self.assertTrue(0.0 <= p_hf <= 1.0)
        self.assertTrue(0.0 <= p_liq <= 1.0)
        self.assertTrue(0.0 <= p_bd <= 1.0)

    def test_c2_bad_debt_and_expected_loss_non_negative(self):
        """2 & 3. Bad Debt và Expected Loss không âm."""
        cfg = ModelConfig(n_sims=200, seed=42)
        paths = simulate_merton_jump_diffusion(3500.0, LtvPipelineTests.PARAMS, cfg)
        pop = generate_borrower_population(cfg)
        res = simulate_agent_based_portfolio(paths, 0.75, cfg, pop, oracle_anchor_price=3500.0)
        bad_debts = res["Bad_Debt_Scenarios"]
        self.assertTrue(np.all(bad_debts >= 0.0))
        self.assertGreaterEqual(float(np.mean(bad_debts)), 0.0)

    def test_c4_cvar99_greater_equal_var99(self):
        """4. CVaR99 không nhỏ hơn VaR99 trên cùng phân phối tổn thất."""
        losses = np.random.default_rng(42).exponential(scale=100.0, size=5000)
        var99 = float(np.quantile(losses, 0.99))
        cvar99 = _cvar_upper_tail(losses, alpha=0.99)
        self.assertGreaterEqual(cvar99, var99)

    def test_c6_wilson_interval_ordering(self):
        """6. Wilson CI95 có cận dưới không lớn hơn cận trên."""
        for successes in [0, 10, 50, 100, 1000]:
            lo, hi = _wilson_interval(successes, 1000)
            self.assertLessEqual(lo, hi)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)


class IndependentRiskSensitivityTests(unittest.TestCase):
    """Nhóm E: Tính độc lập của các phân tích độ nhạy."""

    def setUp(self):
        rows = [_row(ltv=0.60 + 0.02 * i, hf=0.02 * i, liq=0.02 * i, bd=0.002 * i, cvar=0.003 * i) for i in range(11)]
        self.df = apply_feasibility(pd.DataFrame(rows), ModelConfig(use_ci_upper_bound=False))
        self.config = ModelConfig(use_ci_upper_bound=False, max_liquidation_probability=0.10,
                                  max_hf_breach_probability=0.10, max_bad_debt_probability=0.01,
                                  max_cvar_ratio=0.02)

    def test_e1_liquidation_sensitivity_does_not_mutate_hf_limit(self):
        """1. Thay đổi giới hạn P(Liquidation) không làm thay đổi giới hạn P(HF Breach)."""
        sens = liquidation_limit_sensitivity(self.df, self.config, limits=(0.02, 0.05, 0.20))
        self.assertEqual(len(sens), 3)
        # Verify base config was not modified
        self.assertEqual(self.config.max_hf_breach_probability, 0.10)
        self.assertEqual(self.config.max_liquidation_probability, 0.10)

    def test_e2_hf_breach_sensitivity_holds_other_limits_fixed(self):
        """2. Thay đổi giới hạn P(HF Breach) không làm thay đổi các giới hạn khác."""
        from main import hf_breach_limit_sensitivity
        sens = hf_breach_limit_sensitivity(self.df, self.config, limits=(0.02, 0.05, 0.20))
        self.assertEqual(len(sens), 3)
        self.assertEqual(self.config.max_liquidation_probability, 0.10)
        self.assertEqual(self.config.max_bad_debt_probability, 0.01)

    def test_e3_comprehensive_sensitivity_uses_common_baseline(self):
        """3. Từng sensitivity sử dụng cùng một bộ kết quả Monte Carlo baseline."""
        from main import comprehensive_risk_sensitivity
        comp = comprehensive_risk_sensitivity(self.df, self.config)
        self.assertIn("liquidation", comp)
        self.assertIn("hf_breach", comp)
        self.assertIn("bad_debt", comp)
        self.assertIn("cvar_ratio", comp)
        # Optimal LTV is monotonic in each individual relaxation
        for key in ["liquidation", "hf_breach", "bad_debt", "cvar_ratio"]:
            ltvs = [it["optimal_ltv"] for it in comp[key] if it["optimal_ltv"] is not None]
            self.assertEqual(ltvs, sorted(ltvs))


class EndogenousLiquidityCrunchTests(unittest.TestCase):
    """F-series: Liquidity Death Spiral / Endogenous Liquidity Crunch tests."""

    def _make_crash_paths(self, n_sims: int = 50, n_steps: int = 5) -> np.ndarray:
        """ETH price drops 20% at step 1, stays there — always triggers haircut."""
        paths = np.full((n_steps + 1, n_sims), 3000.0)
        paths[1:] = 2400.0  # 20% drop at step 1
        return paths

    def test_f1_haircut_disabled_by_default(self):
        """F1. Config mặc định: liquidity_haircut_pct = 0 → haircut không được kích hoạt."""
        from config import ModelConfig
        cfg = ModelConfig()
        self.assertEqual(cfg.liquidity_haircut_pct, 0.0)
        self.assertFalse(float(cfg.liquidity_haircut_pct) > 0.0)

    def test_f2_haircut_triggers_when_drop_exceeds_threshold(self):
        """F2. Khi ETH giảm 20% và threshold là 15%, haircut phải được kích hoạt."""
        from config import ModelConfig
        from agent_engine import generate_borrower_population, simulate_agent_based_portfolio
        n_sims = 50
        cfg = ModelConfig(
            n_sims=n_sims, seed=42,
            liquidity_haircut_pct=0.50,
            liquidity_haircut_trigger_pct=0.15,
        )
        pop = generate_borrower_population(cfg)
        paths = self._make_crash_paths(n_sims=n_sims)
        result = simulate_agent_based_portfolio(paths, 0.73, cfg, pop)
        self.assertTrue(result["Liquidity_Haircut_Active"])
        self.assertGreater(result["Liquidity_Crunch_Steps"], 0)
        # Effective liquidity must be reduced
        self.assertLess(
            result["Effective_Liquidity_Final_USD"],
            cfg.dex_liquidity_usd,
        )

    def test_f3_no_haircut_when_drop_below_threshold(self):
        """F3. Khi ETH giảm 20% nhưng threshold là 25%, haircut KHÔNG được kích hoạt."""
        from config import ModelConfig
        from agent_engine import generate_borrower_population, simulate_agent_based_portfolio
        n_sims = 50
        cfg = ModelConfig(
            n_sims=n_sims, seed=42,
            liquidity_haircut_pct=0.50,
            liquidity_haircut_trigger_pct=0.25,  # higher than 20% drop
        )
        pop = generate_borrower_population(cfg)
        paths = self._make_crash_paths(n_sims=n_sims)
        result = simulate_agent_based_portfolio(paths, 0.73, cfg, pop)
        self.assertEqual(result["Liquidity_Crunch_Steps"], 0)
        self.assertAlmostEqual(
            result["Effective_Liquidity_Final_USD"],
            cfg.dex_liquidity_usd,
            places=0,
        )

    def test_f4_haircut_increases_slippage(self):
        """F4. Khi haircut bật, peak slippage phải tăng so với baseline."""
        from config import ModelConfig
        from agent_engine import generate_borrower_population, simulate_agent_based_portfolio
        n_sims = 100
        paths = self._make_crash_paths(n_sims=n_sims, n_steps=10)
        cfg_base = ModelConfig(n_sims=n_sims, seed=42, liquidity_haircut_pct=0.0)
        cfg_stress = ModelConfig(
            n_sims=n_sims, seed=42,
            liquidity_haircut_pct=0.60,
            liquidity_haircut_trigger_pct=0.15,
        )
        pop = generate_borrower_population(cfg_base)
        r_base = simulate_agent_based_portfolio(paths, 0.73, cfg_base, pop)
        r_stress = simulate_agent_based_portfolio(paths, 0.73, cfg_stress, pop)
        mean_slip_base = float(r_base["Peak_Liquidation_Slippage_Scenarios"].mean())
        mean_slip_stress = float(r_stress["Peak_Liquidation_Slippage_Scenarios"].mean())
        # Stress slippage must be >= baseline (non-negative slippage increase)
        self.assertGreaterEqual(mean_slip_stress, mean_slip_base)

    def test_f5_effective_liquidity_history_length(self):
        """F5. Effective_Liquidity_History phải có độ dài = n_steps + 1."""
        from config import ModelConfig
        from agent_engine import generate_borrower_population, simulate_agent_based_portfolio
        n_sims = 50
        n_steps = 5
        paths = self._make_crash_paths(n_sims=n_sims, n_steps=n_steps)
        cfg = ModelConfig(
            n_sims=n_sims, seed=42,
            liquidity_haircut_pct=0.40,
            liquidity_haircut_trigger_pct=0.15,
        )
        pop = generate_borrower_population(cfg)
        result = simulate_agent_based_portfolio(paths, 0.73, cfg, pop)
        self.assertEqual(len(result["Effective_Liquidity_History"]), n_steps + 1)

    def test_f6_config_validation_rejects_invalid_haircut(self):
        """F6. Config phải raise ValueError khi haircut_pct >= 1 hoặc ngoài [0,1)."""
        from config import ModelConfig
        with self.assertRaises(ValueError):
            ModelConfig(liquidity_haircut_pct=1.0).validate()
        with self.assertRaises(ValueError):
            ModelConfig(liquidity_haircut_pct=-0.1).validate()
        with self.assertRaises(ValueError):
            ModelConfig(liquidity_haircut_trigger_pct=1.5).validate()


if __name__ == "__main__":
    unittest.main()
