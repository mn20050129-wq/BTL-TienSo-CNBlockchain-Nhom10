from dataclasses import dataclass
import math


@dataclass
class ModelConfig:
    # Market simulation
    horizon_days: int = 7
    steps_per_day: int = 24
    n_sims: int = 10_000
    seed: int = 42

    # Borrower-agent population (stylized portfolio)
    n_borrower_agents: int = 10
    borrower_size_sigma: float = 0.45
    min_borrow_utilization: float = 0.80
    max_borrow_utilization: float = 1.00
    borrower_collateral_scale: float = 1.0  # Multiplier for total borrower collateral (enables liquidity/slippage stress tests)

    # Asset architecture: Collateral = ETH, Debt = FLEX, Numeraire = USD
    flex_price_mode: str = "pegged"  # "pegged" (Mode A: 1 FLEX = $1 fixed) or "scenario" (Mode B: stress test depeg)
    flex_initial_price_usd: float = 1.0  # Initial FLEX price in USD
    flex_scenario_shock_pct: float = 0.0  # Immediate percentage shock to FLEX price (+0.10 means FLEX appreciates to $1.10)
    flex_price_source: str = "assumed"  # "assumed" (baseline assumption) or "scenario_stress" / "custom"

    # Lending protocol rules
    liquidation_threshold: float = 0.85
    liquidation_bonus: float = 0.05
    close_factor: float = 0.50
    borrow_apr: float = 0.0  # APR used by the single-rate compatibility wrapper
    borrow_apr_scenarios: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)  # Illustrative annual APR scenarios
    liquidation_gas_cost_usd: float = 1.0  # Set from a measured/assumed gas scenario

    # Oracle staleness proxy, calibrated from recent Chainlink rounds
    oracle_delay_hours: float = 1.0

    # Execution / Uniswap proxy
    # NOTE: dex_liquidity_usd represents a Uniswap V3 pool depth *snapshot* at t=0.
    # It is a proxy for executable depth, not the full reported TVL.
    dex_liquidity_usd: float = 2_000_000.0
    slippage_coefficient: float = 0.10
    slippage_exponent: float = 0.70
    max_slippage: float = 0.50

    # ------------------------------------------------------------------
    # Endogenous Liquidity Crunch / DEX Withdrawal (Liquidity Death Spiral)
    #
    # When ETH price drops sharply intra-day, Uniswap V3 LPs withdraw to
    # avoid Impermanent Loss. This reduces effective pool depth, increasing
    # slippage and making liquidations harder/unprofitable.
    #
    # Mechanism:
    #   - If ETH price drops >= liquidity_haircut_trigger_pct in a single step,
    #     effective_liquidity *= (1 - liquidity_haircut_pct).
    #   - Haircut is applied instantly and persists for the rest of the horizon
    #     (liquidity_haircut_recovery_steps = 0 means no recovery).
    #   - Set liquidity_haircut_pct = 0.0 to disable (baseline / backward-compat).
    #
    # Calibration reference: empirical observation on 2024-08-05 ETH crash
    # (Uniswap V3 ETH/USDC depth fell ~40–60% intra-day during stress).
    # ------------------------------------------------------------------
    liquidity_haircut_trigger_pct: float = 0.15   # ETH intra-step drop threshold (15%)
    liquidity_haircut_pct: float = 0.0            # Fraction of liquidity lost (0 = disabled; 0.4–0.6 = stress)
    liquidity_haircut_recovery_steps: int = 0     # Steps after which liquidity recovers (0 = permanent within horizon)

    # LTV search grid
    min_ltv: float = 0.50
    max_ltv: float = 0.82
    ltv_step: float = 0.01

    # ------------------------------------------------------------------
    # Risk-governance limits used to screen candidate LTVs.
    #
    # IMPORTANT: every number below is a *study assumption* (risk appetite of
    # the protocol), NOT an estimated or observed quantity. They must be
    # justified in the report and stress-tested; see
    # main.liquidation_limit_sensitivity(), which shows how the optimal LTV
    # moves when the liquidation limit changes.
    #
    # A candidate LTV is feasible only if ALL active limits hold.
    # ------------------------------------------------------------------
    max_bad_debt_probability: float = 0.01   # P(Bad Debt) = P(terminal close-out shortfall > 0)
    max_cvar_ratio: float = 0.02             # CVaR99 / initial portfolio debt
    max_liquidation_probability: float = 0.10  # P(at least one executed liquidation in the horizon)
    # P(HF Breach) is the liquidator-independent distress measure: with
    # liquidation_gas_cost_usd > 0 liquidators may refuse to act, which LOWERS
    # P(Liquidation) while borrowers are still unhealthy. Keeping this limit
    # prevents that loophole. Set to None to disable it.
    max_hf_breach_probability: float | None = 0.10

    # Judge probability limits on the Wilson 95% upper bound instead of the
    # point estimate, so a candidate is not accepted just because of Monte
    # Carlo sampling luck. Wilson covers sampling error only.
    use_ci_upper_bound: bool = True

    def validate(self) -> None:
        if self.horizon_days <= 0 or self.steps_per_day <= 0 or self.n_sims <= 0:
            raise ValueError("horizon_days, steps_per_day và n_sims phải > 0.")
        if self.n_borrower_agents <= 0:
            raise ValueError("n_borrower_agents phải > 0.")
        if self.borrower_size_sigma < 0:
            raise ValueError("borrower_size_sigma không được âm.")
        if self.borrower_collateral_scale <= 0 or not math.isfinite(self.borrower_collateral_scale):
            raise ValueError("borrower_collateral_scale phải là số hữu hạn > 0.")
        if self.flex_price_mode not in ("pegged", "scenario"):
            raise ValueError("flex_price_mode phải là 'pegged' hoặc 'scenario'.")
        if not isinstance(self.flex_price_source, str) or not self.flex_price_source.strip():
            raise ValueError("flex_price_source phải là chuỗi ký tự hợp lệ.")
        if self.flex_initial_price_usd <= 0 or not math.isfinite(self.flex_initial_price_usd):
            raise ValueError("flex_initial_price_usd phải là số hữu hạn > 0.")
        if not math.isfinite(self.flex_scenario_shock_pct) or self.flex_scenario_shock_pct <= -1.0:
            raise ValueError("flex_scenario_shock_pct phải là số hữu hạn > -1.0.")
        if not (0 <= self.min_borrow_utilization <= self.max_borrow_utilization <= 1):
            raise ValueError("Borrow utilization phải nằm trong [0, 1].")
        if not (0 < self.liquidation_threshold <= 1):
            raise ValueError("liquidation_threshold phải thuộc (0, 1].")
        if not (0 < self.close_factor <= 1):
            raise ValueError("close_factor phải thuộc (0, 1].")
        if self.liquidation_bonus < 0 or self.borrow_apr < 0 or self.liquidation_gas_cost_usd < 0 or any(rate < 0 for rate in self.borrow_apr_scenarios):
            raise ValueError("liquidation_bonus và borrow APR không được âm.")
        if not math.isfinite(self.borrower_size_sigma) or not math.isfinite(self.oracle_delay_hours):
            raise ValueError("Các tham số phân phối borrower và oracle phải hữu hạn.")
        if self.oracle_delay_hours < 0:
            raise ValueError("oracle_delay_hours không được âm.")
        if not self.borrow_apr_scenarios:
            raise ValueError("Cần ít nhất một kịch bản borrow APR.")
        if self.dex_liquidity_usd <= 0:
            raise ValueError("dex_liquidity_usd phải > 0.")
        if not (0 <= self.slippage_coefficient and 0 < self.slippage_exponent):
            raise ValueError("Slippage coefficient phải >= 0 và exponent phải > 0.")
        if not (0 < self.max_slippage < 1):
            raise ValueError("max_slippage phải thuộc (0, 1).")
        if not (0.0 <= self.liquidity_haircut_trigger_pct <= 1.0):
            raise ValueError("liquidity_haircut_trigger_pct phải thuộc [0, 1].")
        if not (0.0 <= self.liquidity_haircut_pct < 1.0):
            raise ValueError("liquidity_haircut_pct phải thuộc [0, 1).")
        if self.liquidity_haircut_recovery_steps < 0:
            raise ValueError("liquidity_haircut_recovery_steps không được âm.")
        if not (0 < self.min_ltv <= self.max_ltv < 1) or self.ltv_step <= 0:
            raise ValueError("Khoảng LTV không hợp lệ.")
        if self.max_bad_debt_probability < 0 or self.max_bad_debt_probability > 1:
            raise ValueError("max_bad_debt_probability phải thuộc [0, 1].")
        if self.max_cvar_ratio < 0:
            raise ValueError("max_cvar_ratio không được âm.")
        if not (0 <= self.max_liquidation_probability <= 1):
            raise ValueError("max_liquidation_probability phải thuộc [0, 1].")
        if self.max_hf_breach_probability is not None and not (
            0 <= self.max_hf_breach_probability <= 1
        ):
            raise ValueError("max_hf_breach_probability phải là None hoặc thuộc [0, 1].")
