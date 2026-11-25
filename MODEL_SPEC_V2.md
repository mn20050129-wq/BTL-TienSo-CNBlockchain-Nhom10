# Model specification — V2.1

## 1. Research question and alignment with the original method matrix

The original method matrix places this project under **Collateral Factor**, with the proposed combination **ABM + Monte Carlo using on-chain data**. The stated purpose is to test resilience to extreme collateral-price shocks and estimate tail losses.

V2.0 implements this as a layered model:

1. A downside-jump diffusion process calibrated from historical ETH/USDT returns describes market-price dynamics.
2. Monte Carlo generates the market scenarios.
3. A rule-based agent layer represents heterogeneous borrowers, protocol monitoring/liquidation rules, and profit-seeking liquidators.
4. Chainlink and Uniswap V3 observations calibrate the oracle anchor/update-lag proxy and the pool-liquidity proxy.
5. Risk measures are aggregated over portfolio scenarios and used to screen candidate LTVs.

This is a **stylized rule-based ABM**, not a full protocol replica. Borrowers are heterogeneous in collateral size and utilization but do not actively top up collateral or repay voluntarily. Liquidators share one profitability rule and are not modeled with separate capacity or gas-bidding strategies.

## 2. Market process: downside-jump diffusion

Daily log returns are calculated from completed Binance daily candles:

\[
r_t=\ln(P_t/P_{t-1})
\]

A downside jump is classified when:

\[
r_t<-c\,s_r, \qquad c=3
\]

where `s_r` is the sample standard deviation of all daily log returns. The threshold is a research rule, not an externally verified label of a market crash.

The diffusion parameters are estimated from observations not classified as downside jumps:

\[
\mu_d=365\,\overline{r}_{diff},\qquad
\sigma=\sqrt{365}\,s(r_{diff})
\]

The annual downside-jump intensity and jump moments are:

\[
\lambda=\frac{N_{jump}}{T_{years}},\qquad
\mu_J=\overline{r}_{jump},\qquad
\sigma_J=s(r_{jump})
\]

For one simulation step of length `dt = 1/(365 × steps_per_day)`:

\[
\Delta\ln P_t=\mu_d dt+\sigma\sqrt{dt}Z_t+
\sum_{j=1}^{N_t}Y_{t,j}
\]

\[
Z_t\sim N(0,1),\qquad N_t\sim Poisson(\lambda dt),\qquad
Y_{t,j}\sim N(\mu_J,\sigma_J^2)
\]

The code samples the compound jump sum conditional on the Poisson count:

\[
\sum_{j=1}^{N_t}Y_{t,j}\mid N_t=n
\sim N(n\mu_J,n\sigma_J^2)
\]

Important parameter convention: `mu` is the **annualized mean log drift of non-jump observations**, not the arithmetic price drift. Therefore the simulation uses `mu × dt` directly; it does not subtract `0.5 × sigma²` again or apply a second jump compensator. This is a physical/stress-testing calibration, not risk-neutral option pricing.

## 3. Borrower-agent population

For a candidate protocol LTV, V2.0 creates `n_borrower_agents` positions. Collateral sizes are lognormal and normalized to a cross-agent mean of 1 ETH:

\[
C_i=\frac{\widetilde C_i}{mean(\widetilde C)},\qquad
\widetilde C_i\sim LogNormal(-s_C^2/2,s_C^2)
\]

Borrow utilization is sampled as:

\[
u_i\sim Uniform(u_{min},u_{max})
\]

Initial debt per borrower is:

\[
D_{i,0}=C_i P_0 LTV\,u_i
\]

The same borrower population and Monte Carlo price paths are reused across all LTV and APR candidates (common random numbers), making comparisons less noisy.

## 4. Oracle, protocol and liquidator rules

The current Chainlink price anchors the oracle series. The simulated Binance relative price return is applied to that anchor. The lag `delta` is set to the P90 of recent Chainlink round-update intervals and interpolated in log-price space:

\[
P^{oracle}_t=P^{Chainlink}_0\times
\frac{P^{market}_{t-\delta}}{P^{market}_0}
\]

This is a constant-lag proxy, not a replay of future Chainlink updates.

Health Factor:

\[
HF_{i,t}=\frac{C_{i,t}P^{oracle}_t LT}{D_{i,t}}
\]

A position becomes eligible for liquidation when `HF < 1`. The protocol close-factor rule requests repayment of:

\[
R_i=CF\times D_i
\]

The collateral seized, capped by remaining collateral, is:

\[
Q_i=\min\left(C_i,\frac{R_i(1+LB)}{P^{oracle}_t}\right)
\]

The actual repayment cannot exceed the value of collateral seized under the oracle/bonus rule.

Liquidators sell the seized ETH at the market price. Orders in the same scenario and time step are aggregated, so they share a slippage estimate:

\[
slippage_t=\min\left(k\left(\frac{Notional_t}{LiquidityProxy}\right)^\alpha,
slippage_{max}\right)
\]

The liquidator executes only when:

\[
Q_iP^{market}_t(1-slippage_t)-R_i-GasCost>0
\]

This creates the basic interaction channel: more simultaneous liquidations increase aggregate notional, which increases slippage and can make marginal liquidations unprofitable.

## 5. Interest accrual

For annual continuous APR `r`:

\[
D_{i,t+dt}=D_{i,t}e^{r dt}
\]

The tested APR values are illustrative scenarios in `config.py` (0%, 5%, 10%, 20%). They are not represented as observed market lending rates. The same price paths and borrower population are used across APR scenarios.

## 6. Loss measurement and LTV selection

At the end of the horizon, the model applies a **horizon-closeout assumption**: remaining collateral is assumed sold at the final market price after a terminal portfolio slippage proxy. Bad debt for borrower `i` is:

\[
BD_i=\max(D_{i,T}-Recovery_{i,T},0)
\]

Portfolio loss in scenario `s` is the sum of borrower bad debts. This is a modeled closeout shortfall, not an observed on-chain default or a claim that a live protocol must liquidate every remaining position at the horizon. Reported metrics are:

- `P(HF Breach)`: share of scenarios in which at least one borrower has `HF < 1`.
- `P(Liquidation)`: share of scenarios with at least one executed liquidation.
- `P(Bad Debt)`: share of scenarios with positive portfolio bad debt.
- `Expected Loss`: mean portfolio bad debt in USD.
- `VaR99`: empirical 99th percentile of portfolio bad debt.
- `CVaR99`: mean loss over exactly the upper 1% tail mass (fractional boundary observation handled explicitly).
- `CVaR99/InitialDebt`: CVaR99 divided by initial portfolio debt.
- Wilson 95% confidence intervals for the HF-breach, executed-liquidation and bad-debt probabilities. These quantify finite Monte Carlo sampling error only.

A candidate LTV is feasible only when **all** active limits hold (V2.2):

\[
P(HFBreach)\leq L_{HF},\quad P(Liquidation)\leq L_{liq},\quad
P(BadDebt)\leq L_{BD},\quad CVaR_{99}/InitialDebt\leq L_{CVaR}
\]

- Probabilities are compared with the **Wilson 95% upper bound** (`use_ci_upper_bound=True`), so an LTV is not accepted because of Monte Carlo sampling luck. With zero observed events the upper bound is still `z²/(n+z²)`, hence a limit `p` is reachable only if `n_sims ≥ z²(1−p)/p` (≈ 381 for 1%). `main.py` prints a warning otherwise.
- `CVaR99/InitialDebt` is a point estimate (it rests on ~100 tail scenarios at 10,000 paths; a bootstrap CI is not yet computed).
- `P(HF Breach)` is liquidator-independent. `P(Liquidation)` alone can be gamed: with `liquidation_gas_cost_usd > 0` liquidators may refuse to act, which *lowers* `P(Liquidation)` while borrowers are still unhealthy. Both limits are therefore kept; `max_hf_breach_probability=None` disables the HF limit.
- Both probabilities are **scenario-level**: "at least one of the `n_borrower_agents` positions". They are not the share of borrowers liquidated.

The optimal LTV is the **largest feasible LTV** for each APR scenario. If no LTV is feasible the result is `None` / `NO_FEASIBLE_LTV`; no "closest" LTV is substituted. The output also reports the next grid LTV and the constraint(s) that rejected it, flags an optimum sitting on the grid's upper edge (censored result) and flags non-monotone feasibility.

| Parameter | Status | Basis |
|---|---|---|
| `max_bad_debt_probability` 1%, `max_cvar_ratio` 2% | study assumption | inherited from V2.0/V2.1; must be justified |
| `max_liquidation_probability` 10%, `max_hf_breach_probability` 10% | **placeholder assumption** | risk appetite, not estimated from data; `main.liquidation_limit_sensitivity()` shows how the optimum moves for 2%–20% |

## 7. Data and model limitations to disclose

1. Binance ETH/USDT is used as a proxy for ETH/USD. The initial price basis gap with Chainlink ETH/USD is calculated and warned when it exceeds 1%.
2. A three-year daily sample may contain few downside jumps. If jump count is small, `lambda`, `mu_J` and `sigma_J` are statistically uncertain; the program warns rather than silently injecting artificial jumps.
3. The 3-sigma downside-jump classification and normal jump-size distribution are modeling assumptions.
4. Chainlink P90 update interval is only a historical cadence proxy; the simulation does not forecast future deviation-triggered update timing.
5. Uniswap V3 token balances produce a pool-balance TVL proxy. They do not measure executable depth at a specific trade size. A proper V3 quote/tick-walk engine remains a next-stage enhancement.
6. Gas cost defaults to zero unless configured. This can overstate liquidator participation during congestion.
7. Borrowers do not voluntarily repay or add collateral; liquidation agents have a common rule and unlimited number of concurrent decisions. These simplify real protocol and keeper behavior.
8. All borrower positions use the same ETH market path; there is no collateral-asset diversification or idiosyncratic borrower default process.
9. APR is an illustrative parameter. The 7-day horizon means even a 20% annual APR adds relatively little debt compared with a severe ETH price shock.
10. Monte Carlo results are conditional on the calibration window and assumptions. Confidence intervals, seed sensitivity, alternative jump thresholds, and out-of-sample backtesting should be added before making strong claims about predictive accuracy.

11. **Slippage channel is dormant at the default portfolio scale.** The default portfolio is 10 borrowers of ~1 ETH (≈ $25k of debt) against a pool-liquidity proxy of millions of USD. Interim-liquidation slippage peaks at a fraction of a percent, far below the 5% liquidation bonus, so liquidators remain profitable and pool depth has almost no effect on P(Liquidation), Bad Debt or the optimal LTV. It only matters when aggregate sale flow approaches ~0.3–0.4× the liquidity proxy (with k=0.10, α=0.70). Any liquidity stress test must set the portfolio scale explicitly; `main.py` prints the observed peak slippage so this is visible in every run.
12. **One-pass executable-liquidation set.** Liquidators are selected by pricing the whole eligible flow once, dropping unprofitable orders and re-pricing once. A fixed-point solution can admit marginally more orders. On mock inputs the difference is nil in the base case and at most ~0.6 pp of P(Liquidation) / ~8% of expected loss in a harsh case (6h oracle lag, 200k liquidity, $100 gas).
13. **Gas is portfolio-scale dependent.** A fixed USD gas cost is compared with a liquidation profit of order 5% × 50% × position size; with ~$3.5k positions a $50–500 gas cost can switch liquidators off entirely. Gas results are meaningful only together with an explicit position scale.

## V2.1 implementation notes

- `Expected Slippage Cost` includes interim executed liquidations and slippage under the assumed terminal closeout sale.
- Additional validation rejects invalid borrower, oracle-delay, slippage and risk-limit inputs.

## V2.3 loss decomposition and diagnostics

Bad Debt in scenario `s` is split by origin:

\[
BD_s=\underbrace{\sum_{i:\,C_{i,T}=0}BD_{i}}_{Realized}+\underbrace{\sum_{i:\,C_{i,T}>0}BD_{i}}_{Closeout}
\]

- **Realized**: the borrower's collateral was fully seized during the horizon and debt remains. This loss is independent of the closeout assumption.
- **Closeout**: the borrower still holds collateral and the shortfall exists only because all remaining collateral is assumed sold at the horizon.
- `main.py` reports `P(Realized Shortfall)`, `Expected Realized Loss` and `Closeout Share of Expected Loss`. They are diagnostics: the feasibility test still uses `P(Bad Debt)` and `CVaR99` exactly as before.
- On mock inputs the closeout share is ≤ ~2% in the base case (so the horizon-closeout assumption barely drives the result) but reaches ~100% when liquidators do not act (e.g. high gas), where the unliquidated unhealthy positions are only captured by the closeout. Re-check the share on real inputs before claiming either way.
- `Peak Liquidation Slippage` is the highest pool-impact estimate applied to an interim liquidation batch (max over scenarios). Slippage on interim liquidations is borne by the liquidator; terminal closeout slippage reduces protocol recovery. `Expected Slippage Cost` adds both and is a market-impact indicator, not a protocol loss.
