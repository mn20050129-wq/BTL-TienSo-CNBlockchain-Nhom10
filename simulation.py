import numpy as np

from config import ModelConfig


def simulate_merton_jump_diffusion(
    S0: float,
    params: dict,
    config: ModelConfig,
) -> np.ndarray:
    """Simulate downside-jump ETH price paths under a Merton-style process.

    params['mu'] is the annualized log drift of non-jump observations. For a
    Poisson count N, the sum of N iid Normal(mu_j, sigma_j^2) jumps is sampled
    as Normal(N*mu_j, sqrt(N)*sigma_j), which handles multiple jumps correctly.
    """
    config.validate()
    if not np.isfinite(S0) or S0 <= 0:
        raise ValueError("S0 phải là giá dương và hữu hạn.")

    n_sims = int(config.n_sims)
    n_steps = int(config.horizon_days * config.steps_per_day)
    dt = 1.0 / (365.0 * config.steps_per_day)

    mu = float(params["mu_diffusion_log"] if "mu_diffusion_log" in params else params["mu"])
    sigma = float(params["sigma"])
    lam = float(params["lam"])
    mu_j = float(params["mu_j"])
    sigma_j = float(params["sigma_j"])
    if sigma < 0 or lam < 0 or sigma_j < 0:
        raise ValueError("sigma, lambda và sigma_j không được âm.")

    rng = np.random.default_rng(config.seed)
    diffusion = sigma * np.sqrt(dt) * rng.standard_normal((n_steps, n_sims))

    jump_counts = rng.poisson(lam * dt, size=(n_steps, n_sims))
    jump_totals = rng.normal(
        loc=mu_j * jump_counts,
        scale=sigma_j * np.sqrt(jump_counts),
    )

    # mu is calibrated as the mean log return of non-jump observations.
    # Therefore it is already a log-drift: do not subtract 0.5*sigma^2 again,
    # and do not apply an arithmetic-drift jump compensator.
    log_returns = mu * dt + diffusion + jump_totals

    price_paths = np.empty((n_steps + 1, n_sims), dtype=np.float64)
    price_paths[0, :] = float(S0)
    price_paths[1:, :] = S0 * np.exp(np.cumsum(log_returns, axis=0))
    return price_paths
