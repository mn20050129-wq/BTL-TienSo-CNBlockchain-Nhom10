from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


BINANCE_MIRRORS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]


def fetch_realtime_price(symbol: str = "ETHUSDT") -> float:
    """Fetch the latest Binance spot price trying multiple official mirror domains."""
    last_err = None
    for base_url in BINANCE_MIRRORS:
        try:
            response = requests.get(
                f"{base_url}/api/v3/ticker/price",
                params={"symbol": symbol},
                timeout=10,
            )
            response.raise_for_status()
            price = float(response.json()["price"])
            if np.isfinite(price) and price > 0:
                return price
        except Exception as e:
            last_err = e
            continue
    raise ConnectionError(f"Không thể kết nối API Binance qua các mirror. Lỗi: {last_err}")


def fetch_historical_data(symbol: str = "ETHUSDT", days: int = 1095) -> pd.DataFrame:
    """Fetch daily candles and exclude the still-open daily candle."""
    now = datetime.now(timezone.utc)
    end_time = int(now.timestamp() * 1000)
    start_time = int((now - timedelta(days=days + 2)).timestamp() * 1000)

    working_base_url = None
    for base_url in BINANCE_MIRRORS:
        try:
            test_res = requests.get(
                f"{base_url}/api/v3/klines",
                params={"symbol": symbol, "interval": "1d", "limit": 1},
                timeout=10,
            )
            if test_res.status_code == 200:
                working_base_url = base_url
                break
        except Exception:
            continue

    if not working_base_url:
        raise ConnectionError("Không thể kết nối lấy nến lịch sử Binance qua các mirror.")

    url = f"{working_base_url}/api/v3/klines"
    all_klines = []
    current_start = start_time

    print(f"Đang kéo dữ liệu lịch sử từ Binance ({working_base_url})...")
    while current_start < end_time:
        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000,
            },
            timeout=20,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        all_klines.extend(batch)
        next_start = int(batch[-1][0]) + 1
        if next_start <= current_start:
            raise RuntimeError("Binance pagination không tiến về phía trước.")
        current_start = next_start

    if not all_klines:
        raise RuntimeError("Không nhận được nến lịch sử từ Binance.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qav", "num_trades", "tbbv", "tbqv", "ignore",
    ]
    df = pd.DataFrame(all_klines, columns=columns)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

    # Prevent a partial current-day candle from contaminating calibration.
    df = df[df["close_time"] < end_time].copy()
    df = df.dropna(subset=["close"]).sort_values("open_time")
    df = df.drop_duplicates(subset=["open_time"], keep="last")
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return"])
    if len(df) < 100:
        raise RuntimeError(f"Chỉ có {len(df)} quan sát hợp lệ; cần tối thiểu 100.")
    return df.reset_index(drop=True)


def calibrate_merton_parameters(
    df: pd.DataFrame,
    jump_threshold_sigma: float = 3.0,
) -> dict:
    """Calibrate an asymmetric downside-jump diffusion from daily log returns.

    mu is the annualized mean log return on non-downside-jump days. The jump
    component is added separately, so no Merton arithmetic-drift compensator
    is applied in simulation (avoids counting historical jumps twice).
    """
    if "log_return" not in df:
        raise KeyError("DataFrame phải có cột log_return.")

    returns = np.asarray(df["log_return"].dropna(), dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 100:
        raise ValueError("Không đủ dữ liệu lợi suất để hiệu chỉnh Merton.")
    if jump_threshold_sigma <= 0:
        raise ValueError("jump_threshold_sigma phải > 0.")

    daily_sigma_all = float(np.std(returns, ddof=1))
    threshold = jump_threshold_sigma * daily_sigma_all
    jump_mask = returns < -threshold
    negative_jumps = returns[jump_mask]
    diffusion_returns = returns[~jump_mask]

    if diffusion_returns.size < 30:
        raise ValueError("Quá ít quan sát diffusion sau khi tách downside jumps.")

    mu_log_annual = float(np.mean(diffusion_returns) * 365.0)
    sigma_annual = float(np.std(diffusion_returns, ddof=1) * np.sqrt(365.0))
    observation_years = returns.size / 365.0
    jump_intensity = float(negative_jumps.size / observation_years)

    if negative_jumps.size:
        jump_mean = float(np.mean(negative_jumps))
        # With one observed jump, its sample dispersion cannot be estimated.
        jump_std = float(np.std(negative_jumps, ddof=1)) if negative_jumps.size > 1 else 0.0
    else:
        jump_mean, jump_std = 0.0, 0.0

    return {
        "mu": mu_log_annual,
        "mu_diffusion_log": mu_log_annual,
        "sigma": max(sigma_annual, 1e-8),
        "lam": jump_intensity,
        "mu_j": jump_mean,
        "sigma_j": max(jump_std, 0.0),
        "jump_count": int(negative_jumps.size),
        "observations": int(returns.size),
        "jump_threshold_sigma": float(jump_threshold_sigma),
        "jump_threshold_daily": float(threshold),
        "calibration_method": "downside-only threshold jumps; non-jump log-return diffusion",
    }
