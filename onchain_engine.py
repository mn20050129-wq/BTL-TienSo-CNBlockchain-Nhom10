import json
import os
import time
from pathlib import Path
from statistics import median

import numpy as np
from dotenv import load_dotenv
try:
    from web3 import Web3
except ImportError:  # Allow offline model tests without the optional RPC package.
    Web3 = None

load_dotenv(Path(__file__).with_name(".env"))
ALCHEMY_URL = os.getenv("ALCHEMY_URL", "").strip()
_w3 = None

CHAINLINK_ETH_USD_ADDRESS = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
UNI_V3_ETH_USDC_POOL = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"

CHAINLINK_ABI = json.loads(r'''[
  {"inputs":[],"name":"latestRoundData","outputs":[
    {"internalType":"uint80","name":"roundId","type":"uint80"},
    {"internalType":"int256","name":"answer","type":"int256"},
    {"internalType":"uint256","name":"startedAt","type":"uint256"},
    {"internalType":"uint256","name":"updatedAt","type":"uint256"},
    {"internalType":"uint80","name":"answeredInRound","type":"uint80"}],
    "stateMutability":"view","type":"function"},
  {"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[
    {"internalType":"uint80","name":"roundId","type":"uint80"},
    {"internalType":"int256","name":"answer","type":"int256"},
    {"internalType":"uint256","name":"startedAt","type":"uint256"},
    {"internalType":"uint256","name":"updatedAt","type":"uint256"},
    {"internalType":"uint80","name":"answeredInRound","type":"uint80"}],
    "stateMutability":"view","type":"function"}
]''')

ERC20_ABI = json.loads(r'''[
  {"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},
  {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}
]''')

UNISWAP_V3_POOL_ABI = json.loads(r'''[
  {"inputs":[],"name":"slot0","outputs":[
    {"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},
    {"internalType":"int24","name":"tick","type":"int24"},
    {"internalType":"uint16","name":"observationIndex","type":"uint16"},
    {"internalType":"uint16","name":"observationCardinality","type":"uint16"},
    {"internalType":"uint16","name":"observationCardinalityNext","type":"uint16"},
    {"internalType":"uint8","name":"feeProtocol","type":"uint8"},
    {"internalType":"bool","name":"unlocked","type":"bool"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"liquidity","outputs":[{"internalType":"uint128","name":"","type":"uint128"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"token0","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"token1","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"}
]''')


FALLBACK_RPCS = [
    "https://eth.llamarpc.com",
    "https://ethereum.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://cloudflare-eth.com",
]


def _get_web3() -> Web3:
    global _w3
    if Web3 is None:
        raise RuntimeError("Thiếu package web3. Cài dependencies bằng: python -m pip install -r requirements.txt")
    
    # Reload .env dynamically so runtime changes in .env are picked up immediately
    load_dotenv(Path(__file__).with_name(".env"), override=True)
    custom_url = os.getenv("ALCHEMY_URL", "").strip()

    urls_to_try = []
    if custom_url:
        urls_to_try.append(custom_url)
    urls_to_try.extend(FALLBACK_RPCS)

    if _w3 is not None:
        try:
            if _w3.is_connected():
                return _w3
        except Exception:
            _w3 = None

    last_error = None
    for url in urls_to_try:
        try:
            candidate_w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 15}))
            if candidate_w3.is_connected():
                _w3 = candidate_w3
                if url != custom_url and custom_url:
                    print(f"CẢNH BÁO: Không thể kết nối ALCHEMY_URL ({custom_url[:35]}...), đang dùng RPC dự phòng: {url}")
                return _w3
        except Exception as e:
            last_error = e
            continue

    raise ConnectionError(
        f"Không thể kết nối tới Ethereum RPC (đã thử cả Alchemy và các RPC công khai). "
        f"Vui lòng kiểm tra lại kết nối mạng Internet. Chi tiết lỗi: {last_error}"
    )


def fetch_chainlink_oracle_data(sample_rounds: int = 40) -> dict:
    """Fetch latest ETH/USD and summarize recent Chainlink update intervals.

    oracle_age is the observed age of the latest round. Median/P90 intervals
    are historical cadence proxies; neither is a guarantee of future delay.
    """
    w3 = _get_web3()
    contract = w3.eth.contract(address=Web3.to_checksum_address(CHAINLINK_ETH_USD_ADDRESS), abi=CHAINLINK_ABI)
    latest = contract.functions.latestRoundData().call()
    round_id, answer, _, updated_at, answered_in_round = latest
    if answer <= 0 or updated_at <= 0:
        raise ValueError("Chainlink trả về giá hoặc timestamp không hợp lệ.")
    if answered_in_round < round_id:
        raise ValueError("Chainlink round chưa được xác nhận đầy đủ.")

    now = int(time.time())
    oracle_age_seconds = max(0, now - int(updated_at))
    timestamps = [int(updated_at)]
    for offset in range(1, max(2, int(sample_rounds))):
        previous_id = int(round_id) - offset
        if previous_id <= 0:
            break
        try:
            row = contract.functions.getRoundData(previous_id).call()
            ts = int(row[3])
            if ts > 0:
                timestamps.append(ts)
        except Exception:
            continue

    timestamps = sorted(set(timestamps), reverse=True)
    intervals = np.asarray(
        [timestamps[i] - timestamps[i + 1] for i in range(len(timestamps) - 1)
         if timestamps[i] > timestamps[i + 1]],
        dtype=float,
    )
    if intervals.size:
        median_interval = float(np.median(intervals))
        p90_interval = float(np.quantile(intervals, 0.90))
    else:
        median_interval = float(oracle_age_seconds)
        p90_interval = float(oracle_age_seconds)

    return {
        "oracle_price": float(answer) / 1e8,
        "round_id": int(round_id),
        "updated_at": int(updated_at),
        "oracle_age_seconds": oracle_age_seconds,
        "oracle_age_hours": oracle_age_seconds / 3600.0,
        "oracle_update_interval_seconds": median_interval,
        "oracle_update_interval_hours": median_interval / 3600.0,
        "oracle_p90_update_interval_seconds": p90_interval,
        "oracle_p90_update_interval_hours": p90_interval / 3600.0,
        "sampled_rounds": len(timestamps),
        "delay_is_historical_proxy": True,
    }


def fetch_uniswap_liquidity(oracle_price: float) -> dict:
    """Read V3 pool state and report pool-balance TVL as a coarse proxy.

    Token balances held by a Uniswap V3 pool are NOT the same as executable
    depth at a chosen trade size. True V3 price impact requires a swap quote or
    tick-by-tick liquidity walk; this function intentionally labels the result
    as a TVL proxy so the risk engine does not overstate its precision.
    """
    w3 = _get_web3()
    pool_address = Web3.to_checksum_address(UNI_V3_ETH_USDC_POOL)
    pool = w3.eth.contract(address=pool_address, abi=UNISWAP_V3_POOL_ABI)
    token0 = Web3.to_checksum_address(pool.functions.token0().call())
    token1 = Web3.to_checksum_address(pool.functions.token1().call())
    expected_pair = {USDC_ADDRESS.lower(), WETH_ADDRESS.lower()}
    if {token0.lower(), token1.lower()} != expected_pair:
        raise ValueError("Pool Uniswap trả về không đúng cặp USDC/WETH dự kiến.")

    slot0 = pool.functions.slot0().call()
    active_liquidity = int(pool.functions.liquidity().call())

    balances = {}
    for address, symbol in [(USDC_ADDRESS, "USDC"), (WETH_ADDRESS, "WETH")]:
        token = w3.eth.contract(address=Web3.to_checksum_address(address), abi=ERC20_ABI)
        decimals = int(token.functions.decimals().call())
        raw_balance = int(token.functions.balanceOf(UNI_V3_ETH_USDC_POOL).call())
        balances[symbol] = raw_balance / (10 ** decimals)

    usdc_value = balances["USDC"]
    weth_value = balances["WETH"] * float(oracle_price)
    tvl_proxy = usdc_value + weth_value
    if not np.isfinite(tvl_proxy) or tvl_proxy <= 0:
        raise ValueError("Pool TVL proxy không hợp lệ.")

    return {
        "pool_address": pool_address,
        "token0": token0,
        "token1": token1,
        "sqrt_price_x96": int(slot0[0]),
        "tick": int(slot0[1]),
        "active_liquidity": active_liquidity,
        "usdc_balance": balances["USDC"],
        "weth_balance": balances["WETH"],
        "liquidity_usd": float(tvl_proxy),
        "liquidity_measure": "pool token-balance TVL proxy; not executable depth",
    }


if __name__ == "__main__":
    oracle = fetch_chainlink_oracle_data()
    print(f"Chainlink ETH/USD: ${oracle['oracle_price']:,.2f}")
    print(f"Latest oracle age: {oracle['oracle_age_seconds']} sec")
    print(f"P90 update interval proxy: {oracle['oracle_p90_update_interval_hours']:.4f} h")
    dex = fetch_uniswap_liquidity(oracle["oracle_price"])
    print(f"Uniswap V3 pool-balance TVL proxy: ${dex['liquidity_usd']:,.2f}")
    print(f"Active liquidity scalar: {dex['active_liquidity']:,}")
