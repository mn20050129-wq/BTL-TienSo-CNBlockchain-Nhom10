"""Diagnostic script to test network, RPC and Binance connectivity."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def run_diagnostics():
    print("=" * 70)
    print("KIỂM TRA KẾT NỐI MẠNG, ETHEREUM RPC & DỮ LIỆU THỊ TRƯỜNG")
    print("=" * 70)

    # 1. Check web3
    print("\n1. Kiểm tra thư viện Web3:")
    try:
        from web3 import Web3
        print("   [OK] Package web3 đã được cài đặt.")
    except ImportError:
        print("   [LỖI] Chưa cài package web3! Hãy chạy: pip install web3")
        return

    # 2. Check .env
    print("\n2. Kiểm tra file .env:")
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        print(f"   [LỖI] Không tìm thấy file .env tại: {env_path}")
        return
    
    load_dotenv(env_path, override=True)
    alchemy_url = os.getenv("ALCHEMY_URL", "").strip()
    print(f"   [OK] Tìm thấy file .env.")
    if not alchemy_url:
        print("   [LỖI] ALCHEMY_URL trong .env đang để trống!")
        return
    # Mask key for privacy
    masked_url = alchemy_url[:35] + "..." if len(alchemy_url) > 35 else alchemy_url
    print(f"   [OK] ALCHEMY_URL = {masked_url}")

    # 3. Test Web3 connection to Alchemy
    print("\n3. Kiểm tra kết nối Ethereum Mainnet RPC:")
    w3 = Web3(Web3.HTTPProvider(alchemy_url, request_kwargs={"timeout": 15}))
    try:
        is_connected = w3.is_connected()
        if is_connected:
            block = w3.eth.block_number
            print(f"   [OK] Kết nối RPC THÀNH CÔNG! Block hiện tại: #{block:,}")
        else:
            print("   [LỖI] w3.is_connected() trả về False.")
            print("   -> Nguyên nhân có thể do URL Alchemy không đúng, API key chưa kích hoạt, hoặc mạng chặn.")
    except Exception as e:
        print(f"   [LỖI KẾT NỐI] {type(e).__name__}: {e}")

    # 4. Test Chainlink & Uniswap functions
    print("\n4. Kiểm tra lấy dữ liệu Smart Contract on-chain:")
    try:
        from onchain_engine import fetch_chainlink_oracle_data, fetch_uniswap_liquidity
        oracle = fetch_chainlink_oracle_data(sample_rounds=5)
        print(f"   [OK] Chainlink Oracle ETH/USD: ${oracle['oracle_price']:,.2f}")
        print(f"   [OK] Oracle age: {oracle['oracle_age_seconds']} giây")
        dex = fetch_uniswap_liquidity(oracle["oracle_price"])
        print(f"   [OK] Uniswap V3 TVL proxy: ${dex['liquidity_usd']:,.2f}")
    except Exception as e:
        print(f"   [LỖI ON-CHAIN] {type(e).__name__}: {e}")

    # 5. Test Binance API
    print("\n5. Kiểm tra kết nối Binance Market Data:")
    import requests
    mirrors = [
        "https://data-api.binance.vision",
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
    ]
    binance_ok = False
    for url in mirrors:
        try:
            res = requests.get(f"{url}/api/v3/ticker/price", params={"symbol": "ETHUSDT"}, timeout=5)
            if res.status_code == 200:
                price = float(res.json()["price"])
                print(f"   [OK] Kết nối Binance thành công qua {url}! Giá ETH: ${price:,.2f}")
                binance_ok = True
                break
        except Exception as e:
            print(f"   [THẤT BẠI] Không thể kết nối {url}: {type(e).__name__}")
    
    if not binance_ok:
        print("   [CẢNH BÁO] Tất cả mirror Binance đều không thể truy cập (thường do nhà mạng tại VN chặn).")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_diagnostics()
