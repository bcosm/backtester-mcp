"""Generate dataset files for the repo.

Downloads SPY/BTC data and generates synthetic prediction market data.
Run once, commit the parquet files.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def generate_spy():
    """Download SPY daily data using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — generating synthetic SPY data instead")
        return _synthetic_equity("SPY", 2520, 200.0)

    df = yf.download("SPY", start="2015-01-01", end="2025-12-31", progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df


def generate_btc():
    """Download BTC-USD hourly data using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — generating synthetic BTC data instead")
        return _synthetic_equity("BTC", 8760, 30000.0)

    # yfinance only provides ~730 days of hourly data
    df = yf.download("BTC-USD", period="730d", interval="1h", progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"
    return df


def _synthetic_equity(name: str, n: int, start_price: float) -> pd.DataFrame:
    """Fallback: geometric Brownian motion."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0003, 0.015, n)
    prices = start_price * np.cumprod(1 + returns)

    high = prices * (1 + rng.uniform(0, 0.01, n))
    low = prices * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 10_000_000, n)

    dates = pd.bdate_range(start="2015-01-02", periods=n)
    return pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.002, n)),
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume,
    }, index=pd.Index(dates, name="date"))


def generate_kalshi():
    """Generate synthetic prediction market binary contract data.

    Simulates a contract that resolves to 0 or 1, with price mean-reverting
    toward the eventual outcome as expiration approaches.
    """
    rng = np.random.default_rng(123)
    n_contracts = 5
    rows = []

    for contract_id in range(n_contracts):
        n_hours = rng.integers(100, 500)
        outcome = rng.choice([0.0, 1.0])

        # start near 0.5, drift toward outcome with noise
        price = 0.5
        for t in range(n_hours):
            progress = t / n_hours
            # pull toward outcome more strongly as expiration nears
            pull = 0.01 + 0.05 * progress
            price += pull * (outcome - price) + rng.normal(0, 0.03)
            price = np.clip(price, 0.01, 0.99)

            volume = int(rng.exponential(500) + 10)
            rows.append({
                "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=t),
                "price": round(price, 4),
                "volume": volume,
            })

    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def main():
    DATASETS_DIR.mkdir(exist_ok=True)

    print("generating SPY daily data...")
    spy = generate_spy()
    spy.to_parquet(DATASETS_DIR / "spy_daily.parquet")
    print(f"  saved {len(spy)} rows")

    print("generating BTC hourly data...")
    btc = generate_btc()
    btc.to_parquet(DATASETS_DIR / "btc_hourly.parquet")
    print(f"  saved {len(btc)} rows")

    print("generating synthetic Kalshi data...")
    kalshi = generate_kalshi()
    kalshi.to_parquet(DATASETS_DIR / "synthetic_kalshi.parquet")
    print(f"  saved {len(kalshi)} rows")

    print("done.")


if __name__ == "__main__":
    main()
