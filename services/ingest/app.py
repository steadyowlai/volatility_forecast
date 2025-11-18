import os
import pandas as pd
import yfinance as yf
from pathlib import Path

from libs.schemas import raw_market_schema, curated_market_daily_schema

# Configuration
DATA_RAW = Path("data/raw.market")
DATA_CURATED = Path("data/curated.market")

# Configurable lookback period (can override via environment variable)
# 10 years = ~2,500 trading days, optimal for XGBoost training
LOOKBACK_PERIOD = os.getenv("LOOKBACK_PERIOD", "10y")

TICKERS = [
    "SPY",     # S&P 500 ETF (equity benchmark)
    "^VIX",    # CBOE Volatility Index (30-day implied vol)
    "^VIX3M",  # 3-month VIX (90-day implied vol)
    "TLT",     # Long-term US Treasury bond ETF (~20-year duration)
    "HYG",     # High-yield corporate bond ETF (credit risk)
]

def download_one(symbol: str, period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    """
    Download one symbol from yfinance and return a normalized DF with the columns:
    date, open, high, low, close, adj_close, volume, symbol
    
    Args:
        symbol: Ticker symbol to download
        period: Lookback period (default: 10y). Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    TODO (Level 2): Implement incremental updates
        - Check last date in existing data
        - Only download new data since last date
        - Add --force-refresh flag to re-download everything
        - This will improve efficiency for daily production runs
    """
    
    df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
    
    if df.empty:
        raise RuntimeError(f"No data downloaded for {symbol}")
    
    # yfinance returns MultiIndex columns when downloading a single ticker
    # Flatten the column names: ('Close', 'SPY') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.rename_axis("date").reset_index()  # renames the index to 'date' and resets index to make 'date' a column
    df = df.rename(columns=str.lower) # lowercase column names
    
    # yfinance gives "adj close" for equities; for indices like VIX, VIX3M, it gives no adj close
    if "adj close" not in df.columns:
        df["adj close"] = df["close"]
    
    df["symbol"] = symbol
    
    expected_cols = ["symbol", "date", "open", "high", "low", "close", "adj close", "volume"]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns {missing} for symbol {symbol}")
    
    return df[expected_cols]
  
def build_raw(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build raw.market view:
    symbol, date, open, high, low, close, volume
    """
    
    raw = df_all[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["volume"] = raw["volume"].astype(float) # to be schema-compliant
    
    return raw
  
def build_curated(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build curated.market.daily view:
    symbol, date, close, ret, adj_close
    where ret is daily return (null for first day of each symbol)
    """
    
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["adj close"]
    
    df = df.sort_values(["symbol", "date"])
    df["ret"] = df.groupby("symbol")["adj_close"].pct_change()
    curated = df[["symbol", "date", "close", "ret", "adj_close"]].copy()
    
    return curated
  
def write_partitions(df: pd.DataFrame, root: Path, fname: str) -> None:
    """
    Write df patitioned by date:
    root/date-=YYYY-MM-DD/fname.parquet
    """
    root.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    for d, g in df.groupby(df["date"].dt.strftime("%Y-%m-%d")):
        outdir = root / f"date={d}"
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{fname}.parquet "
        g.to_parquet(outpath, index=False)


def main():
    print(f"Downloading market data (period={LOOKBACK_PERIOD})...")
    frames = [download_one(t) for t in TICKERS]
    all_data = pd.concat(frames, ignore_index=True)
    
    print(f"Downloaded {len(all_data)} total rows across {len(TICKERS)} tickers")
    
    print("Building raw.market...")
    raw = build_raw(all_data)
    
    print("Validating raw.market schema...")
    raw_market_schema.validate(raw)
    
    print("Writing raw.market partitions...")
    write_partitions(raw, DATA_RAW, "raw.market")
    
    print("Building curated.market.daily...")
    curated = build_curated(all_data)
    
    print("Validating curated.market.daily schema...")
    curated_market_daily_schema.validate(curated)
    
    print("Writing curated.market.daily partitions...")
    write_partitions(curated, DATA_CURATED, "daily")
    
    print("Ingestion complete.")
    
if __name__ == "__main__":
    main()
  