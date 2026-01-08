import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

from libs.schemas import raw_market_schema, curated_market_daily_schema

# Configuration
DATA_RAW = Path("data/raw.market")
DATA_CURATED = Path("data/curated.market")

# Configurable start date (can override via environment variable)
# Default: 2010-01-01 - Post-financial crisis "new normal" regime
# ~3,770 trading days (15 years), captures multiple market cycles
# Excludes 2008-2009 crisis (structural break in market dynamics)
START_DATE = os.getenv("START_DATE", "2010-01-01")

TICKERS = [
    "SPY",     # S&P 500 ETF (equity benchmark)
    "^VIX",    # CBOE Volatility Index (30-day implied vol)
    "^VIX3M",  # 3-month VIX (90-day implied vol)
    "TLT",     # Long-term US Treasury bond ETF (~20-year duration)
    "HYG",     # High-yield corporate bond ETF (credit risk)
]

def get_existing_dates(data_dir: Path) -> list:
    """
    get list of dates we already have data for
    checks partition folders like date=2010-01-04
    """
    if not data_dir.exists():
        return []
    
    dates = []
    for date_folder in data_dir.iterdir():
        if date_folder.is_dir() and date_folder.name.startswith("date="):
            date_str = date_folder.name.replace("date=", "")
            try:
                dates.append(pd.to_datetime(date_str).date())
            except:
                pass
    
    return sorted(dates)

def determine_download_range(data_dir: Path, default_start: str) -> tuple:
    """
    figure out what date range to download
    
    returns: (start_date, end_date, is_incremental)
    """
    existing_dates = get_existing_dates(data_dir)
    
    end_date = datetime.now().date()
    
    if not existing_dates:
        #no existing data, download from default start
        start_date = pd.to_datetime(default_start).date()
        print(f"no existing data found")
        print(f"will download full history from {start_date}")
        return start_date, end_date, False
    
    #we have existing data, download only new stuff
    last_date = max(existing_dates)
    start_date = last_date + timedelta(days=1)
    
    print(f"existing data up to {last_date}")
    
    if start_date >= end_date:
        print(f"already up to date, nothing to download")
        return None, None, True
    
    days_to_download = (end_date - start_date).days
    print(f"will download {days_to_download} days from {start_date} to {end_date}")
    
    return start_date, end_date, True

def download_one(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    download one symbol from yfinance
    
    returns normalized df with columns:
    date, open, high, low, close, adj_close, volume, symbol
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if df.empty:
        #this can happen on weekends or holidays
        print(f"warning: no data downloaded for {symbol} from {start_date} to {end_date}")
        return pd.DataFrame()
    
    #yfinance returns MultiIndex columns when downloading a single ticker
    #flatten the column names: ('Close', 'SPY') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.rename_axis("date").reset_index()
    df = df.rename(columns=str.lower)
    
    #yfinance gives "adj close" for equities; for indices like VIX, VIX3M, it gives no adj close
    if "adj close" not in df.columns:
        df["adj close"] = df["close"]
    
    df["symbol"] = symbol
    
    expected_cols = ["symbol", "date", "open", "high", "low", "close", "adj close", "volume"]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"missing columns {missing} for symbol {symbol}")
    
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
    symbol, date, close, adj_close, ret
    
    Columns:
        symbol: Ticker symbol
        date: Trading date
        close: Closing price (unadjusted)
        adj_close: Adjusted closing price (accounts for dividends/splits)
        ret: Daily log return = ln(adj_close_t / adj_close_{t-1})
    
    Note: ret is LOG returns (not simple returns) for mathematical properties:
          - Additive: multi-period return = sum of daily returns
          - Symmetric: +10% and -10% have similar magnitude
          - Standard in quantitative finance
    
    To convert to simple returns: simple_ret = exp(ret) - 1
    """
    
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["adj close"]
    
    df = df.sort_values(["symbol", "date"])
    
    # Compute log returns (canonical)
    df["ret"] = np.log(df["adj_close"] / df.groupby("symbol")["adj_close"].shift(1))
    
    curated = df[["symbol", "date", "close", "adj_close", "ret"]].copy()
    
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
        outpath = outdir / f"{fname}.parquet"
        g.to_parquet(outpath, index=False)


def main():
    print("Market Data Ingestion Service")
    print("="*60)
    
    #check what dates we already have
    start_date, end_date, is_incremental = determine_download_range(DATA_CURATED, START_DATE)
    
    #if already up to date, exit early
    if start_date is None:
        print("ingestion complete (already up to date)")
        return
    
    #download data
    mode = "incremental" if is_incremental else "full"
    print(f"\ndownloading market data (mode={mode})...")
    print(f"date range: {start_date} to {end_date}")
    
    frames = []
    for ticker in TICKERS:
        df = download_one(ticker, str(start_date), str(end_date))
        if not df.empty:
            frames.append(df)
    
    if not frames:
        print("no new data downloaded (likely weekend or holiday)")
        return
    
    all_data = pd.concat(frames, ignore_index=True)
    print(f"downloaded {len(all_data)} rows across {len(frames)} tickers")
    
    #build raw
    print("\nbuilding raw.market...")
    raw = build_raw(all_data)
    
    print("validating raw.market schema...")
    raw_market_schema.validate(raw)
    
    print("writing raw.market partitions...")
    write_partitions(raw, DATA_RAW, "raw.market")
    
    #build curated
    print("\nbuilding curated.market.daily...")
    curated = build_curated(all_data)
    
    print("validating curated.market.daily schema...")
    curated_market_daily_schema.validate(curated)
    
    print("writing curated.market.daily partitions...")
    write_partitions(curated, DATA_CURATED, "daily")
    
    print("ingestion complete.")
    
if __name__ == "__main__":
    main()
  