import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

from schemas import raw_market_schema, curated_market_daily_schema

# Configuration
DATA_RAW = Path("data/raw.market")
DATA_CURATED = Path("data/curated.market")
MASTER_DATASET = Path("data/master_dataset.parquet")

# Configurable start date (can override via environment variable)
# Default: 2010-01-01 - Post-financial crisis "new normal" regime
# ~3,770 trading days (15 years), captures multiple market cycles
# Excludes 2008-2009 crisis (structural break in market dynamics)
START_DATE = os.getenv("START_DATE", "2010-01-01")

TICKERS = [
    "SPY", # S&P 500 ETF (equity benchmark)
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

def get_manifest_dates(manifest_path: Path) -> list:
    """Read dates from manifest file."""
    if not manifest_path.exists():
        return []
    
    try:
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        return data.get('dates', [])
    except Exception as e:
        print(f"warning: could not read manifest: {e}")
        return []


def determine_download_range(data_dir: Path, default_start: str) -> tuple:
    """
    Figure out what date range to download.
    
    Source of truth: curated.market/_manifest.json
    
    1. Check curated manifest for last date
    2. If no manifest, scan partitions
    3. If no data, download full history from default_start
    
    Returns: (start_date, end_date, is_incremental)
    """
    end_date = datetime.now().date()
    
    # Priority 1: Check curated manifest
    manifest_path = data_dir / "_manifest.json"
    manifest_dates = get_manifest_dates(manifest_path)
    
    if manifest_dates:
        last_date = pd.to_datetime(max(manifest_dates)).date()
        print(f"found last date in curated manifest: {last_date}")
        start_date = last_date + timedelta(days=1)
        
        if start_date >= end_date:
            print(f"already up to date, nothing to download")
            return None, None, True
        
        days_to_download = (end_date - start_date).days
        print(f"will download {days_to_download} days from {start_date} to {end_date}")
        return start_date, end_date, True
    
    # Priority 2: Check existing curated.market partitions (fallback if manifest missing)
    existing_dates = get_existing_dates(data_dir)
    
    if existing_dates:
        last_date = max(existing_dates)
        print(f"manifest not found, scanned curated.market/ partitions")
        print(f"found existing data up to {last_date}")
        start_date = last_date + timedelta(days=1)
        
        if start_date >= end_date:
            print(f"already up to date, nothing to download")
            return None, None, True
        
        days_to_download = (end_date - start_date).days
        print(f"will download {days_to_download} days from {start_date} to {end_date}")
        return start_date, end_date, True
    
    # Priority 3: No existing data, download full history
    start_date = pd.to_datetime(default_start).date()
    print(f"no existing curated data found")
    print(f"will download full history from {start_date}")
    return start_date, end_date, False

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
  
def build_curated(df_all: pd.DataFrame, is_incremental: bool = False) -> pd.DataFrame:
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
    
    Args:
        df_all: Downloaded market data
        is_incremental: If True, load previous day's data to compute first return
    """
    
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["adj close"]
    
    # If incremental, load last date from curated manifest for return calculation
    if is_incremental:
        manifest_dates = get_manifest_dates(DATA_CURATED / "_manifest.json")
        if manifest_dates:
            last_date_str = max(manifest_dates)
            prev_file = DATA_CURATED / f"date={last_date_str}" / "daily.parquet"
            
            if prev_file.exists():
                try:
                    prev_df = pd.read_parquet(prev_file)
                    prev_df["date"] = pd.to_datetime(prev_df["date"])
                    # Append previous day's data to compute returns correctly
                    df = pd.concat([prev_df, df], ignore_index=True)
                    print(f"  loaded previous date ({last_date_str}) for return calculation")
                except Exception as e:
                    print(f"  warning: could not load previous date for returns: {e}")
    
    df = df.sort_values(["symbol", "date"])
    
    # Compute log returns (canonical)
    df["ret"] = np.log(df["adj_close"] / df.groupby("symbol")["adj_close"].shift(1))
    
    # If we loaded previous data, filter to only new dates
    if is_incremental:
        min_new_date = pd.to_datetime(df_all["date"]).min()
        df = df[df["date"] >= min_new_date]
    
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


def update_manifest(root: Path, new_dates: list) -> None:
    """Write a date manifest to avoid full directory scans."""
    manifest_path = root / "_manifest.json"
    # Use on-disk partitions as the authoritative source for completeness
    disk_dates = [d.strftime("%Y-%m-%d") for d in get_existing_dates(root)]
    existing_dates = []

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                payload = json.load(f)
            existing_dates = payload.get("dates", [])
        except Exception:
            existing_dates = []

    merged = sorted(set(existing_dates) | set(disk_dates) | set(new_dates))
    payload = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "dates": merged
    }

    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    print("Market Data Ingestion Service")
    print("="*60)
    
    #check what dates we already have
    existing_dates = get_existing_dates(DATA_CURATED)
    start_date, end_date, is_incremental = determine_download_range(DATA_CURATED, START_DATE)
    
    #if already up to date, exit early
    if start_date is None:
        if existing_dates:
            update_manifest(DATA_CURATED, [d.strftime("%Y-%m-%d") for d in existing_dates])
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
    curated = build_curated(all_data, is_incremental=is_incremental)
    
    print("validating curated.market.daily schema...")
    curated_market_daily_schema.validate(curated)
    
    print("writing curated.market.daily partitions...")
    write_partitions(curated, DATA_CURATED, "daily")

    curated_dates = sorted(pd.to_datetime(curated["date"]).dt.strftime("%Y-%m-%d").unique())
    all_dates = [d.strftime("%Y-%m-%d") for d in existing_dates] + curated_dates
    update_manifest(DATA_CURATED, all_dates)
    
    print("ingestion complete.")
    
if __name__ == "__main__":
    main()
  