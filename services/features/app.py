"""
Feature Engineering Service (Level 1)

Implements all features from local/docs/scope/Level 1 Features.md

Incremental processing: Only computes features for new dates not yet in features.L1/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

from data_status import get_last_date


def get_available_curated_dates():
    """Get all dates that have curated data"""
    curated_path = Path("data/curated.market")
    files = list(curated_path.glob("date=*/daily.parquet"))
    
    dates = []
    for f in files:
        date_str = f.parent.name.replace("date=", "")
        try:
            dates.append(pd.to_datetime(date_str).date())
        except:
            pass
    
    return sorted(dates)


def get_existing_feature_dates():
    """Get dates that already have features computed"""
    features_path = Path("data/features.L1")
    if not features_path.exists():
        return []
    
    files = list(features_path.glob("date=*/features.parquet"))
    
    dates = []
    for f in files:
        date_str = f.parent.name.replace("date=", "")
        try:
            dates.append(pd.to_datetime(date_str).date())
        except:
            pass
    
    return sorted(dates)


def determine_dates_to_process():
    """
    Determine which dates need feature computation.
    
    Logic:
    1. Find last date in master_dataset (from JSON or parquet file)
    2. Find last date with features (from features.L1/)
    3. Process the gap between them
    
    If no master_dataset exists, process all curated data (first run).
    
    Returns:
        tuple: (dates_to_process, is_full_rebuild)
    """
    from datetime import timedelta
    
    # Step 1: Find last date in master_dataset
    last_master_date = None
    last_master_date_str = get_last_date()
    
    if last_master_date_str:
        # Priority 1: Read from data_status.json (fast)
        last_master_date = pd.to_datetime(last_master_date_str).date()
        print(f"found last_date in data_status.json: {last_master_date}")
    else:
        # Priority 2: Read master_dataset.parquet (slower)
        master_path = Path("data/master_dataset.parquet")
        if master_path.exists():
            try:
                print("data_status.json not found, reading master_dataset.parquet...")
                master_df = pd.read_parquet(master_path)
                if not master_df.empty:
                    last_master_date = pd.to_datetime(master_df['date']).max().date()
                    print(f"found last date in master_dataset.parquet: {last_master_date}")
            except Exception as e:
                print(f"warning: could not read master_dataset.parquet: {e}")
    
    # If no master_dataset exists, this is a first run
    if last_master_date is None:
        print("no master_dataset found - will process all curated data (first run)")
        available_dates = get_available_curated_dates()
        if not available_dates:
            print("ERROR: No curated data available")
            return [], False
        print(f"  date range: {available_dates[0]} to {available_dates[-1]}")
        return available_dates, True
    
    # Step 2: Find last date with features
    last_feature_date = None
    existing_dates = get_existing_feature_dates()
    
    if existing_dates:
        last_feature_date = max(existing_dates)
        print(f"found last date in features.L1/: {last_feature_date}")
    else:
        print("no existing features found")
    
    # Step 3: Determine what to process
    if last_feature_date is None:
        # Features don't exist - need to process everything in master_dataset
        print("will process all dates from master_dataset")
        master_path = Path("data/master_dataset.parquet")
        master_df = pd.read_parquet(master_path)
        all_dates = sorted(pd.to_datetime(master_df['date']).dt.date.unique())
        print(f"  date range: {all_dates[0]} to {all_dates[-1]}")
        return all_dates, True
    
    elif last_feature_date >= last_master_date:
        # Features are up to date
        print("features are up to date - no new dates to process")
        return [], False
    
    else:
        # Incremental update - generate date range
        date = last_feature_date + timedelta(days=1)
        new_dates = []
        while date <= last_master_date:
            new_dates.append(date)
            date += timedelta(days=1)
        
        print(f"found {len(new_dates)} new dates to process")
        print(f"  date range: {new_dates[0]} to {new_dates[-1]}")
        return new_dates, False


def load_curated_data(date_filter=None):
    """
    Load curated market data, optionally filtered by dates.
    
    For incremental processing, we need context (60 days lookback) for rolling windows.
    
    Args:
        date_filter: list of dates to process (None = load all)
    """
    curated_path = Path("data/curated.market")
    files = list(curated_path.glob("date=*/daily.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No curated data found in {curated_path}")
    
    # If filtering, load context window for rolling calculations
    if date_filter:
        # Need 60 days before first date for rolling windows
        context_start = min(date_filter) - timedelta(days=70)  # buffer
        context_end = max(date_filter)
        
        date_strs = set()
        for f in files:
            date_str = f.parent.name.replace("date=", "")
            try:
                file_date = pd.to_datetime(date_str).date()
                if context_start <= file_date <= context_end:
                    date_strs.add(date_str)
            except:
                pass
        
        files = [f for f in files if f.parent.name.replace("date=", "") in date_strs]
        print(f"Loading {len(files)} curated partitions (with 60-day context window)...")
    else:
        print(f"Loading {len(files)} curated partitions (full)...")
    
    if not files:
        return pd.DataFrame()
    
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    return df


def compute_spy_returns(df):
    """
    Compute SPY multi-period log returns
    
    Uses pre-computed daily log returns from curated data.
    Multi-period returns are computed by summing daily log returns
    (log returns are additive).
    """
    spy = df[df['symbol'] == 'SPY'].copy()
    
    # Use pre-computed log returns from curated data (efficient!)
    for window in [1, 5, 10, 20, 60]:
        col_name = f'spy_ret_{window}d'
        if window == 1:
            # 1-day return is just the daily log return
            spy[col_name] = spy['ret']
        else:
            # Multi-day return = sum of daily log returns (log returns are additive)
            spy[col_name] = spy['ret'].rolling(window).sum()
    
    # Keep only date and return columns
    ret_cols = ['date'] + [f'spy_ret_{w}d' for w in [1, 5, 10, 20, 60]]
    return spy[ret_cols]


def compute_spy_volatility(df):
    """
    Compute SPY realized volatility using sqrt(sum(r^2))
    
    Uses pre-computed daily log returns from curated data.
    Realized volatility = sqrt(sum of squared log returns)
    """
    spy = df[df['symbol'] == 'SPY'].copy()
    
    for window in [5, 10, 20, 60]:
        col_name = f'spy_vol_{window}d'
        # Realized vol: sqrt(sum of squared log returns)
        spy[col_name] = spy['ret'].rolling(window).apply(
            lambda x: np.sqrt(np.sum(x**2)), raw=True
        )
    
    vol_cols = ['date'] + [f'spy_vol_{w}d' for w in [5, 10, 20, 60]]
    return spy[vol_cols]


def compute_drawdown(df):
    """Compute 60-day peak-to-trough drawdown for SPY"""
    spy = df[df['symbol'] == 'SPY'].copy()
    
    # Rolling 60-day max
    rolling_max = spy['adj_close'].rolling(60).max()
    spy['drawdown_60d'] = 1 - (spy['adj_close'] / rolling_max)
    
    return spy[['date', 'drawdown_60d']]


def compute_vix_features(df):
    """Extract VIX, VIX3M, and compute term structure"""
    vix = df[df['symbol'] == '^VIX'][['date', 'adj_close']].rename(columns={'adj_close': 'vix'})
    vix3m = df[df['symbol'] == '^VIX3M'][['date', 'adj_close']].rename(columns={'adj_close': 'vix3m'})
    
    # Merge and compute term structure
    vix_df = vix.merge(vix3m, on='date', how='inner')
    vix_df['vix_term'] = vix_df['vix3m'] / vix_df['vix']
    
    return vix_df


def compute_rsi(df, window=14):
    """Compute 14-day RSI for SPY"""
    spy = df[df['symbol'] == 'SPY'].copy()
    
    # Calculate price changes
    delta = spy['adj_close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    spy['rsi_spy_14'] = 100 - (100 / (1 + rs))
    
    return spy[['date', 'rsi_spy_14']]


def compute_correlations(df):
    """
    Compute rolling correlations between SPY and other assets
    
    Uses pre-computed daily log returns from curated data.
    """
    # Pivot log returns to wide format
    returns_wide = df.pivot(index='date', columns='symbol', values='ret')
    
    # 20-day correlations
    corr_spy_tlt_20d = returns_wide['SPY'].rolling(20).corr(returns_wide['TLT'])
    corr_spy_hyg_20d = returns_wide['SPY'].rolling(20).corr(returns_wide['HYG'])
    
    # 60-day correlations
    corr_spy_tlt_60d = returns_wide['SPY'].rolling(60).corr(returns_wide['TLT'])
    corr_spy_hyg_60d = returns_wide['SPY'].rolling(60).corr(returns_wide['HYG'])
    
    corr_df = pd.DataFrame({
        'date': returns_wide.index,
        'corr_spy_tlt_20d': corr_spy_tlt_20d.values,
        'corr_spy_hyg_20d': corr_spy_hyg_20d.values,
        'corr_spy_tlt_60d': corr_spy_tlt_60d.values,
        'corr_spy_hyg_60d': corr_spy_hyg_60d.values
    })
    
    return corr_df


def compute_spreads(df):
    """Compute HYG-TLT spread and realized vol vs VIX spread"""
    # Get returns for HYG and TLT
    hyg = df[df['symbol'] == 'HYG'][['date', 'ret']].rename(columns={'ret': 'ret_hyg'})
    tlt = df[df['symbol'] == 'TLT'][['date', 'ret']].rename(columns={'ret': 'ret_tlt'})
    
    # HYG-TLT spread
    spread_df = hyg.merge(tlt, on='date', how='inner')
    spread_df['hyg_tlt_spread'] = spread_df['ret_hyg'] - spread_df['ret_tlt']
    
    return spread_df[['date', 'hyg_tlt_spread']]


def build_features(df):
    """Build all Level 1 features and merge into single dataframe per date"""
    print("\nComputing features...")
    
    print("  - SPY returns (1d, 5d, 10d, 20d, 60d)...")
    spy_returns = compute_spy_returns(df)
    
    print("  - SPY volatility (5d, 10d, 20d, 60d)...")
    spy_vol = compute_spy_volatility(df)
    
    print("  - Drawdown (60d)...")
    drawdown = compute_drawdown(df)
    
    print("  - VIX features (vix, vix3m, vix_term)...")
    vix_features = compute_vix_features(df)
    
    print("  - RSI (14d)...")
    rsi = compute_rsi(df)
    
    print("  - Correlations (20d, 60d)...")
    correlations = compute_correlations(df)
    
    print("  - Spreads (HYG-TLT, RV-VIX)...")
    spreads = compute_spreads(df)
    
    # Merge all features on date
    print("\n  - Merging features...")
    features = spy_returns
    for feat_df in [spy_vol, drawdown, vix_features, rsi, correlations, spreads]:
        features = features.merge(feat_df, on='date', how='left')
    
    # Compute RV-VIX spread (needs spy_vol_20d and vix)
    features['rv_vix_spread_20d'] = features['spy_vol_20d'] - features['vix']
    
    return features


def write_partitions(df, output_dir="data/features.L1"):
    """Write features partitioned by date"""
    output_path = Path(output_dir)
    
    # Group by date and write partitions
    dates = df['date'].unique()
    print(f"\nWriting {len(dates)} feature partitions...")
    
    for date in dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        partition_dir = output_path / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        date_df = df[df['date'] == date]
        outfile = partition_dir / "features.parquet"
        date_df.to_parquet(outfile, index=False)
    
    print(f"Wrote {len(dates)} partitions to {output_path}")


def main():
    print("=" * 60)
    print("Feature Engineering Service (Level 1)")
    print("=" * 60)
    
    # Determine which dates to process
    print("\nStep 1: Determining dates to process...")
    dates_to_process, is_full_rebuild = determine_dates_to_process()
    
    if not dates_to_process:
        print("\n" + "=" * 60)
        print("Features Already Up-to-Date - No Action Needed")
        print("=" * 60)
        return
    
    mode = "full rebuild" if is_full_rebuild else "incremental update"
    print(f"\nMode: {mode}")
    print(f"Processing {len(dates_to_process)} dates")
    
    # Load curated data (with context for rolling windows)
    print("\nStep 2: Loading curated data...")
    if not is_full_rebuild:
        df = load_curated_data(date_filter=dates_to_process)
    else:
        df = load_curated_data()
    
    # Build features
    print("\nStep 3: Computing features...")
    df = build_features(df)
    
    # Filter to only dates we're processing
    if not is_full_rebuild:
        df['date'] = pd.to_datetime(df['date'])
        filter_dates = pd.to_datetime([d for d in dates_to_process])
        df = df[df['date'].isin(filter_dates)]
        print(f"Filtered to {len(df)} rows for target dates")
    
    # Write partitioned output
    print("\nStep 4: Writing features...")
    write_partitions(df)
    
    print("\nFeature engineering complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
