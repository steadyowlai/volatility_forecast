"""
Feature Engineering Service (Level 1)

Implements all features from local/docs/scope/Level 1 Features.md
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_curated_data():
    """Load all curated market data"""
    curated_path = Path("data/curated.market")
    files = list(curated_path.glob("date=*/daily.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No curated data found in {curated_path}")
    
    print(f"Loading {len(files)} curated partitions...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    return df


def compute_spy_returns(df):
    """Compute SPY log returns over multiple windows"""
    spy = df[df['symbol'] == 'SPY'].copy()
    
    # Log returns over different windows
    for window in [1, 5, 10, 20, 60]:
        col_name = f'spy_ret_{window}d'
        spy[col_name] = np.log(spy['adj_close'] / spy['adj_close'].shift(window))
    
    # Keep only date and return columns
    ret_cols = ['date'] + [f'spy_ret_{w}d' for w in [1, 5, 10, 20, 60]]
    return spy[ret_cols]


def compute_spy_volatility(df):
    """Compute SPY realized volatility using sqrt(sum(r^2))"""
    spy = df[df['symbol'] == 'SPY'].copy()
    
    for window in [5, 10, 20, 60]:
        col_name = f'spy_vol_{window}d'
        # Realized vol: sqrt(sum of squared returns)
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
    """Compute rolling correlations between SPY and other assets"""
    # Pivot returns to wide format
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
    
    # Load curated data
    df = load_curated_data()
    
    # Build features
    df = build_features(df)
    
    # Write partitioned output
    write_partitions(df)
    
    print("\nFeature engineering complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
