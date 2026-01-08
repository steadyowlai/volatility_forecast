"""
Dataset Preparation Service

Maintains the master dataset for training and monitoring.

Workflow:
1. Load all features from data/features.L1/
2. Load curated market data from data/curated.market/
3. Compute RV_5d labels (5-day forward realized volatility)
4. Merge features + labels
5. Drop NaN values
6. Save to data/master_dataset.parquet

This creates a single source of truth that both training and monitoring services use.
Runs daily after features are computed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# Configuration
DATA_CURATED = Path("data/curated.market")
DATA_FEATURES = Path("data/features.L1")
OUTPUT_PATH = Path("data/master_dataset.parquet")


def load_curated_data():
    """Load all curated market data."""
    curated_path = DATA_CURATED
    files = list(curated_path.glob("date=*/daily.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No curated data found in {curated_path}")
    
    print(f"loading {len(files)} curated partitions...")
    
    dfs = []
    skipped = 0
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"warning: skipping corrupted file {f.parent.name}/{f.name}: {str(e)[:50]}")
            skipped += 1
    
    if skipped > 0:
        print(f"skipped {skipped} corrupted files")
    
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    print(f"loaded {len(df)} rows from curated data")
    
    return df


def compute_rv_5d(df):
    """
    Compute 5-day forward realized volatility (target variable).
    
    RV_5d = sqrt(sum of next 5 daily squared returns)
    This is what we're trying to predict.
    
    We only compute this for SPY (the target asset).
    """
    spy = df[df['symbol'] == 'SPY'].copy()
    spy = spy.sort_values('date').reset_index(drop=True)
    
    # Forward looking: sum of next 5 squared returns
    spy['rv_5d'] = np.sqrt(
        spy['ret'].shift(-1)**2 + 
        spy['ret'].shift(-2)**2 + 
        spy['ret'].shift(-3)**2 + 
        spy['ret'].shift(-4)**2 + 
        spy['ret'].shift(-5)**2
    )
    
    print(f"computed rv_5d for {len(spy)} dates")
    
    return spy[['date', 'rv_5d']]


def load_features():
    """Load all computed features."""
    features_path = DATA_FEATURES
    files = list(features_path.glob("date=*/features.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No features found in {features_path}")
    
    print(f"loading {len(files)} feature partitions...")
    
    dfs = []
    skipped = 0
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"warning: skipping corrupted file {f.parent.name}/{f.name}: {str(e)[:50]}")
            skipped += 1
    
    if skipped > 0:
        print(f"skipped {skipped} corrupted files")
    
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"loaded {len(df)} rows from features")
    
    return df


def prepare_dataset():
    """
    Load features and labels, merge them, drop NaNs, save master dataset.
    """
    print("\n" + "="*60)
    print("Dataset Preparation Service")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nStep 1: Loading curated data for label generation...")
    curated = load_curated_data()

    print("\nStep 2: Computing 5-day forward realized volatility (RV_5d)...")
    labels = compute_rv_5d(curated)

    print("\nStep 3: Loading features...")
    features = load_features()

    print("\nStep 4: Merging features with labels...")
    df = features.merge(labels, on='date', how='inner')
    
    print(f"  merged dataset: {len(df)} rows")
    print(f"  date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Drop rows with any NaN values
    # Happens at the start (rolling windows not full) and end (no future returns for RV_5d)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    print(f"\nStep 5: Cleaning data...")
    print(f"  dropped {dropped_rows} rows with NaN values")
    print(f"  final dataset: {len(df)} rows")
    print(f"  clean date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Save master dataset
    print(f"\nStep 6: Saving master dataset...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    
    # Get file size
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    
    print(f"  saved to: {OUTPUT_PATH}")
    print(f"  file size: {file_size_mb:.2f} MB")
    print(f"  rows: {len(df)}")
    print(f"  columns: {len(df.columns)}")
    print(f"  features: {len([c for c in df.columns if c not in ['date', 'rv_5d']])}")
    
    # Summary statistics
    print(f"\nDataset Summary:")
    print(f"  date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  total days: {(df['date'].max() - df['date'].min()).days}")
    print(f"  samples: {len(df)}")
    print(f"  rv_5d stats: mean={df['rv_5d'].mean():.6f}, std={df['rv_5d'].std():.6f}")
    print(f"  rv_5d range: [{df['rv_5d'].min():.6f}, {df['rv_5d'].max():.6f}]")
    
    print("\n" + "="*60)
    print("Dataset Preparation Complete")
    print("="*60)
    
    return df


def main():
    try:
        prepare_dataset()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
