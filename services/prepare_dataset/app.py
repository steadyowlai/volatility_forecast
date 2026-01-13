"""
Dataset Preparation Service

Maintains the master dataset for training and monitoring.

Workflow:
1. Check existing master_dataset.parquet (if exists)
2. Identify new dates available in features that aren't in master dataset
3. Load only new features and curated data for those dates
4. Compute RV_5d labels for new dates
5. Merge features + labels for new dates
6. Append to existing master_dataset (or create new if first run)
7. Save updated master_dataset.parquet

This creates a single source of truth that both training and monitoring services use.
Runs daily after features are computed (incremental updates).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from data_status import update_status


# Configuration
DATA_CURATED = Path("data/curated.market")
DATA_FEATURES = Path("data/features.L1")
OUTPUT_PATH = Path("data/master_dataset.parquet")


def get_available_feature_dates():
    """Get all dates that have features computed."""
    features_path = DATA_FEATURES
    files = list(features_path.glob("date=*/features.parquet"))
    
    dates = []
    for f in files:
        date_str = f.parent.name.replace("date=", "")
        try:
            dates.append(pd.to_datetime(date_str).date())
        except:
            pass
    
    return sorted(dates)


def get_existing_master_dates():
    """Get dates already present in master_dataset.parquet."""
    if not OUTPUT_PATH.exists():
        return []
    
    try:
        df = pd.read_parquet(OUTPUT_PATH, columns=['date'])
        df['date'] = pd.to_datetime(df['date'])
        return sorted(df['date'].dt.date.unique())
    except Exception as e:
        print(f"warning: could not read existing master_dataset: {e}")
        return []


def determine_dates_to_process():
    """
    Determine which dates need to be added to master dataset.
    
    Strategy:
    - Only process dates AFTER the last date in master_dataset
    - Ignore older dates with features but not in master (were filtered due to NaN)
    
    Returns:
        tuple: (dates_to_process, is_full_rebuild)
        - dates_to_process: list of dates to add
        - is_full_rebuild: True if building from scratch, False if incremental
    """
    available_dates = get_available_feature_dates()
    existing_dates = get_existing_master_dates()
    
    if not existing_dates:
        print("no existing master_dataset found - will build from scratch")
        return available_dates, True
    
    # Get last date in master dataset
    last_master_date = max(existing_dates)
    
    # Only process dates AFTER last master date (incremental)
    new_dates = [d for d in available_dates if d > last_master_date]
    
    if not new_dates:
        print("master_dataset is up to date - no new dates to process")
        return [], False
    
    print(f"found {len(new_dates)} new dates to add to master_dataset")
    print(f"  date range: {new_dates[0]} to {new_dates[-1]}")
    
    return new_dates, False


def load_curated_data(date_filter=None, start_date=None):
    """
    Load curated market data, optionally filtered by dates or from a start date.
    
    Args:
        date_filter: list of specific dates to load (None = load all)
        start_date: load all dates >= this date (None = load all)
    """
    curated_path = DATA_CURATED
    files = list(curated_path.glob("date=*/daily.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No curated data found in {curated_path}")
    
    # Filter files by date if specified
    if date_filter:
        date_strs = {d.strftime("%Y-%m-%d") for d in date_filter}
        files = [f for f in files if f.parent.name.replace("date=", "") in date_strs]
        print(f"loading {len(files)} curated partitions for specific dates...")
    elif start_date:
        filtered_files = []
        for f in files:
            date_str = f.parent.name.replace("date=", "")
            try:
                file_date = pd.to_datetime(date_str).date()
                if file_date >= start_date:
                    filtered_files.append(f)
            except:
                pass
        files = filtered_files
        print(f"loading {len(files)} curated partitions from {start_date} onwards...")
    else:
        print(f"loading {len(files)} curated partitions (full)...")
    
    if not files:
        return pd.DataFrame()
    
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
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    print(f"loaded {len(df)} rows from curated data")
    
    return df


def compute_rv_5d(df, need_context_dates=None):
    """
    Compute 5-day forward realized volatility (target variable).
    
    RV_5d = sqrt(sum of next 5 daily squared returns)
    This is what we're trying to predict.
    
    We only compute this for SPY (the target asset).
    
    Args:
        df: DataFrame with curated data
        need_context_dates: If provided, only return RV for these dates, 
                           but compute using context from all dates in df
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
    
    # Filter to only requested dates if specified
    if need_context_dates:
        need_dates = pd.to_datetime([d for d in need_context_dates])
        spy = spy[spy['date'].isin(need_dates)]
    
    print(f"computed rv_5d for {len(spy)} dates")
    
    return spy[['date', 'rv_5d']]


def load_features(date_filter=None):
    """
    Load computed features, optionally filtered by dates.
    
    Args:
        date_filter: list of dates to load (None = load all)
    """
    features_path = DATA_FEATURES
    files = list(features_path.glob("date=*/features.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No features found in {features_path}")
    
    # Filter files by date if specified
    if date_filter:
        date_strs = {d.strftime("%Y-%m-%d") for d in date_filter}
        files = [f for f in files if f.parent.name.replace("date=", "") in date_strs]
        print(f"loading {len(files)} feature partitions for new dates...")
    else:
        print(f"loading {len(files)} feature partitions (full)...")
    
    if not files:
        return pd.DataFrame()
    
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
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"loaded {len(df)} rows from features")
    
    return df


def prepare_dataset():
    """
    Load features and labels, merge them, drop NaNs, save/append to master dataset.
    
    If master_dataset exists: incrementally add only new dates
    If master_dataset doesn't exist: build from scratch
    """
    print("\n" + "="*60)
    print("Dataset Preparation Service")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine which dates to process
    print("\nStep 1: Determining dates to process...")
    dates_to_process, is_full_rebuild = determine_dates_to_process()
    
    if not dates_to_process:
        print("\n" + "="*60)
        print("Dataset Already Up-to-Date - No Action Needed")
        print("="*60)
        return None
    
    mode = "full rebuild" if is_full_rebuild else "incremental update"
    print(f"\nMode: {mode}")
    print(f"Processing {len(dates_to_process)} dates")
    
    # For incremental updates, we need future data for RV computation
    # RV_5d needs the next 5 trading days of returns to compute forward volatility
    if not is_full_rebuild:
        # Load all curated data from first new date onwards
        # This ensures we have enough future data for RV_5d computation
        # (no magic numbers - just load whatever is available)
        context_start = dates_to_process[0]
        
        print(f"\nStep 2: Loading curated data from {context_start} onwards...")
        
        # Load from start date onwards (includes all future data needed for RV_5d)
        curated = load_curated_data(start_date=context_start)
    else:
        print(f"\nStep 2: Loading all curated data...")
        curated = load_curated_data()
    
    print(f"\nStep 3: Computing 5-day forward realized volatility (RV_5d)...")
    if not is_full_rebuild:
        # Only return RV for dates we're processing
        labels = compute_rv_5d(curated, need_context_dates=dates_to_process)
    else:
        labels = compute_rv_5d(curated)

    print(f"\nStep 4: Loading features...")
    if not is_full_rebuild:
        features = load_features(date_filter=dates_to_process)
    else:
        features = load_features()

    print(f"\nStep 5: Merging features with labels...")
    new_data = features.merge(labels, on='date', how='inner')
    
    print(f"  merged new data: {len(new_data)} rows")
    if len(new_data) > 0:
        print(f"  date range: {new_data['date'].min().date()} to {new_data['date'].max().date()}")
    
    # Drop rows with any NaN values
    initial_rows = len(new_data)
    new_data = new_data.dropna()
    dropped_rows = initial_rows - len(new_data)
    
    print(f"\nStep 6: Cleaning new data...")
    print(f"  dropped {dropped_rows} rows with NaN values")
    print(f"  clean new data: {len(new_data)} rows")
    
    if len(new_data) == 0:
        print("\nWarning: No valid data after cleaning - nothing to add")
        return None
    
    # Load existing master dataset and append new data
    print(f"\nStep 7: Updating master dataset...")
    if not is_full_rebuild and OUTPUT_PATH.exists():
        print("  loading existing master_dataset...")
        existing_data = pd.read_parquet(OUTPUT_PATH)
        existing_data['date'] = pd.to_datetime(existing_data['date'])
        
        print(f"  existing data: {len(existing_data)} rows")
        print(f"  existing date range: {existing_data['date'].min().date()} to {existing_data['date'].max().date()}")
        
        # Combine and sort
        df = pd.concat([existing_data, new_data], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  appended {len(new_data)} new rows")
    else:
        df = new_data
        print(f"  created new master_dataset")
    
    print(f"  final date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Save master dataset
    print(f"\nStep 8: Saving master dataset...")
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
    
    # Update data status (ONLY service that should update the JSON!)
    last_date = df['date'].max().strftime('%Y-%m-%d')
    print(f"\nStep 9: Updating data_status.json...")
    print(f"  last_date: {last_date}")
    print(f"  rows: {len(df)}")
    update_status(
        last_date=last_date,
        rows=len(df)
    )
    
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
