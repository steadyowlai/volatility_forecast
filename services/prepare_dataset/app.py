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
from storage import Storage


# Initialize storage
storage = Storage()

# Configuration
DATA_CURATED = "data/curated.market"
DATA_FEATURES = "data/features.L1"
OUTPUT_PATH = "data/master_dataset.parquet"


def get_available_feature_dates():
    """Get all dates that have features computed."""
    # List all feature files - only once
    print("  scanning feature partitions...")
    files = storage.list_files(f"{DATA_FEATURES}/")
    
    dates = set()  # Use set for faster lookups
    for f in files:
        if 'features.parquet' in f:
            # Extract date from path like "data/features.L1/date=2024-01-01/features.parquet"
            try:
                # Fast extraction: split and find date= part
                date_str = f.split('/date=')[1].split('/')[0]
                dates.add(pd.to_datetime(date_str).date())
            except:
                pass
    
    return sorted(dates)


def get_existing_master_dates():
    """Get dates already present in master_dataset.parquet."""
    if not storage.exists(OUTPUT_PATH):
        return []
    
    try:
        df = storage.read_parquet(OUTPUT_PATH)
        df['date'] = pd.to_datetime(df['date'])
        # Only read date column for efficiency
        return sorted(df['date'].dt.date.unique())
    except Exception as e:
        print(f"warning: could not read existing master_dataset: {e}")
        return []


def determine_dates_to_process():
    """
    Determine which dates need to be added to master dataset.
    
    Strategy:
    - Check data_status.json for last processed date (O(1) - just read JSON)
    - Check if next sequential dates exist by testing file existence (O(n) where n = new dates)
    - No need to list 4000+ files!
    
    Returns:
        tuple: (dates_to_process, is_full_rebuild)
        - dates_to_process: list of dates to add
        - is_full_rebuild: True if building from scratch, False if incremental
    """
    from data_status import get_status
    from datetime import timedelta
    
    # Check data_status.json first (fast - just read JSON)
    status = get_status()
    
    if not status or 'last_date' not in status:
        print("no existing master_dataset found - will build from scratch")
        # Only for full rebuild do we need to scan all files
        available_dates = get_available_feature_dates()
        return available_dates, True
    
    # We know the last date - just check if next dates exist!
    last_date = pd.to_datetime(status['last_date']).date()
    print(f"last processed date: {last_date}")
    
    # Check for new dates by testing existence of sequential date folders
    # This is O(n) where n = number of new dates (usually 1-10)
    # Instead of O(4000+) listing all files!
    print("  checking for new feature dates...")
    new_dates = []
    check_date = last_date + timedelta(days=1)
    max_gap = 30  # Stop after 30 consecutive missing dates (reasonable gap for trading days)
    consecutive_missing = 0
    
    while consecutive_missing < max_gap:
        # Check if this date has features computed
        feature_path = f"{DATA_FEATURES}/date={check_date}/features.parquet"
        if storage.exists(feature_path):
            new_dates.append(check_date)
            consecutive_missing = 0  # Reset counter
        else:
            consecutive_missing += 1
        
        check_date += timedelta(days=1)
    
    if not new_dates:
        print("master_dataset is up to date - no new dates to process")
        return [], False
    
    print(f"found {len(new_dates)} new dates to add to master_dataset")
    print(f"  date range: {new_dates[0]} to {new_dates[-1]}")
    
    return new_dates, False


def load_curated_data(date_filter=None, start_date=None, file_list=None):
    """
    Load curated market data, optionally filtered by dates or from a start date.
    
    Args:
        date_filter: list of specific dates to load (None = load all)
        start_date: load all dates >= this date (None = load all)
        file_list: pre-fetched list of files (optimization to avoid re-listing)
    """
    # Get all curated files (use cached list if provided)
    if file_list is None:
        print("  scanning curated partitions...")
        all_files = storage.list_files(f"{DATA_CURATED}/")
        files = [f for f in all_files if 'daily.parquet' in f]
    else:
        files = [f for f in file_list if 'daily.parquet' in f]
    
    if not files:
        raise FileNotFoundError(f"No curated data found in {DATA_CURATED}")
    
    # Filter files by date if specified
    if date_filter:
        date_strs = {d.strftime("%Y-%m-%d") for d in date_filter}
        files = [f for f in files if f.split('/date=')[1].split('/')[0] in date_strs]
        print(f"loading {len(files)} curated partitions for specific dates...")
    elif start_date:
        start_str = start_date.strftime("%Y-%m-%d")
        files = [f for f in files if f.split('/date=')[1].split('/')[0] >= start_str]
        print(f"loading {len(files)} curated partitions from {start_date} onwards...")
    else:
        print(f"loading {len(files)} curated partitions (full)...")
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    skipped = 0
    for f in files:
        try:
            dfs.append(storage.read_parquet(f))
        except Exception as e:
            # Extract just the date folder for cleaner error message
            date_folder = [p for p in f.split('/') if p.startswith('date=')]
            date_folder = date_folder[0] if date_folder else f
            print(f"warning: skipping corrupted file {date_folder}: {str(e)[:50]}")
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
    # Get all feature files
    all_files = storage.list_files(f"{DATA_FEATURES}/")
    files = [f for f in all_files if 'features.parquet' in f]
    
    if not files:
        raise FileNotFoundError(f"No features found in {DATA_FEATURES}")
    
    # Filter files by date if specified
    if date_filter:
        date_strs = {d.strftime("%Y-%m-%d") for d in date_filter}
        filtered = []
        for f in files:
            # Extract date from path like "data/features.L1/date=2024-01-01/features.parquet"
            parts = f.split('/')
            for part in parts:
                if part.startswith('date='):
                    if part.replace("date=", "") in date_strs:
                        filtered.append(f)
                    break
        files = filtered
        print(f"loading {len(files)} feature partitions for new dates...")
    else:
        print(f"loading {len(files)} feature partitions (full)...")
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    skipped = 0
    for f in files:
        try:
            dfs.append(storage.read_parquet(f))
        except Exception as e:
            # Extract just the date folder for cleaner error message
            date_folder = [p for p in f.split('/') if p.startswith('date=')]
            date_folder = date_folder[0] if date_folder else f
            print(f"warning: skipping corrupted file {date_folder}: {str(e)[:50]}")
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
    if not is_full_rebuild and storage.exists(OUTPUT_PATH):
        print("  loading existing master_dataset...")
        existing_data = storage.read_parquet(OUTPUT_PATH)
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
    storage.write_parquet(df, OUTPUT_PATH)
    
    print(f"  saved to: {OUTPUT_PATH}")
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
