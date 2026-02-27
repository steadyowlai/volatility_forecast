"""
Dataset Preparation Service

Creates and maintains 3 datasets from features:
1. master_dataset.parquet - All features + rv_5d labels (keeps all rows including NaN)
2. predict_dataset.parquet - Last 35 rows, features ONLY (no rv_5d column)
3. labels_for_predict.parquet - Last ~30 rows with labels (date + rv_5d only)

Workflow:
1. Read features manifest (source of truth for available dates)
2. Check data_status.json for last processed date
3. Load only NEW feature partitions since last run (incremental)
4. For rv_5d computation: Load last 5 rows from existing master (for context)
5. Compute rv_5d on features_df (includes 5 context rows + new data)
6. Create all 3 datasets in memory (efficient, no disk re-reads):
   - master_new = features_df[5:] (exclude context rows)
   - Append master_new to existing master_dataset.parquet
   - predict = master_full.tail(35).drop('rv_5d')
   - labels = master_full.tail(35)[['date','rv_5d']].dropna()
7. Update data_status.json with all 3 dataset metadata

Key optimizations:
- Manifest-driven: O(1) incremental detection using set difference
- Load only new partitions: Don't re-read 4000+ partitions
- In-memory processing: Create all 3 datasets from same DataFrame
- Smart appending: Only master grows, predict/labels always overwritten
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from storage import Storage

# Initialize storage
storage = Storage()

# Configuration
DATA_FEATURES = "data/features.L1"
FEATURES_MANIFEST = f"{DATA_FEATURES}/_manifest.json"
MASTER_PATH = "data/master_dataset.parquet"
PREDICT_PATH = "data/predict_dataset.parquet"
LABELS_PATH = "data/labels_for_predict.parquet"
STATUS_PATH = "data/data_status.json"


def read_features_manifest():
    """Read features manifest to get all available dates."""
    if not storage.exists(FEATURES_MANIFEST):
        raise FileNotFoundError(f"Features manifest not found: {FEATURES_MANIFEST}")
    
    manifest = storage.read_json(FEATURES_MANIFEST)
    dates = sorted(manifest.get('dates', []))
    print(f"Features manifest: {len(dates)} dates available")
    return dates


def read_status():
    """Read data_status.json to get last processed date."""
    if not storage.exists(STATUS_PATH):
        return None
    
    try:
        status = storage.read_json(STATUS_PATH)
        return status
    except Exception as e:
        print(f"Warning: Could not read data_status.json: {e}")
        return None


def write_status(master_rows, master_last_date, predict_rows, predict_last_date, labels_rows):
    """Write data_status.json with all dataset metadata."""
    status = {
        "master_dataset": {
            "last_date": master_last_date,
            "rows": master_rows,
            "updated_at": datetime.utcnow().isoformat() + 'Z'
        },
        "predict_dataset": {
            "last_date": predict_last_date,
            "rows": predict_rows,
            "updated_at": datetime.utcnow().isoformat() + 'Z'
        },
        "labels_for_predict": {
            "rows": labels_rows,
            "updated_at": datetime.utcnow().isoformat() + 'Z'
        }
    }
    
    storage.write_json(status, STATUS_PATH)
    print(f"Updated data_status.json:")
    print(f"  master: {master_rows} rows, last_date={master_last_date}")
    print(f"  predict: {predict_rows} rows, last_date={predict_last_date}")
    print(f"  labels: {labels_rows} rows")


def detect_new_dates(manifest_dates):
    """
    Detect which dates need to be processed.
    
    Returns:
        tuple: (new_dates, is_full_rebuild)
    """
    status = read_status()
    
    # First run or no existing master
    if status is None or 'master_dataset' not in status:
        print("No existing master_dataset found - will build from scratch")
        return manifest_dates, True
    
    last_date = status['master_dataset'].get('last_date')
    if not last_date:
        print("No last_date in status - will build from scratch")
        return manifest_dates, True
    
    # Find new dates after last_date
    new_dates = sorted([d for d in manifest_dates if d > last_date])
    
    if not new_dates:
        print(f"Master dataset is up to date (last_date: {last_date})")
        return [], False
    
    print(f"Found {len(new_dates)} new dates since {last_date}")
    print(f"  Date range: {new_dates[0]} to {new_dates[-1]}")
    return new_dates, False


def load_feature_partitions(dates):
    """Load feature partitions for specific dates."""
    if not dates:
        return pd.DataFrame()
    
    print(f"Loading {len(dates)} feature partitions...")
    
    dfs = []
    missing = 0
    
    for date_str in dates:
        path = f"{DATA_FEATURES}/date={date_str}/features.parquet"
        if storage.exists(path):
            try:
                df = storage.read_parquet(path)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Failed to read {date_str}: {e}")
                missing += 1
        else:
            print(f"  Warning: Missing partition for {date_str}")
            missing += 1
    
    if missing > 0:
        print(f"  Skipped {missing} missing/corrupted partitions")
    
    if not dfs:
        return pd.DataFrame()
    
    # Concatenate all partitions
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"  Loaded {len(df)} rows from {len(dfs)} partitions")
    return df


def load_last_n_rows_from_master(n=5):
    """Load last N rows from existing master_dataset for rv_5d context."""
    if not storage.exists(MASTER_PATH):
        return pd.DataFrame()
    
    try:
        master = storage.read_parquet(MASTER_PATH)
        master['date'] = pd.to_datetime(master['date'])
        last_n = master.tail(n).copy()
        print(f"Loaded last {len(last_n)} rows from existing master for rv_5d context")
        print(f"  Context dates: {last_n['date'].min().date()} to {last_n['date'].max().date()}")
        return last_n
    except Exception as e:
        print(f"Warning: Could not load existing master: {e}")
        return pd.DataFrame()


def compute_rv_5d(features_df):
    """
    Compute 5-day forward realized volatility (rv_5d) label.
    
    RV_5d = sqrt(sum of next 5 daily squared returns)
    
    Uses spy_ret_1d (which equals daily SPY log return) to compute forward
    5-day realized volatility. spy_ret_1d is saved in features and equals
    the raw daily return (spy_ret) used during feature computation.
    
    Args:
        features_df: DataFrame with features (1 row per date), must contain 'spy_ret_1d'
    
    Returns:
        DataFrame with rv_5d column added
    """
    df = features_df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Use spy_ret_1d (daily SPY returns) from features
    # spy_ret_1d equals spy_ret (raw daily return) by definition
    if 'spy_ret_1d' not in df.columns:
        raise ValueError("spy_ret_1d column not found in features")
    
    # Forward looking: sum of next 5 squared returns
    # shift(-1) = tomorrow's return, shift(-5) = return 5 days from now
    df['rv_5d'] = np.sqrt(
        df['spy_ret_1d'].shift(-1)**2 +   # Tomorrow's return
        df['spy_ret_1d'].shift(-2)**2 +   # Day after tomorrow
        df['spy_ret_1d'].shift(-3)**2 +   # 3 days ahead
        df['spy_ret_1d'].shift(-4)**2 +   # 4 days ahead
        df['spy_ret_1d'].shift(-5)**2     # 5 days ahead
    )
    
    # Last 5 rows will have NaN rv_5d (expected behavior)
    nan_count = df['rv_5d'].isna().sum()
    print(f"Computed rv_5d: {len(df) - nan_count} with labels, {nan_count} without (last 5 rows)")
    
    return df


def create_all_datasets_in_memory(master_full):
    """
    Create all 3 datasets from master in memory (efficient, no disk re-reads).
    
    Args:
        master_full: Full master dataset (all rows)
    
    Returns:
        tuple: (master_full, predict_dataset, labels_for_predict)
    """
    # 1. Master: Already complete (keep all rows including NaN)
    print(f"\n1. Master dataset: {len(master_full)} rows (keeping all rows including NaN)")
    
    # 2. Predict: Last 35 rows, features ONLY (drop rv_5d column)
    predict = master_full.tail(35).copy()
    predict_dataset = predict.drop(columns=['rv_5d'])
    print(f"2. Predict dataset: {len(predict_dataset)} rows (last 35, features only)")
    
    # 3. Labels: Last 35 rows, but only where rv_5d exists (date + rv_5d only)
    labels_for_predict = predict[['date', 'rv_5d']].dropna()
    print(f"3. Labels for predict: {len(labels_for_predict)} rows (date + rv_5d, ~30 rows expected)")
    
    return master_full, predict_dataset, labels_for_predict


def prepare_dataset():
    """
    Main function to prepare all datasets.
    
    Process:
    1. Read features manifest (source of truth)
    2. Detect new dates using data_status.json
    3. Load only new feature partitions
    4. Load last 5 rows from existing master (for rv_5d context)
    5. Compute rv_5d on features_df (context + new)
    6. Create all 3 datasets in memory
    7. Append to master, overwrite predict/labels
    8. Update data_status.json
    """
    print("\n" + "="*70)
    print("Dataset Preparation Service")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Read features manifest
    print("\nStep 1: Reading features manifest...")
    manifest_dates = read_features_manifest()
    
    if not manifest_dates:
        print("No features available - nothing to process")
        return
    
    print(f"  Available: {manifest_dates[0]} to {manifest_dates[-1]}")
    
    # Step 2: Detect new dates
    print("\nStep 2: Detecting new dates to process...")
    new_dates, is_full_rebuild = detect_new_dates(manifest_dates)
    
    if not new_dates:
        print("\n✅ Master dataset is up to date - no action needed")
        
        # Still create predict/labels from existing master
        if storage.exists(MASTER_PATH):
            print("\nRefreshing predict and labels datasets from existing master...")
            master = storage.read_parquet(MASTER_PATH)
            master['date'] = pd.to_datetime(master['date'])
            _, predict, labels = create_all_datasets_in_memory(master)
            
            storage.write_parquet(predict, PREDICT_PATH)
            storage.write_parquet(labels, LABELS_PATH)
            
            print(f"\n✅ Refreshed predict ({len(predict)} rows) and labels ({len(labels)} rows)")
        
        return
    
    mode = "FULL REBUILD" if is_full_rebuild else "INCREMENTAL UPDATE"
    print(f"\nMode: {mode}")
    print(f"Processing: {len(new_dates)} dates")
    
    # Step 3: Load feature partitions
    print("\nStep 3: Loading feature partitions...")
    
    if is_full_rebuild:
        # Load all features
        features_df = load_feature_partitions(manifest_dates)
        context_rows = pd.DataFrame()  # No context needed for full rebuild
    else:
        # Load last 5 rows from existing master for rv_5d context
        print("\nStep 3a: Loading context from existing master...")
        context_rows = load_last_n_rows_from_master(n=5)
        
        # Load only new feature partitions
        print("\nStep 3b: Loading new feature partitions...")
        new_features = load_feature_partitions(new_dates)
        
        # Combine context + new features
        if len(context_rows) > 0:
            # Drop rv_5d from context (we'll recompute it)
            context_rows = context_rows.drop(columns=['rv_5d'], errors='ignore')
            features_df = pd.concat([context_rows, new_features], ignore_index=True)
            features_df = features_df.sort_values('date').reset_index(drop=True)
            print(f"  Combined: {len(context_rows)} context + {len(new_features)} new = {len(features_df)} total")
        else:
            features_df = new_features
    
    if len(features_df) == 0:
        print("No features loaded - nothing to process")
        return
    
    # Step 4: Compute rv_5d
    print("\nStep 4: Computing rv_5d labels...")
    features_df = compute_rv_5d(features_df)
    
    # Step 5: Extract new rows (exclude context if incremental)
    if is_full_rebuild:
        master_new = features_df.copy()
    else:
        # Split: context rows (with freshly computed rv_5d) and truly new rows
        context_count = len(context_rows)
        context_updated = features_df.iloc[:context_count].copy()   # context with backfilled rv_5d
        master_new = features_df.iloc[context_count:].copy()        # brand new rows
        print(f"\nStep 5: {context_count} context rows backfilled, {len(master_new)} new rows to append")

    # Step 6: Append to master or create new
    print("\nStep 6: Updating master dataset...")

    if is_full_rebuild or not storage.exists(MASTER_PATH):
        print("  Creating new master_dataset from scratch")
        master_full = master_new.copy()
    else:
        print("  Loading existing master...")
        existing_master = storage.read_parquet(MASTER_PATH)
        existing_master['date'] = pd.to_datetime(existing_master['date'])
        print(f"  Existing: {len(existing_master)} rows")

        # Backfill rv_5d for the context rows already in master
        existing_master = existing_master.set_index('date')
        for _, row in context_updated.iterrows():
            d = row['date']
            if d in existing_master.index and pd.notna(row['rv_5d']):
                existing_master.at[d, 'rv_5d'] = row['rv_5d']
        existing_master = existing_master.reset_index()
        backfilled = context_updated['rv_5d'].notna().sum()
        print(f"  Backfilled rv_5d for {backfilled} context rows")

        print(f"  Appending {len(master_new)} new rows...")
        master_full = pd.concat([existing_master, master_new], ignore_index=True)
        master_full = master_full.sort_values('date').reset_index(drop=True)
    
    print(f"  Final master: {len(master_full)} rows")
    print(f"  Date range: {master_full['date'].min().date()} to {master_full['date'].max().date()}")
    
    # Step 7: Create all 3 datasets in memory
    print("\nStep 7: Creating all datasets in memory...")
    master_full, predict_dataset, labels_for_predict = create_all_datasets_in_memory(master_full)
    
    # Step 8: Write all datasets to disk
    print("\nStep 8: Writing datasets to disk...")
    
    print("  Writing master_dataset.parquet...")
    storage.write_parquet(master_full, MASTER_PATH)
    print(f"    ✅ {MASTER_PATH} ({len(master_full)} rows)")
    
    print("  Writing predict_dataset.parquet...")
    storage.write_parquet(predict_dataset, PREDICT_PATH)
    print(f"    ✅ {PREDICT_PATH} ({len(predict_dataset)} rows, features only)")
    
    print("  Writing labels_for_predict.parquet...")
    storage.write_parquet(labels_for_predict, LABELS_PATH)
    print(f"    ✅ {LABELS_PATH} ({len(labels_for_predict)} rows, date + rv_5d only)")
    
    # Step 9: Update data_status.json
    print("\nStep 9: Updating data_status.json...")
    write_status(
        master_rows=len(master_full),
        master_last_date=master_full['date'].max().strftime('%Y-%m-%d'),
        predict_rows=len(predict_dataset),
        predict_last_date=predict_dataset['date'].max().strftime('%Y-%m-%d'),
        labels_rows=len(labels_for_predict)
    )
    
    # Summary
    print("\n" + "="*70)
    print("Dataset Preparation Complete ✅")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Master: {len(master_full)} rows, {len(master_full.columns)} columns")
    print(f"    - Date range: {master_full['date'].min().date()} to {master_full['date'].max().date()}")
    print(f"    - Features: {len([c for c in master_full.columns if c not in ['date', 'rv_5d']])}")
    print(f"    - Labels (rv_5d): {master_full['rv_5d'].notna().sum()} rows")
    print(f"    - Without labels: {master_full['rv_5d'].isna().sum()} rows (expected: last 5)")
    print(f"\n  Predict: {len(predict_dataset)} rows (last 35, features only)")
    print(f"    - For inference by predict service")
    print(f"    - NO rv_5d column (security: prevent label leakage)")
    print(f"\n  Labels: {len(labels_for_predict)} rows (~30 expected)")
    print(f"    - For monitoring by monitor service")
    print(f"    - Only date + rv_5d columns")
    
    if len(master_full) > 0:
        valid_rv = master_full['rv_5d'].dropna()
        if len(valid_rv) > 0:
            print(f"\n  RV_5d statistics:")
            print(f"    - Mean: {valid_rv.mean():.6f}")
            print(f"    - Std:  {valid_rv.std():.6f}")
            print(f"    - Min:  {valid_rv.min():.6f}")
            print(f"    - Max:  {valid_rv.max():.6f}")


def main():
    try:
        prepare_dataset()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


def lambda_handler(event, context):
    """AWS Lambda entry point."""
    try:
        prepare_dataset()
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Dataset preparation completed successfully',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }
        # Chain: trigger vf-predict asynchronously
        try:
            import boto3
            boto3.client('lambda', region_name='us-east-1').invoke(
                FunctionName='vf-predict',
                InvocationType='Event'
            )
            print("Triggered vf-predict")
        except Exception as chain_err:
            print(f"WARNING: Failed to trigger vf-predict: {chain_err}")
        return response
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Dataset preparation failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }


if __name__ == "__main__":
    main()
