"""
Feature Engineering Service (Level 1)

Implements all features from local/docs/scope/Level 1 Features.md

Incremental processing: Only computes features for new dates not yet in features.L1/
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from storage import Storage
from validate_features import validate_feature_partition, validate_features_batch

# Initialize storage (auto-detects local vs S3)
storage = Storage()

CURATED_MANIFEST = "data/curated.market/_manifest.json"
FEATURES_MANIFEST = "data/features.L1/_manifest.json"


def get_manifest_dates(manifest_path: str) -> list:
    try:
        data = storage.read_json(manifest_path)
        return data.get("dates", [])
    except Exception:
        return []


def get_available_curated_dates():
    """Get all dates that have curated data"""
    manifest_dates = get_manifest_dates(CURATED_MANIFEST)
    if manifest_dates:
        return sorted([pd.to_datetime(d).date() for d in manifest_dates])

    # Fallback: scan partitions via storage layer
    date_strings = storage.list_partitions("data/curated.market")
    dates = []
    for d in date_strings:
        try:
            dates.append(pd.to_datetime(d).date())
        except Exception:
            pass
    return sorted(dates)


def get_existing_feature_dates():
    """Get dates that already have features computed, from manifest only."""
    manifest_dates = get_manifest_dates(FEATURES_MANIFEST)
    return sorted([pd.to_datetime(d).date() for d in manifest_dates])


def determine_dates_to_process():
    """
    Determine which dates need feature computation.

    Uses manifest dates to compute missing feature partitions.
    """
    curated_dates = get_available_curated_dates()
    if not curated_dates:
        print("ERROR: No curated data available")
        return [], False

    feature_dates = get_existing_feature_dates()
    if feature_dates:
        print(f"found last date in features.L1/: {max(feature_dates)}")
    else:
        print("no existing features found")

    missing_dates = sorted(set(curated_dates) - set(feature_dates))
    if not missing_dates:
        print("features are up to date - no new dates to process")
        return [], False

    print(f"found {len(missing_dates)} new dates to process")
    print(f"  date range: {missing_dates[0]} to {missing_dates[-1]}")
    is_full_rebuild = len(feature_dates) == 0
    return missing_dates, is_full_rebuild


def load_curated_data(date_filter=None):
    """
    Load curated market data, optionally filtered by dates.

    For incremental processing, we need context (60 days lookback) for rolling windows.

    Args:
        date_filter: list of dates to process (None = load all)
    """
    manifest_dates = get_manifest_dates(CURATED_MANIFEST)
    if manifest_dates:
        available_dates = sorted([pd.to_datetime(d).date() for d in manifest_dates])
        all_date_strs = manifest_dates
    else:
        available_dates = []
        all_date_strs = storage.list_partitions("data/curated.market")

    if not all_date_strs:
        raise FileNotFoundError("No curated data found in data/curated.market")

    # If filtering, load context window for rolling calculations
    if date_filter:
        buffer_rows = 70
        if available_dates:
            date_filter = sorted(date_filter)
            start_date = date_filter[0]
            end_date = date_filter[-1]

            if start_date in available_dates:
                start_idx = available_dates.index(start_date)
            else:
                start_idx = max(0, len([d for d in available_dates if d < start_date]) - 1)

            end_idx = max(i for i, d in enumerate(available_dates) if d <= end_date)
            context_start_idx = max(0, start_idx - buffer_rows)
            context_dates = available_dates[context_start_idx:end_idx + 1]
            load_strs = {d.strftime("%Y-%m-%d") for d in context_dates}
        else:
            context_start = min(date_filter) - timedelta(days=150)
            context_end = max(date_filter)
            load_strs = {
                d for d in all_date_strs
                if context_start <= pd.to_datetime(d).date() <= context_end
            }
        files_to_load = [s for s in all_date_strs if s in load_strs]
        print(f"Loading {len(files_to_load)} curated partitions (with {buffer_rows}-row context window)...")
    else:
        files_to_load = all_date_strs
        print(f"Loading {len(files_to_load)} curated partitions (full)...")

    if not files_to_load:
        return pd.DataFrame()

    frames = []
    for d in files_to_load:
        path = f"data/curated.market/date={d}/daily.parquet"
        try:
            frames.append(storage.read_parquet(path))
        except Exception as e:
            print(f"  warning: could not read {path}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    return df


def pivot_curated_data(df):
    """
    Pivot curated data so each row = one trading date.
    This makes rolling calculations much clearer: rolling(60) = 60 trading days.
    
    Input: Long format (5 rows per date, one per symbol)
    Output: Wide format (1 row per date, symbols as column prefixes)
    
    Columns: date, spy_ret, spy_adj_close, tlt_ret, tlt_adj_close, hyg_ret, hyg_adj_close,
             vix, vix3m
    """
    # Pivot returns and prices
    returns = df.pivot(index='date', columns='symbol', values='ret')
    prices = df.pivot(index='date', columns='symbol', values='adj_close')
    
    # Flatten and rename columns
    wide_df = pd.DataFrame({
        'date': returns.index,
        'spy_ret': returns['SPY'].values,
        'spy_adj_close': prices['SPY'].values,
        'tlt_ret': returns['TLT'].values,
        'tlt_adj_close': prices['TLT'].values,
        'hyg_ret': returns['HYG'].values,
        'hyg_adj_close': prices['HYG'].values,
        'vix': prices['^VIX'].values,
        'vix3m': prices['^VIX3M'].values,
    }).reset_index(drop=True)
    
    return wide_df


def compute_spy_returns(df):
    """
    Compute SPY multi-period log returns
    
    Input: Wide-format dataframe with spy_ret column (1 row per date)
    Output: Same format with spy_ret_Xd columns added
    
    Uses pre-computed daily log returns from curated data.
    Multi-period returns are computed by summing daily log returns
    (log returns are additive).
    
    Now rolling(60) means exactly 60 trading days!
    """
    df = df.copy()
    
    for window in [1, 5, 10, 20, 60]:
        col_name = f'spy_ret_{window}d'
        if window == 1:
            df[col_name] = df['spy_ret']
        else:
            df[col_name] = df['spy_ret'].rolling(window).sum()
    
    return df


def compute_spy_volatility(df):
    """
    Compute SPY realized volatility using sqrt(sum(r^2))
    
    Input: Wide-format dataframe with spy_ret column
    Output: Same format with spy_vol_Xd columns added
    
    Realized volatility = sqrt(sum of squared log returns)
    """
    df = df.copy()
    
    for window in [5, 10, 20, 60]:
        col_name = f'spy_vol_{window}d'
        df[col_name] = df['spy_ret'].rolling(window).apply(
            lambda x: np.sqrt(np.sum(x**2)), raw=True
        )
    
    return df


def compute_drawdown(df):
    """Compute 60-day peak-to-trough drawdown for SPY"""
    df = df.copy()
    
    rolling_max = df['spy_adj_close'].rolling(60).max()
    df['drawdown_60d'] = 1 - (df['spy_adj_close'] / rolling_max)
    
    return df


def compute_vix_features(df):
    """Extract VIX, VIX3M, and compute term structure"""
    df = df.copy()
    
    # VIX and VIX3M already in wide format
    df['vix_term'] = df['vix3m'] / df['vix']
    
    return df


def compute_rsi(df, window=14):
    """Compute 14-day RSI for SPY"""
    df = df.copy()
    
    # Calculate price changes
    delta = df['spy_adj_close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    df['rsi_spy_14'] = 100 - (100 / (1 + rs))
    
    return df


def compute_correlations(df):
    """
    Compute rolling correlations between SPY and other assets
    
    Input: Wide-format dataframe with spy_ret, tlt_ret, hyg_ret columns
    Output: Same format with correlation columns added
    """
    df = df.copy()
    
    # 20-day correlations
    df['corr_spy_tlt_20d'] = df['spy_ret'].rolling(20).corr(df['tlt_ret'])
    df['corr_spy_hyg_20d'] = df['spy_ret'].rolling(20).corr(df['hyg_ret'])
    
    # 60-day correlations
    df['corr_spy_tlt_60d'] = df['spy_ret'].rolling(60).corr(df['tlt_ret'])
    df['corr_spy_hyg_60d'] = df['spy_ret'].rolling(60).corr(df['hyg_ret'])
    
    return df


def compute_spreads(df):
    """Compute HYG-TLT spread"""
    df = df.copy()
    
    # HYG-TLT spread (difference in daily returns)
    df['hyg_tlt_spread'] = df['hyg_ret'] - df['tlt_ret']
    
    return df


def build_features(df):
    """
    Build all Level 1 features from curated data.
    
    New approach: Pivot data to wide format (1 row = 1 date) FIRST,
    then compute all features on the wide dataframe.
    This makes rolling calculations crystal clear: rolling(60) = 60 trading days!
    """
    print("\nComputing features...")
    
    print("  - Pivoting data (1 row = 1 trading date)...")
    df = pivot_curated_data(df)
    
    print("  - SPY returns (1d, 5d, 10d, 20d, 60d)...")
    df = compute_spy_returns(df)
    
    print("  - SPY volatility (5d, 10d, 20d, 60d)...")
    df = compute_spy_volatility(df)
    
    print("  - Drawdown (60d)...")
    drawdown = compute_drawdown(df)
    
    print("  - VIX features (vix, vix3m, vix_term)...")
    vix_features = compute_vix_features(df)
    
    print("  - Drawdown (60d)...")
    df = compute_drawdown(df)
    
    print("  - VIX features (vix, vix3m, vix_term)...")
    df = compute_vix_features(df)
    
    print("  - RSI (14d)...")
    df = compute_rsi(df)
    
    print("  - Correlations (20d, 60d)...")
    df = compute_correlations(df)
    
    print("  - Spreads (HYG-TLT)...")
    df = compute_spreads(df)
    
    # Compute RV-VIX spread (needs spy_vol_20d and vix)
    df['rv_vix_spread_20d'] = df['spy_vol_20d'] - df['vix']
    
    # Select final feature columns in desired order
    feature_cols = [
        'date',
        'spy_ret_1d', 'spy_ret_5d', 'spy_ret_10d', 'spy_ret_20d', 'spy_ret_60d',
        'spy_vol_5d', 'spy_vol_10d', 'spy_vol_20d', 'spy_vol_60d',
        'drawdown_60d',
        'vix', 'vix3m', 'vix_term',
        'rsi_spy_14',
        'corr_spy_tlt_20d', 'corr_spy_hyg_20d', 'corr_spy_tlt_60d', 'corr_spy_hyg_60d',
        'hyg_tlt_spread',
        'rv_vix_spread_20d'
    ]
    
    return df[feature_cols]


def write_partitions(df, output_dir="data/features.L1"):
    """Write features partitioned by date using storage layer"""

    # Quick validation before writing
    print("\nValidating features...")
    is_valid, errors = validate_features_batch(df)

    if not is_valid:
        print("Feature validation FAILED:")
        for error in errors:
            print(f"     {error}")
        raise ValueError("Feature validation failed. Aborting write.")

    print("Features validated")

    # Group by date and write partitions
    dates = df['date'].unique()
    print(f"\nWriting {len(dates)} feature partitions...")

    for date in dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        date_df = df[df['date'] == date]

        if len(date_df) != 1:
            raise ValueError(f"Partition {date_str} has {len(date_df)} rows, expected 1")

        outpath = f"{output_dir}/date={date_str}/features.parquet"
        storage.write_parquet(date_df, outpath)

    print(f"Wrote {len(dates)} partitions to {output_dir}")

    # Update manifest: merge existing manifest dates + new writes
    existing_dates = get_manifest_dates(FEATURES_MANIFEST)
    new_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]
    merged = sorted(set(existing_dates) | set(new_dates))

    payload = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "dates": merged
    }
    storage.write_json(payload, FEATURES_MANIFEST)

    print(f"Updated manifest with {len(merged)} total dates")


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
    
    # Load curated data with sufficient context for rolling windows
    print("\nStep 2: Loading curated data...")
    if is_full_rebuild:
        print("  Full rebuild: loading ALL curated data...")
        df = load_curated_data()
    else:
        # Incremental: load new dates + 120-row buffer for rolling features
        print(f"  Incremental: loading {len(dates_to_process)} new dates + 120-row context buffer...")
        df = load_curated_data(date_filter=dates_to_process)
    
    print(f"  Loaded {len(df)} total rows for feature computation")
    
    # Build features on loaded dataset
    print("\nStep 3: Computing features...")
    features_df = build_features(df)
    
    # Filter to only dates we're processing (write only new partitions)
    if not is_full_rebuild:
        features_df['date'] = pd.to_datetime(features_df['date'])
        filter_dates = pd.to_datetime([d for d in dates_to_process])
        features_df = features_df[features_df['date'].isin(filter_dates)]
        print(f"Filtered to {len(features_df)} feature rows for {len(dates_to_process)} target dates")
    
    # Write partitioned output
    print("\nStep 4: Writing features...")
    write_partitions(features_df)
    
    print("\nFeature engineering complete.")
    print("=" * 60)


def lambda_handler(event, context):
    """AWS Lambda entry point."""
    print("=" * 60)
    print("AWS Lambda - Feature Engineering Service")
    print("=" * 60)

    try:
        main()
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Feature engineering completed successfully',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }
        # Chain: trigger vf-prepare-dataset asynchronously
        try:
            import boto3
            boto3.client('lambda', region_name='us-east-1').invoke(
                FunctionName='vf-prepare-dataset',
                InvocationType='Event'
            )
            print("Triggered vf-prepare-dataset")
        except Exception as chain_err:
            print(f"WARNING: Failed to trigger vf-prepare-dataset: {chain_err}")
        return response
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }


if __name__ == "__main__":
    main()
