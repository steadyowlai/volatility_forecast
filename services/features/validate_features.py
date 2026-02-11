"""
Minimal feature validation - just check the essentials.
"""
import pandas as pd


EXPECTED_COLUMNS = [
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


class FeatureValidationError(Exception):
    """Raised when feature validation fails"""
    pass


def validate_feature_partition(df: pd.DataFrame, date_str: str = None) -> None:
    """
    Basic validation for a single feature partition.
    
    Checks:
    1. Exactly 1 row (not 5 rows of raw curated data)
    2. Has 21 columns (date + 20 features)
    3. No 'symbol' column (would indicate raw curated data)
    4. All expected feature columns present
    
    Raises FeatureValidationError if validation fails.
    """
    errors = []
    
    # Check 1: Must be exactly 1 row per partition
    if len(df) != 1:
        errors.append(f"Expected 1 row, got {len(df)} rows")
    
    # Check 2: Must have 21 columns
    if len(df.columns) != 21:
        errors.append(f"Expected 21 columns, got {len(df.columns)}")
    
    # Check 3: Must NOT have 'symbol' column (raw curated data indicator)
    if 'symbol' in df.columns:
        errors.append("Contains 'symbol' column - this is raw curated data, not features!")
    
    # Check 4: Must have all expected columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    if errors:
        error_msg = f"❌ Feature validation failed for {date_str or 'partition'}:\n"
        error_msg += "\n".join(f"     - {err}" for err in errors)
        raise FeatureValidationError(error_msg)


def validate_features_batch(df: pd.DataFrame) -> tuple:
    """
    Validate a batch of features (multiple dates).
    
    Returns (is_valid, errors).
    """
    errors = []
    
    # Check: Must have 21 columns
    if len(df.columns) != 21:
        errors.append(f"Expected 21 columns, got {len(df.columns)}")
    
    # Check: Must NOT have 'symbol' column
    if 'symbol' in df.columns:
        errors.append("Contains 'symbol' column - this is raw curated data, not features!")
    
    # Check: Must have all expected columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check: No duplicate dates
    if len(df) > 1:
        date_counts = df['date'].value_counts()
        duplicates = date_counts[date_counts > 1]
        if len(duplicates) > 0:
            errors.append(f"Found {len(duplicates)} dates with duplicate rows")
    
    is_valid = len(errors) == 0
    return is_valid, errors
