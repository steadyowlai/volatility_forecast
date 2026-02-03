"""
Data Status Tracking Module

Maintains a lightweight metadata file (data_status.json) to track the state
of the master_dataset.parquet without having to read the large parquet file.

IMPORTANT: Only prepare_dataset service should UPDATE this file.
All other services should only READ from it.

The single source of truth is master_dataset.parquet.
This JSON is just a performance optimization to avoid reading the parquet file.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


# Default status file location
DEFAULT_STATUS_PATH = Path("data/data_status.json")


def get_status(status_path: Path = DEFAULT_STATUS_PATH) -> dict:
    """
    Read the current data status.
    
    Returns:
        dict with nested structure:
            - master_dataset:
                - last_date: Last date in master_dataset.parquet (YYYY-MM-DD)
                - rows: Number of rows in master_dataset
                - updated_at: ISO timestamp of last update
            - predict_dataset:
                - last_date: Last date in predict_dataset.parquet
                - rows: Number of rows in predict_dataset
                - recent_start/end: Date range of recent window (last 30 rows)
                - future_start/end: Date range of future dates (optional)
                - updated_at: ISO timestamp of last update
            
        Also includes legacy top-level fields for backward compatibility:
            - last_date: Same as master_dataset.last_date
            - rows: Same as master_dataset.rows
            - updated_at: Timestamp
    
    Returns empty dict if file doesn't exist.
    """
    if not status_path.exists():
        return {}
    
    try:
        with open(status_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"warning: could not read data_status.json: {e}")
        return {}


def update_status(
    status_path: Path = DEFAULT_STATUS_PATH,
    last_date: Optional[str] = None,
    rows: Optional[int] = None,
) -> None:
    """
    Update the data status file.
    
    ONLY prepare_dataset service should call this!
    
    Args:
        status_path: Path to status file
        last_date: Last date in master_dataset (YYYY-MM-DD)
        rows: Number of rows in master_dataset
    """
    # Read existing status
    status = get_status(status_path)
    
    # Update fields
    if last_date is not None:
        status['last_date'] = str(last_date)
    
    if rows is not None:
        status['rows'] = rows
    
    # Always update timestamp
    status['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    
    # Write back to file
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)

    
    # Always update timestamp
    status['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    
    # Write back to file
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)


def get_last_date(status_path: Path = DEFAULT_STATUS_PATH) -> Optional[str]:
    """
    Get the last date in master_dataset from status file.
    
    Supports both new nested structure and legacy format:
    - New: status['master_dataset']['last_date']
    - Legacy: status['last_date']
    
    Returns:
        Date string (YYYY-MM-DD) or None if not available
    """
    status = get_status(status_path)
    
    # Try new nested structure first
    if 'master_dataset' in status and 'last_date' in status['master_dataset']:
        return status['master_dataset']['last_date']
    
    # Fallback to legacy top-level field
    return status.get('last_date')
