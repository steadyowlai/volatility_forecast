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
from datetime import datetime
from typing import Optional
from storage import Storage

# Initialize storage
storage = Storage()

# Default status file location
DEFAULT_STATUS_PATH = "data/data_status.json"


def get_status(status_path: str = DEFAULT_STATUS_PATH) -> dict:
    """
    Read the current data status.
    
    Returns:
        dict with keys:
            - last_date: Last date in master_dataset.parquet (YYYY-MM-DD)
            - rows: Number of rows in master_dataset
            - updated_at: ISO timestamp of last update
    
    Returns empty dict if file doesn't exist.
    """
    if not storage.exists(status_path):
        return {}
    
    try:
        return storage.read_json(status_path)
    except Exception as e:
        print(f"warning: could not read data_status.json: {e}")
        return {}


def update_status(
    status_path: str = DEFAULT_STATUS_PATH,
    last_date: Optional[str] = None,
    rows: Optional[int] = None,
    # New parameters for tracking both datasets
    master_last_date: Optional[str] = None,
    master_rows: Optional[int] = None,
    predict_last_date: Optional[str] = None,
    predict_rows: Optional[int] = None,
    predict_recent_start: Optional[str] = None,
    predict_recent_end: Optional[str] = None,
    predict_future_start: Optional[str] = None,
    predict_future_end: Optional[str] = None,
) -> None:
    """
    Update the data status file.
    
    ONLY prepare_dataset service should call this!
    
    Args:
        status_path: Path to status file
        last_date: (DEPRECATED) Last date in master_dataset (YYYY-MM-DD)
        rows: (DEPRECATED) Number of rows in master_dataset
        master_last_date: Last date in master_dataset (YYYY-MM-DD)
        master_rows: Number of rows in master_dataset
        predict_last_date: Last date in predict_dataset (YYYY-MM-DD)
        predict_rows: Number of rows in predict_dataset
        predict_recent_start: Start date of recent window (last 30 rows)
        predict_recent_end: End date of recent window
        predict_future_start: Start date of future window (optional)
        predict_future_end: End date of future window (optional)
    """
    # Read existing status
    status = get_status(status_path)
    
    # Initialize nested structures if not present
    if 'master_dataset' not in status:
        status['master_dataset'] = {}
    if 'predict_dataset' not in status:
        status['predict_dataset'] = {}
    
    # Update master_dataset fields
    if master_last_date is not None:
        status['master_dataset']['last_date'] = str(master_last_date)
    if master_rows is not None:
        status['master_dataset']['rows'] = master_rows
    if master_last_date is not None or master_rows is not None:
        status['master_dataset']['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    
    # Update predict_dataset fields
    if predict_last_date is not None:
        status['predict_dataset']['last_date'] = str(predict_last_date)
    if predict_rows is not None:
        status['predict_dataset']['rows'] = predict_rows
    if predict_recent_start is not None:
        status['predict_dataset']['recent_start'] = str(predict_recent_start)
    if predict_recent_end is not None:
        status['predict_dataset']['recent_end'] = str(predict_recent_end)
    if predict_future_start is not None:
        status['predict_dataset']['future_start'] = str(predict_future_start)
    if predict_future_end is not None:
        status['predict_dataset']['future_end'] = str(predict_future_end)
    if any([predict_last_date, predict_rows, predict_recent_start, predict_recent_end, 
            predict_future_start, predict_future_end]):
        status['predict_dataset']['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    
    # Legacy support: update top-level fields if old parameters used
    if last_date is not None:
        status['last_date'] = str(last_date)
        status['master_dataset']['last_date'] = str(last_date)
    if rows is not None:
        status['rows'] = rows
        status['master_dataset']['rows'] = rows
    
    # Always update top-level timestamp
    status['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    
    # Write back to file
    storage.write_json(status, status_path)


def get_last_date(status_path: str = DEFAULT_STATUS_PATH) -> Optional[str]:
    """
    Get the last date in master_dataset from status file.
    
    Returns:
        Date string (YYYY-MM-DD) or None if not available
    """
    status = get_status(status_path)
    
    # Try nested structure first (new format)
    if 'master_dataset' in status and 'last_date' in status['master_dataset']:
        return status['master_dataset']['last_date']
    
    # Fall back to legacy format
    return status.get('last_date')


def get_master_last_date(status_path: str = DEFAULT_STATUS_PATH) -> Optional[str]:
    """
    Get the last date in master_dataset from status file.
    
    Returns:
        Date string (YYYY-MM-DD) or None if not available
    """
    return get_last_date(status_path)


def get_predict_last_date(status_path: str = DEFAULT_STATUS_PATH) -> Optional[str]:
    """
    Get the last date in predict_dataset from status file.
    
    Returns:
        Date string (YYYY-MM-DD) or None if not available
    """
    status = get_status(status_path)
    if 'predict_dataset' in status:
        return status['predict_dataset'].get('last_date')
    return None
