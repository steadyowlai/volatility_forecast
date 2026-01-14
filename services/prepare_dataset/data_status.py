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
    storage.write_json(status, status_path)


def get_last_date(status_path: str = DEFAULT_STATUS_PATH) -> Optional[str]:
    """
    Get the last date in master_dataset from status file.
    
    Returns:
        Date string (YYYY-MM-DD) or None if not available
    """
    status = get_status(status_path)
    return status.get('last_date')
