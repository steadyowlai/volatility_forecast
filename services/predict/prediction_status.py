"""
Prediction Status Tracking Module

Works with existing predictions_latest.json from experiment_tracker.
Adds prediction_history.jsonl for consolidated history.

prediction_latest.json: Maintained by experiment_tracker (last prediction)
prediction_history.jsonl: Consolidated log of all predictions
"""

import json
from datetime import datetime
from typing import Optional, List, Dict
from storage import Storage

# Initialize storage
storage = Storage()

# Prediction files
PREDICTION_LATEST = "models/experiments/predictions_latest.json"
PREDICTION_HISTORY = "data/predictions/prediction_history.jsonl"


def get_last_prediction_date() -> Optional[str]:
    """
    Get the last date we made a prediction for.
    Reads from predictions_latest.json (maintained by experiment_tracker).
    
    Returns:
        Date string (YYYY-MM-DD) or None if no predictions yet
    """
    if not storage.exists(PREDICTION_LATEST):
        return None
    
    try:
        latest = storage.read_json(PREDICTION_LATEST)
        # Extract prediction_date from params
        return latest.get('params', {}).get('prediction_date')
    except Exception as e:
        print(f"warning: could not read predictions_latest.json: {e}")
        return None


def log_prediction(
    prediction_date: str,
    predicted_value: float,
    features_date: str = None,
    model_path: str = None
) -> None:
    """
    Log a prediction to the history file.
    This is SEPARATE from experiment_tracker logging.
    
    Only logs essential fields: prediction_date and predicted_volatility
    
    Args:
        prediction_date: Date being predicted (YYYY-MM-DD)
        predicted_value: Predicted volatility value
        features_date: (unused, for backward compatibility)
        model_path: (unused, for backward compatibility)
    """
    record = {
        'prediction_date': prediction_date,
        'predicted_volatility': float(predicted_value)
    }
    
    # Append to jsonl
    storage.append_jsonl(record, PREDICTION_HISTORY)


def get_prediction_history(limit: Optional[int] = None) -> List[Dict]:
    """
    Get prediction history from consolidated log.
    
    Args:
        limit: Number of most recent predictions to return (None = all)
    
    Returns:
        List of prediction records (most recent first if limit specified)
    """
    if not storage.exists(PREDICTION_HISTORY):
        return []
    
    try:
        content = storage.read_text(PREDICTION_HISTORY)
        lines = [line for line in content.strip().split('\n') if line.strip()]
        
        records = [json.loads(line) for line in lines]
        
        if limit:
            return records[-limit:][::-1]  # Last N, reversed
        
        return records
    except Exception as e:
        print(f"warning: could not read prediction_history.jsonl: {e}")
        return []
