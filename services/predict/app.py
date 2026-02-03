"""
Prediction Service - Multi-Model Daily Predictions

Loads all 6 models and makes daily volatility predictions:
- xgboost_90pct, lightgbm_90pct, ensemble_90pct
- xgboost_100pct, lightgbm_100pct, ensemble_100pct

Saves predictions to data/predict/ with date partitions and latest/ folder.
No MLflow - designed for AWS Lambda deployment.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date

from storage import Storage

# Initialize storage
storage = Storage()

# Configuration
DATA_FEATURES = "data/features.L1"
DATA_PREDICTIONS = Path("data/predict")
MODELS_DIR = Path("data/models")


def get_prediction_date():
    """Get current date in YYYY-MM-DD format for date-based storage"""
    return datetime.now().strftime("%Y-%m-%d")


def append_prediction_history(features_date, prediction_date):
    """
    Append prediction run info to prediction_history.jsonl
    
    Args:
        features_date: date features came from
        prediction_date: date being predicted for
    """
    timestamp = datetime.now().isoformat()
    
    history_entry = {
        'timestamp': timestamp,
        'prediction_date': str(prediction_date),
        'features_date': str(features_date)
    }
    
    # Ensure directory exists
    os.makedirs(str(DATA_PREDICTIONS), exist_ok=True)
    
    # Append to JSONL file
    history_file = DATA_PREDICTIONS / "prediction_history.jsonl"
    with open(str(history_file), "a") as f:
        f.write(json.dumps(history_entry) + '\n')
    
    print(f"appended to {history_file}")


def get_last_prediction_date():
    """
    Get the last prediction date from prediction_history.jsonl
    
    Returns:
        date object or None
    """
    history_file = DATA_PREDICTIONS / "prediction_history.jsonl"
    
    if not history_file.exists():
        return None
    
    try:
        with open(str(history_file), 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Get last line
        last_entry = json.loads(lines[-1])
        pred_date_str = last_entry.get('prediction_date')
        
        if pred_date_str:
            return pd.to_datetime(pred_date_str).date()
        
        return None
    except Exception as e:
        print(f"warning: could not read prediction_history.jsonl: {e}")
        return None


def get_latest_features():
    """
    Load most recent features from data/features.L1/
    
    Optimized: Check backwards from today instead of listing 4000+ files
    
    Returns:
        (features_df, features_date)
    """
    # Start from today and check backwards
    check_date = date.today()
    max_days_back = 10
    
    for days_ago in range(max_days_back):
        features_path = f"{DATA_FEATURES}/date={check_date}/features.parquet"
        if storage.exists(features_path):
            print(f"loading features from {check_date}")
            df = storage.read_parquet(features_path)
            return df, check_date
        
        check_date -= timedelta(days=1)
    
    raise ValueError(f"no features found in last {max_days_back} days")


def load_all_models():
    """
    Load all 6 models from data/models/latest/
    
    Returns:
        dict mapping model_name to model artifact
    """
    models = {}
    model_names = [
        'xgboost_90pct',
        'lightgbm_90pct', 
        'ensemble_90pct',
        'xgboost_100pct',
        'lightgbm_100pct',
        'ensemble_100pct'
    ]
    
    for model_name in model_names:
        model_path = f"{MODELS_DIR}/latest/{model_name}.pkl"
        
        if not storage.exists(model_path):
            print(f"warning: {model_name} not found at {model_path}, skipping")
            continue
        
        models[model_name] = storage.read_pickle(model_path)
        print(f"loaded {model_name}")
    
    if not models:
        raise FileNotFoundError("no models found in data/models/latest/")
    
    return models


def make_prediction(model, features_df, model_name):
    """
    Make prediction using a single model
    
    Args:
        model: model artifact dict with 'model' or 'xgb_model', 'lgbm_model', 'meta_model'
        features_df: dataframe with features
        model_name: name of the model
    
    Returns:
        predicted volatility (float)
    """
    feature_cols = model['feature_cols']
    X = features_df[feature_cols].values
    
    # Check if it's an ensemble or individual model
    if 'ensemble' in model_name:
        # Ensemble model: use all 3 models
        xgb_pred = model['xgb_model'].predict(X)
        lgbm_pred = model['lgbm_model'].predict(X)
        X_meta = np.column_stack([xgb_pred, lgbm_pred])
        prediction = model['meta_model'].predict(X_meta)[0]
    else:
        # Individual model: xgboost or lightgbm
        prediction = model['model'].predict(X)[0]
    
    return float(prediction)


def save_predictions(predictions, features_date, prediction_date):
    """
    Save all predictions to date folder and latest folder
    
    Args:
        predictions: dict mapping model_name to prediction value
        features_date: date features came from
        prediction_date: date being predicted for
    """
    timestamp = datetime.now().isoformat()
    pred_date_str = get_prediction_date()
    
    # Ensure parent directory exists
    os.makedirs(str(DATA_PREDICTIONS), exist_ok=True)
    
    # Create date folder
    date_folder = DATA_PREDICTIONS / f"date={pred_date_str}"
    os.makedirs(str(date_folder), exist_ok=True)
    
    # Create latest folder
    latest_folder = DATA_PREDICTIONS / "latest"
    os.makedirs(str(latest_folder), exist_ok=True)
    
    for model_name, predicted_volatility in predictions.items():
        pred_data = {
            'prediction_date': str(prediction_date),
            'predicted_volatility': predicted_volatility,
            'features_date': str(features_date),
            'model_name': model_name,
            'prediction_timestamp': timestamp
        }
        
        # Save to date folder
        date_file = date_folder / f"{model_name}.json"
        with open(str(date_file), 'w') as f:
            json.dump(pred_data, f, indent=2)
        print(f"saved {date_file}")
        
        # Save to latest folder (overwrite)
        latest_file = latest_folder / f"{model_name}.json"
        with open(str(latest_file), 'w') as f:
            json.dump(pred_data, f, indent=2)
    
    print(f"also saved to {latest_folder}/ for easy access")


def main():
    print("\n" + "="*70)
    print("Volatility Forecasting - Multi-Model Prediction Service")
    print("="*70)
    
    # Check what was last predicted
    print("\nChecking prediction status...")
    last_predicted_date = get_last_prediction_date()
    
    if last_predicted_date:
        print(f"last prediction made for: {last_predicted_date}")
    else:
        print("no previous predictions found")
    
    # Load latest features
    print("\nLoading features...")
    features_df, features_date = get_latest_features()
    print(f"features shape: {features_df.shape}")
    print(f"features available for: {features_date}")
    
    # Determine what date we would predict for
    prediction_date = features_date + timedelta(days=1)
    
    # Check if we already predicted this date
    if last_predicted_date and prediction_date <= last_predicted_date:
        print(f"\n⚠️  Already predicted for {prediction_date}")
        print(f"   Features: {features_date}")
        print(f"   Last prediction: {last_predicted_date}")
        print(f"   Nothing new to predict")
        print("\n" + "="*70)
        print("Prediction Service - Already Up-to-Date")
        print("="*70)
        return
    
    # Load all models
    print("\nLoading models...")
    models = load_all_models()
    print(f"loaded {len(models)} models")
    
    # Make predictions with all models
    print("\nMaking predictions...")
    predictions = {}
    
    for model_name, model in models.items():
        prediction = make_prediction(model, features_df, model_name)
        predictions[model_name] = prediction
        print(f"{model_name}: {prediction:.6f}")
    
    print(f"\nfeatures from: {features_date}")
    print(f"predicting for: {prediction_date}")
    
    # Save predictions
    print("\nSaving predictions...")
    save_predictions(predictions, features_date, prediction_date)
    
    # Log to prediction history
    print("\nLogging to prediction history...")
    append_prediction_history(features_date, prediction_date)
    
    # Summary
    print("\n" + "="*70)
    print("Prediction Complete")
    print("="*70)
    print(f"\nPredictions for {prediction_date}:")
    
    # Group by model type
    print("\n90% Split Models:")
    for name in ['xgboost_90pct', 'lightgbm_90pct', 'ensemble_90pct']:
        if name in predictions:
            print(f"  {name}: {predictions[name]:.6f}")
    
    print("\n100% Split Models:")
    for name in ['xgboost_100pct', 'lightgbm_100pct', 'ensemble_100pct']:
        if name in predictions:
            print(f"  {name}: {predictions[name]:.6f}")
    
    print(f"\nArtifacts saved:")
    print(f"data/predict/date={get_prediction_date()}/ (6 JSON files)")
    print(f"data/predict/latest/ (6 JSON files)")
    print(f"data/predict/prediction_history.jsonl")
    
    print("="*70)


if __name__ == "__main__":
    main()
