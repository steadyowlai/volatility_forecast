"""
Prediction Service

loads production model and makes daily volatility predictions
predicts next-day volatility using latest available features

SMART LOGIC:
- Checks prediction_history.jsonl for last predicted date
- Only predicts for dates after last prediction
- Avoids re-predicting same dates
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Local experiment tracker
from experiment_tracker import experiment_run

# Storage abstraction layer
from storage import Storage

# Prediction status tracking
from prediction_status import get_last_prediction_date, log_prediction, get_prediction_history

# Initialize storage
storage = Storage()

#configuration
DATA_FEATURES = "data/features.L1"
DATA_PREDICTIONS = "data/predict"     # Predict service outputs
MODELS_DIR = "data/models"            # Read models from here
EXPERIMENT_NAME = "predictions"


def get_latest_features():
    """
    load most recent features from data/features.L1/
    
    Optimized: Instead of listing all 4000+ files, check backwards from today
    to find the most recent feature date that exists.
    
    returns: (features_df, features_date)
    """
    from datetime import date
    
    # Start from today and check backwards
    # Most likely the latest features are from yesterday or today
    check_date = date.today()
    max_days_back = 10  # Check up to 10 days back
    
    for days_ago in range(max_days_back):
        features_path = f"{DATA_FEATURES}/date={check_date}/features.parquet"
        if storage.exists(features_path):
            print(f"loading features from {check_date}")
            df = storage.read_parquet(features_path)
            return df, check_date
        
        check_date -= timedelta(days=1)
    
    # If nothing found in last 10 days, fall back to scanning all files
    print("warning: no features found in last 10 days, scanning all files...")
    files = storage.list_files(f"{DATA_FEATURES}/")
    
    dates = []
    for file_path in files:
        if "/date=" in file_path and file_path.endswith("features.parquet"):
            try:
                date_str = file_path.split("/date=")[1].split("/")[0]
                dates.append(pd.to_datetime(date_str).date())
            except:
                pass
    
    if not dates:
        raise ValueError("no features found in data/features.L1/")
    
    latest_date = max(dates)
    features_path = f"{DATA_FEATURES}/date={latest_date}/features.parquet"
    
    print(f"loading features from {latest_date}")
    df = storage.read_parquet(features_path)
    
    return df, latest_date


def load_production_model():
    """
    load latest production model from local filesystem
    
    training service saves model to data/models/ensemble_90pct.pkl
    
    returns: (ensemble_dict, model_path)
    """
    model_path = "data/models/ensemble_90pct.pkl"
    
    if not storage.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run training service first to generate model."
        )
    
    print(f"loading model from {model_path}")
    ensemble = storage.read_pickle(model_path)
    print("loaded ensemble successfully")
    
    return ensemble, model_path


def make_prediction(ensemble, features_df, features_date):
    """
    make prediction using ensemble model
    
    ensemble is dict with keys: xgb_model, lgbm_model, meta_model, feature_cols
    
    returns: (prediction_value, prediction_date)
    """
    #extract models and feature columns from ensemble
    xgb_model = ensemble['xgb_model']
    lgbm_model = ensemble['lgbm_model']
    meta_model = ensemble['meta_model']
    feature_cols = ensemble['feature_cols']
    
    #prepare features
    X = features_df[feature_cols].values
    
    #step 1: get base model predictions
    xgb_pred = xgb_model.predict(X)
    lgbm_pred = lgbm_model.predict(X)
    
    #step 2: stack predictions
    X_meta = np.column_stack([xgb_pred, lgbm_pred])
    
    #step 3: final prediction from meta-learner
    prediction = meta_model.predict(X_meta)[0]
    
    #prediction is for next day
    prediction_date = features_date + timedelta(days=1)
    
    return prediction, prediction_date


def save_prediction(prediction, prediction_date, features_date, model_path):
    """
    save prediction to data/predictions/ in partitioned format
    
    saves as date=YYYY-MM-DD/prediction.parquet
    """
    
    #create prediction record
    pred_df = pd.DataFrame([{
        'prediction_date': prediction_date,
        'predicted_volatility': prediction,
        'features_date': features_date,
        'model_path': str(model_path),
        'prediction_timestamp': datetime.now()
    }])
    
    #write to partition
    outpath = f"{DATA_PREDICTIONS}/date={prediction_date}/prediction.parquet"
    storage.write_parquet(pred_df, outpath)
    
    print(f"saved prediction to {outpath}")
    
    return outpath


def log_experiment_prediction(prediction, prediction_date, features_date, model_path):
    """
    log prediction using simple JSON tracker
    This updates predictions_latest.json automatically
    """
    
    with experiment_run(EXPERIMENT_NAME, run_name=f"prediction-{prediction_date}", base_dir=DATA_PREDICTIONS) as tracker:
        # Log prediction metadata
        tracker.log_params({
            'prediction_date': str(prediction_date),
            'features_date': str(features_date),
            'model_path': str(model_path),
            'prediction_timestamp': datetime.now().isoformat()
        })
        
        # Log prediction value
        tracker.log_metrics({
            'predicted_volatility': float(prediction)
        })
        
        print(f"✅ Prediction logged to {DATA_PREDICTIONS}")


def main():
    print("="*60)
    print("Volatility Forecasting - Prediction Service")
    print("="*60)
    
    # Check what was last predicted
    print("\nChecking prediction status...")
    last_predicted_date = get_last_prediction_date()
    
    if last_predicted_date:
        last_pred_date_obj = pd.to_datetime(last_predicted_date).date()
        print(f"last prediction made for: {last_predicted_date}")
    else:
        last_pred_date_obj = None
        print("no previous predictions found")
    
    #load latest features
    print("\nloading features...")
    features_df, features_date = get_latest_features()
    print(f"features shape: {features_df.shape}")
    print(f"features available for: {features_date}")
    
    # Determine what date we would predict for
    prediction_date = features_date + timedelta(days=1)
    
    # Check if we already predicted this date
    if last_pred_date_obj and prediction_date <= last_pred_date_obj:
        print(f"\n⚠️  Already predicted for {prediction_date}")
        print(f"   Last prediction: {last_predicted_date}")
        print(f"   Nothing new to predict")
        print("\n" + "="*60)
        print("Prediction Service - Already Up-to-Date")
        print("="*60)
        return
    
    #load production model
    print("\nloading model...")
    ensemble, model_path = load_production_model()
    
    #make prediction
    print("\nmaking prediction...")
    prediction, prediction_date = make_prediction(ensemble, features_df, features_date)
    
    print(f"\nfeatures from: {features_date}")
    print(f"predicting for: {prediction_date}")
    print(f"predicted volatility: {prediction:.6f}")
    
    #save prediction to partition
    print("\nsaving prediction...")
    save_prediction(prediction, prediction_date, features_date, model_path)
    
    #log to experiment tracker (updates predictions_latest.json)
    print("\nlogging to experiment tracker...")
    log_experiment_prediction(prediction, prediction_date, features_date, model_path)
    
    #log to consolidated history
    print("\nlogging to prediction history...")
    log_prediction(
        prediction_date=str(prediction_date),
        predicted_value=float(prediction),
        features_date=str(features_date),
        model_path=str(model_path)
    )
    
    print("\n" + "="*60)
    print("Prediction Complete")
    print("="*60)
    
    print("\n" + "="*60)
    print("Prediction Complete")
    print("="*60)

if __name__ == "__main__":
    main()
