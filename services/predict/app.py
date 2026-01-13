"""
Prediction Service

loads production model and makes daily volatility predictions
predicts next-day volatility using latest available features
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Local experiment tracker
from experiment_tracker import experiment_run


#configuration
DATA_FEATURES = Path("data/features.L1")
DATA_PREDICTIONS = Path("data/predictions")
EXPERIMENTS_DIR = Path("models/experiments")
EXPERIMENT_NAME = "predictions"


def get_existing_dates(data_dir: Path) -> list:
    """
    get list of dates we already have
    checks partition folders like date=2010-01-04
    """
    if not data_dir.exists():
        return []
    
    dates = []
    for date_folder in data_dir.iterdir():
        if date_folder.is_dir() and date_folder.name.startswith("date="):
            date_str = date_folder.name.replace("date=", "")
            try:
                dates.append(pd.to_datetime(date_str).date())
            except:
                pass
    
    return sorted(dates)


def get_latest_features():
    """
    load most recent features from data/features.L1/
    
    returns: (features_df, features_date)
    """
    if not DATA_FEATURES.exists():
        raise FileNotFoundError(f"features directory not found: {DATA_FEATURES}")
    
    existing_dates = get_existing_dates(DATA_FEATURES)
    
    if not existing_dates:
        raise ValueError("no features found in data/features.L1/")
    
    latest_date = max(existing_dates)
    features_path = DATA_FEATURES / f"date={latest_date}" / "features.parquet"
    
    if not features_path.exists():
        raise FileNotFoundError(f"features file not found: {features_path}")
    
    print(f"loading features from {latest_date}")
    df = pd.read_parquet(features_path)
    
    return df, latest_date


def load_production_model():
    """
    load latest production model from local filesystem
    
    training service saves model to models/latest_ensemble.pkl
    
    returns: (ensemble_dict, model_path)
    """
    model_path = Path("models/latest_ensemble.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run training service first to generate model."
        )
    
    print(f"loading model from {model_path}")
    
    import pickle
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    
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
    DATA_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    
    #create prediction record
    pred_df = pd.DataFrame([{
        'prediction_date': prediction_date,
        'predicted_volatility': prediction,
        'features_date': features_date,
        'model_path': str(model_path),
        'prediction_timestamp': datetime.now()
    }])
    
    #write to partition
    outdir = DATA_PREDICTIONS / f"date={prediction_date}"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "prediction.parquet"
    
    pred_df.to_parquet(outpath, index=False)
    
    print(f"saved prediction to {outpath}")
    
    return outpath


def log_prediction(prediction, prediction_date, features_date, model_path):
    """
    log prediction using simple JSON tracker
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with experiment_run(EXPERIMENT_NAME, run_name=f"prediction-{prediction_date}") as tracker:
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
        
        print(f"✅ Prediction logged to {EXPERIMENTS_DIR}")


def main():
    print("="*60)
    print("Volatility Forecasting - Prediction Service")
    print("="*60)
    
    #load latest features
    print("\nloading features...")
    features_df, features_date = get_latest_features()
    print(f"features shape: {features_df.shape}")
    
    #load production model
    print("\nloading model...")
    ensemble, model_path = load_production_model()
    
    #make prediction
    print("\nmaking prediction...")
    prediction, prediction_date = make_prediction(ensemble, features_df, features_date)
    
    print(f"\nfeatures from: {features_date}")
    print(f"predicting for: {prediction_date}")
    print(f"predicted volatility: {prediction:.6f}")
    
    #save prediction
    print("\nsaving prediction...")
    save_prediction(prediction, prediction_date, features_date, model_path)
    
    #log prediction
    print("\nlogging prediction...")
    log_prediction(prediction, prediction_date, features_date, model_path)
    
    print("\n" + "="*60)
    print("Prediction Complete")
    print("="*60)

if __name__ == "__main__":
    main()
