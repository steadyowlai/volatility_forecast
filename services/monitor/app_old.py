"""
Production Monitoring Service

Monitors model performance on the latest 10% of available data to detect drift.

Monitoring Strategy:
- Load ALL available data (features + actuals)
- Take the latest 10% by date (rolling window)
- Generate predictions on this window using the trained model
- Compare RMSE to training validation baseline
- If significant drift detected → trigger retraining

This differs from training validation:
- Training validation: One-time validation during model training
- Production monitoring: Continuous validation on rolling window of recent data

Purpose: Detect when model performance degrades and trigger retraining
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#configuration
DATA_CURATED = Path("data/curated.market")
DATA_FEATURES = Path("data/features.L1")
MODELS_DIR = Path("models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def load_validation_baseline():
    """
    Load validation baseline from training service.
    This is the RMSE from the most recent training run on historical validation data (90/10 split).
    """
    baseline_path = MODELS_DIR / "validation_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Validation baseline not found at {baseline_path}. "
            f"Run training service first to generate baseline."
        )
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    return baseline


def load_trained_model():
    """
    Load the trained ensemble model for making predictions.
    """
    model_path = MODELS_DIR / "latest_ensemble.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)
    
    return model_artifact


def load_all_features():
    """
    Load ALL available features (entire dataset including new data).
    """
    features_path = DATA_FEATURES
    files = list(features_path.glob("date=*/features.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No features found in {features_path}")
    
    print(f"loading {len(files)} feature partitions...")
    
    dfs = []
    skipped = 0
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            skipped += 1
    
    if skipped > 0:
        print(f"skipped {skipped} corrupted files")
    
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"loaded {len(df)} samples from {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def load_all_actuals():
    """
    Load ALL available actuals (realized volatility from curated market data).
    """
    if not DATA_CURATED.exists():
        raise FileNotFoundError(f"curated data directory not found: {DATA_CURATED}")
    
    curated_files = list(DATA_CURATED.glob("date=*/daily.parquet"))
    
    if not curated_files:
        raise ValueError("no curated data found")
    
    print(f"loading {len(curated_files)} curated partitions...")
    
    dfs = []
    skipped = 0
    for f in curated_files:
        try:
            dfs.append(pd.read_parquet(f))
        except:
            skipped += 1
    
    if skipped > 0:
        print(f"skipped {skipped} corrupted files")
    
    if not dfs:
        raise ValueError("no valid curated files found")
    
    actuals_df = pd.concat(dfs, ignore_index=True)
    actuals_df['date'] = pd.to_datetime(actuals_df['date'])
    actuals_df = actuals_df.sort_values('date')
    
    #compute 5-day realized volatility (forward-looking)
    actuals_df['ret_sq'] = actuals_df['ret'] ** 2
    actuals_df['rv_5d'] = actuals_df['ret_sq'].rolling(window=5).sum().shift(-4)
    actuals_df['rv_5d'] = np.sqrt(actuals_df['rv_5d'])
    
    #keep only date and rv_5d
    actuals_df = actuals_df[['date', 'rv_5d']].dropna()
    
    print(f"computed rv_5d for {len(actuals_df)} dates from {actuals_df['date'].min().date()} to {actuals_df['date'].max().date()}")
    
    return actuals_df


def prepare_monitoring_dataset(features_df, actuals_df, window_pct=0.1):
    """
    Prepare the monitoring dataset: latest 10% of data with features and actuals.
    
    Args:
        features_df: All available features
        actuals_df: All available actuals
        window_pct: Percentage of data to use for monitoring (default 0.1 = 10%)
    
    Returns:
        DataFrame with features and actual RV for monitoring window
    """
    #merge features with actuals
    merged = features_df.merge(actuals_df, on='date', how='inner')
    
    #drop any NaN values
    merged = merged.dropna()
    
    print(f"\nmerged dataset: {len(merged)} samples")
    
    #calculate monitoring window (latest 10% by date)
    merged = merged.sort_values('date').reset_index(drop=True)
    total_samples = len(merged)
    window_start_idx = int(total_samples * (1 - window_pct))
    
    monitoring_df = merged.iloc[window_start_idx:].copy()
    
    print(f"\nmonitoring window (latest {int(window_pct*100)}%):")
    print(f"  samples: {len(monitoring_df)}")
    print(f"  date range: {monitoring_df['date'].min().date()} to {monitoring_df['date'].max().date()}")
    print(f"  coverage: {len(monitoring_df)/total_samples*100:.1f}% of all data")
    
    return monitoring_df


def make_predictions(monitoring_df, model_artifact):
    """
    Generate predictions on monitoring window using trained model.
    
    Args:
        monitoring_df: DataFrame with features and actuals
        model_artifact: Loaded model from pickle (dict with xgb, lgbm, meta models)
    
    Returns:
        predictions array
    """
    feature_cols = model_artifact['feature_cols']
    xgb_model = model_artifact['xgb_model']
    lgbm_model = model_artifact['lgbm_model']
    meta_model = model_artifact['meta_model']
    
    X = monitoring_df[feature_cols].values
    
    #step 1: base model predictions
    xgb_pred = xgb_model.predict(X)
    lgbm_pred = lgbm_model.predict(X)
    
    #step 2: stack predictions for meta-learner
    X_meta = np.column_stack([xgb_pred, lgbm_pred])
    
    #step 3: final ensemble prediction
    predictions = meta_model.predict(X_meta)
    
    return predictions


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    
    Returns: dict with rmse, mae, r2, mape, etc
    """
    #basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    #mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    #directional accuracy
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        if len(true_direction) > 0:
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = None
    else:
        directional_accuracy = None
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'n_samples': len(y_true)
    }
    
    return metrics


def detect_performance_drift(rmse, baseline_rmse):
    """
    Check if model performance has degraded significantly.
    
    With a 10% window (400+ samples), we have statistical significance.
    Thresholds are set to balance sensitivity vs false positives.
    
    Thresholds:
    - >20%: Severe drift - retrain immediately
    - 10-20%: Moderate drift - investigate and plan retraining
    - 5-10%: Minor drift - monitor closely
    - <5%: Normal variation - ok
    
    Args:
        rmse: Current monitoring window RMSE
        baseline_rmse: Training validation RMSE baseline
    
    Returns:
        tuple: (alert_level, message, rmse_vs_baseline_pct)
    """
    rmse_vs_baseline = ((rmse / baseline_rmse) - 1) * 100
    """
    Check if model performance has degraded significantly.
    
    With a rolling window of predictions (20+ samples), we can reliably detect drift.
    Thresholds are set to balance sensitivity vs false positives.
    
    Thresholds:
    - >20%: Severe drift - retrain immediately
    - 10-20%: Moderate drift - investigate and plan retraining
    - 5-10%: Minor drift - monitor closely
    - <5%: Normal variation - ok
    
    Args:
        metrics: Dict with model performance metrics
        baseline_rmse: Training validation RMSE baseline
    
    Returns:
        tuple: (alert_level, message, rmse_vs_baseline_pct)
    """
    rmse_vs_baseline = ((metrics['rmse'] / baseline_rmse) - 1) * 100
    
    if rmse_vs_baseline > 20:
        alert_level = 'critical'
        message = f"CRITICAL: performance degraded by {rmse_vs_baseline:.1f}% - retrain immediately"
    elif rmse_vs_baseline > 10:
        alert_level = 'warning'
        message = f"warning: performance degraded by {rmse_vs_baseline:.1f}% - investigate and plan retraining"
    elif rmse_vs_baseline > 5:
        alert_level = 'monitor'
        message = f"minor drift detected: {rmse_vs_baseline:.1f}% - monitor closely"
    else:
        alert_level = 'ok'
        message = f"performance stable: {rmse_vs_baseline:+.1f}% vs training baseline"
    
    return alert_level, message, rmse_vs_baseline


def log_to_mlflow(metrics, baseline_rmse, rmse_vs_baseline, alert_level, alert_message, merged_df):
    """
    log validation results to mlflow
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("volatility-forecasting-validation")
    
    #get date range for this validation run
    date_range_start = merged_df['prediction_date'].min()
    date_range_end = merged_df['prediction_date'].max()
    
    with mlflow.start_run(run_name=f"validation-{datetime.now().strftime('%Y%m%d')}"):
        #log metrics
        mlflow.log_metrics({
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'mape': metrics['mape'],
            'n_predictions': metrics['n_predictions'],
            'baseline_rmse': baseline_rmse,
            'rmse_vs_baseline_pct': rmse_vs_baseline
        })
        
        if metrics['directional_accuracy'] is not None:
            mlflow.log_metric('directional_accuracy', metrics['directional_accuracy'])
        
        #log parameters
        mlflow.log_params({
            'validation_date': datetime.now().strftime('%Y-%m-%d'),
            'date_range_start': str(date_range_start.date()),
            'date_range_end': str(date_range_end.date()),
            'alert_level': alert_level,
            'alert_message': alert_message
        })
        
        #log alert tag for easy filtering
        mlflow.set_tag('performance_alert', alert_level)
        
        run_id = mlflow.active_run().info.run_id
        print(f"logged to mlflow run: {run_id}")


def main():
    print("="*60)
    print("Volatility Forecasting - Production Monitoring Service")
    print("="*60)
    print("\nChecking predictions from 5+ days ago (where actuals are now available)...")
    
    #load validation baseline
    print("\nloading validation baseline from training...")
    try:
        baseline = load_validation_baseline()
        baseline_rmse = baseline['validation_rmse']
        print(f"training validation RMSE: {baseline_rmse:.6f}")
        print(f"baseline from: {baseline['timestamp'][:10]}")
    except FileNotFoundError as e:
        print(f"error: {e}")
        return
    
    #load predictions
    print("\nloading predictions...")
    predictions_df = load_predictions()
    
    #load actuals
    print("\nloading actual volatility values...")
    actuals_df = load_actuals()
    
    #merge predictions with actuals
    print("\nmerging predictions with actuals...")
    merged_df = merge_predictions_with_actuals(predictions_df, actuals_df)
    
    if len(merged_df) == 0:
        print("\n" + "="*60)
        print("NO PREDICTIONS TO MONITOR YET")
        print("="*60)
        print("\nPredictions need actual outcomes to validate against.")
        print("We need 5 days of future data to compute actual rv_5d.")
        print("\nThis is normal for fresh predictions - check back in 5 days!")
        print("="*60)
        return
    
    #select monitoring window (most recent 10% of predictions with actuals)
    print("\nselecting monitoring window...")
    print(f"total predictions with actuals: {len(merged_df)}")
    
    windowed_df = select_monitoring_window(merged_df, window_pct=0.1, min_samples=20)
    
    if windowed_df is None or len(windowed_df) == 0:
        print("\nno predictions available for monitoring window")
        return
    
    print(f"\nmonitoring window:")
    print(f"  window size: {len(windowed_df)} predictions ({len(windowed_df)/len(merged_df)*100:.1f}% of available)")
    print(f"  date range: {windowed_df['prediction_date'].min().date()} to {windowed_df['prediction_date'].max().date()}")
    
    #calculate metrics on windowed data
    print("\ncalculating metrics on monitoring window...")
    metrics = calculate_metrics(windowed_df)
    
    #detect drift
    alert_level, alert_message, rmse_vs_baseline = detect_performance_drift(metrics, baseline_rmse)
    
    #print results
    print("\n" + "="*60)
    print("PRODUCTION MONITORING - DRIFT DETECTION")
    print("="*60)
    print(f"\nMonitoring window: {len(windowed_df)} predictions")
    print(f"  Date range: {windowed_df['prediction_date'].min().date()} to {windowed_df['prediction_date'].max().date()}")
    print(f"  Coverage: {len(windowed_df)/len(merged_df)*100:.1f}% of available predictions")
    
    print(f"\nLive predictions RMSE: {metrics['rmse']:.6f}")
    print(f"  Live predictions MAE:  {metrics['mae']:.6f}")
    print(f"  Live predictions R2:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    if metrics['directional_accuracy'] is not None:
        print(f"  Directional accuracy: {metrics['directional_accuracy']:.1f}%")
    
    print(f"\nBaseline comparison:")
    print(f"  Training validation RMSE: {baseline_rmse:.6f}")
    print(f"  Live predictions RMSE:    {metrics['rmse']:.6f}")
    print(f"  Drift: {rmse_vs_baseline:+.1f}%")
    
    if alert_level == "ok":
        print(f"\n{alert_message}")
    elif alert_level == "monitor":
        print(f"\n{alert_message}")
    else:
        print(f"\n{alert_message}")
    
    print("="*60)
    
    #log to mlflow
    print("\nlogging to mlflow...")
    log_to_mlflow(metrics, baseline_rmse, rmse_vs_baseline, alert_level, alert_message, windowed_df)
    
    print("\n" + "="*60)
    print("Monitoring Complete")
    print("="*60)


if __name__ == "__main__":
    main()
