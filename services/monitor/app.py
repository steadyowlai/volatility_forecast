"""
Production Monitoring Service - Dual Model Strategy

Monitors model performance using two complementary approaches:

1. DRIFT DETECTION (using Model A - 90%):
   - Model A trained on 90% of data, validated on 10% holdout
   - Track Model A's performance on the validation set (10%) over time
   - Detect if performance degrades vs baseline
   
2. MODEL COMPARISON (both models on new predictions):
   - Every time we get NEW actual data (ground truth)
   - Compare Model A (90%) vs Model B (100%) predictions
   - Track which model performs better on truly new data
   - Decision: Use Model B if it consistently outperforms Model A

Purpose: 
- Detect when model performance degrades (drift)
- Empirically determine which model (90% vs 100%) performs better on new data
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Configuration
DATA_CURATED = Path("data/curated.market")
DATA_FEATURES = Path("data/features.L1")
MASTER_DATASET = Path("data/master_dataset.parquet")
MODELS_DIR = Path("models")
MONITORING_HISTORY = MODELS_DIR / "monitoring_history.jsonl"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def load_validation_baseline():
    """Load validation baseline from training service."""
    baseline_path = MODELS_DIR / "validation_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Validation baseline not found at {baseline_path}. "
            f"Run training service first to generate baseline."
        )
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    return baseline


def load_dual_model_baseline():
    """Load dual model baseline with both Model A and Model B info."""
    baseline_path = MODELS_DIR / "dual_model_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Dual model baseline not found at {baseline_path}. "
            f"Run training service first to generate baseline."
        )
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    return baseline


def load_trained_model(model_name="latest_ensemble"):
    """
    Load a trained ensemble model.
    
    Args:
        model_name: Name of the model file (without .pkl extension)
                   Options: "latest_ensemble", "model_90pct", "model_100pct"
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)
    
    return model_artifact


def load_master_dataset():
    """
    Load pre-computed master dataset from prepare_dataset service.
    
    This ensures consistency with training - both use the exact same data source.
    """
    if not MASTER_DATASET.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {MASTER_DATASET}. "
            f"Run prepare_dataset service first: docker-compose up prepare_dataset"
        )
    
    print(f"loading master dataset from {MASTER_DATASET}...")
    df = pd.read_parquet(MASTER_DATASET)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"loaded {len(df)} samples from {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def save_monitoring_record(dual_baseline, comparisons, drift_pct_model_a, recommendation, test_window_info):
    """
    Save monitoring results to historical log (JSONL format).
    
    Each line is a JSON record of one monitoring run. This allows tracking:
    - Performance trends over time as models are retrained
    - How each training iteration performs on its respective test set
    - Whether model quality is degrading or improving
    
    Args:
        dual_baseline: Model training info (dates, sample counts)
        comparisons: List of comparison results on test windows
        drift_pct_model_a: Drift percentage for Model A
        recommendation: Which model to use (A or B)
        test_window_info: Info about the test data used
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    record = {
        'timestamp': datetime.now().isoformat(),
        'monitoring_date': datetime.now().strftime('%Y-%m-%d'),
        
        # Model training info
        'model_a': {
            'training_samples': dual_baseline['model_a']['training_samples'],
            'training_date_start': dual_baseline['model_a']['training_date_start'],
            'training_date_end': dual_baseline['model_a']['training_date_end'],
        },
        'model_b': {
            'training_samples': dual_baseline['model_b']['training_samples'],
            'training_date_start': dual_baseline['model_b']['training_date_start'],
            'training_date_end': dual_baseline['model_b']['training_date_end'],
        },
        
        # Test data info
        'test_window': {
            'n_samples': test_window_info['n_samples'],
            'date_start': test_window_info['date_start'],
            'date_end': test_window_info['date_end'],
        },
        
        # Performance results
        'drift': {
            'model_a_drift_pct': drift_pct_model_a,
        },
        'comparison': {
            'model_a_rmse': comparisons[0]['model_a_rmse'] if comparisons else None,
            'model_b_rmse': comparisons[0]['model_b_rmse'] if comparisons else None,
            'winner': comparisons[0]['winner'] if comparisons else None,
            'model_b_win_rate': sum(1 for c in comparisons if c['winner'] == 'Model B') / len(comparisons) * 100 if comparisons else 0,
        },
        'recommendation': recommendation,
        
        # All window comparisons
        'windows': [
            {
                'name': c['window_name'],
                'n_samples': c['n_samples'],
                'model_a_rmse': c['model_a_rmse'],
                'model_b_rmse': c['model_b_rmse'],
                'winner': c['winner']
            }
            for c in comparisons
        ] if comparisons else []
    }
    
    # Append to JSONL file (one JSON per line)
    with open(MONITORING_HISTORY, 'a') as f:
        f.write(json.dumps(record) + '\n')
    
    print(f"✓ Saved monitoring record to {MONITORING_HISTORY}")


def load_monitoring_history():
    """
    Load historical monitoring records.
    
    Returns:
        List of monitoring records (dicts), or empty list if no history
    """
    if not MONITORING_HISTORY.exists():
        return []
    
    records = []
    with open(MONITORING_HISTORY, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    return records


def analyze_monitoring_trends(history):
    """
    Analyze trends in monitoring history.
    
    Shows:
    - Performance trend over time
    - Whether models are improving or degrading
    - Historical comparison of training iterations
    """
    if len(history) == 0:
        print("\n📊 No historical monitoring data available yet")
        return
    
    print("\n" + "="*70)
    print("📊 MONITORING HISTORY - Performance Trends")
    print("="*70)
    
    print(f"\nTotal monitoring runs: {len(history)}")
    
    # Show last 5 runs
    recent_runs = history[-5:]
    
    print("\n" + "-"*70)
    print(f"{'Date':<12} {'Model A RMSE':>13} {'Model B RMSE':>13} {'Winner':>10} {'Recommendation':>15}")
    print("-"*70)
    
    for record in recent_runs:
        date = record['monitoring_date']
        model_a_rmse = record['comparison']['model_a_rmse']
        model_b_rmse = record['comparison']['model_b_rmse']
        winner = record['comparison']['winner'] if record['comparison']['winner'] else 'N/A'
        rec = 'Model A' if 'MODEL A' in record['recommendation'] else 'Model B'
        
        if model_a_rmse and model_b_rmse:
            print(f"{date:<12} {model_a_rmse:>13.6f} {model_b_rmse:>13.6f} {winner:>10} {rec:>15}")
    
    print("-"*70)
    
    # Trend analysis
    if len(history) >= 2:
        print("\n📈 Trend Analysis:")
        
        # Model A performance trend
        recent_a_rmse = [r['comparison']['model_a_rmse'] for r in history[-3:] if r['comparison']['model_a_rmse']]
        if len(recent_a_rmse) >= 2:
            trend_a = "improving" if recent_a_rmse[-1] < recent_a_rmse[0] else "degrading"
            change_a = ((recent_a_rmse[-1] / recent_a_rmse[0]) - 1) * 100
            print(f"  Model A: {trend_a} ({change_a:+.2f}% over last {len(recent_a_rmse)} runs)")
        
        # Model B performance trend
        recent_b_rmse = [r['comparison']['model_b_rmse'] for r in history[-3:] if r['comparison']['model_b_rmse']]
        if len(recent_b_rmse) >= 2:
            trend_b = "improving" if recent_b_rmse[-1] < recent_b_rmse[0] else "degrading"
            change_b = ((recent_b_rmse[-1] / recent_b_rmse[0]) - 1) * 100
            print(f"  Model B: {trend_b} ({change_b:+.2f}% over last {len(recent_b_rmse)} runs)")
    
    print("="*70)


def make_predictions(monitoring_df, model_artifact):
    """
    Generate predictions on monitoring window using trained model.
    
    Args:
        monitoring_df: DataFrame with features and actuals
        model_artifact: Loaded model from pickle
    
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
    """Calculate performance metrics."""
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
    
    Thresholds:
    - >20%: Severe drift - retrain immediately
    - 10-20%: Moderate drift - investigate and plan retraining
    - 5-10%: Minor drift - monitor closely
    - <5%: Normal variation - ok
    """
    rmse_vs_baseline = ((rmse / baseline_rmse) - 1) * 100
    
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


def compare_models_on_window(monitoring_df, model_a_artifact, model_b_artifact, window_name, 
                            model_a_cutoff, model_b_cutoff):
    """
    Compare Model A and Model B on a monitoring window.
    
    SAFETY: Filters out any data before training cutoff dates to ensure
    we only test on data the models have NOT seen during training.
    
    Args:
        monitoring_df: DataFrame with features and actuals
        model_a_artifact: Model A (90%) artifact
        model_b_artifact: Model B (100%) artifact
        window_name: Name of the window for display
        model_a_cutoff: Last date Model A was trained on (exclusive)
        model_b_cutoff: Last date Model B was trained on (exclusive)
    
    Returns:
        dict with comparison results
    """
    # SAFETY CHECK: Filter out training data
    # Only use data AFTER the latest cutoff date
    latest_cutoff = max(model_a_cutoff, model_b_cutoff)
    monitoring_df_filtered = monitoring_df[monitoring_df['date'] > latest_cutoff].copy()
    
    if len(monitoring_df_filtered) == 0:
        # No valid data after cutoff
        return None
    
    actuals = monitoring_df_filtered['rv_5d'].values
    
    # Generate predictions from both models
    pred_a = make_predictions(monitoring_df_filtered, model_a_artifact)
    pred_b = make_predictions(monitoring_df_filtered, model_b_artifact)
    
    # Calculate metrics for both
    metrics_a = calculate_metrics(actuals, pred_a)
    metrics_b = calculate_metrics(actuals, pred_b)
    
    # Determine winner
    winner = "Model B" if metrics_b['rmse'] < metrics_a['rmse'] else "Model A"
    rmse_diff_pct = ((metrics_b['rmse'] / metrics_a['rmse']) - 1) * 100
    
    return {
        'window_name': window_name,
        'n_samples': len(monitoring_df_filtered),
        'date_start': monitoring_df_filtered['date'].min(),
        'date_end': monitoring_df_filtered['date'].max(),
        'model_a_rmse': metrics_a['rmse'],
        'model_b_rmse': metrics_b['rmse'],
        'winner': winner,
        'rmse_diff_pct': rmse_diff_pct,
        'metrics_a': metrics_a,
        'metrics_b': metrics_b
    }


def print_model_comparison_summary(comparisons, dual_baseline, drift_pct_model_a):
    """
    Print a summary of dual model comparisons.
    
    Returns:
        recommendation: String indicating which model to use
    """
    print("\n" + "="*70)
    print("DUAL MODEL COMPARISON - Model A (90%) vs Model B (100%)")
    print("="*70)
    
    print("\nModel Training Info:")
    print(f"  Model A: {dual_baseline['model_a']['training_samples']} samples "
          f"({dual_baseline['model_a']['training_date_start']} to {dual_baseline['model_a']['training_date_end']})")
    print(f"  Model B: {dual_baseline['model_b']['training_samples']} samples "
          f"({dual_baseline['model_b']['training_date_start']} to {dual_baseline['model_b']['training_date_end']})")
    
    print("\n" + "-"*70)
    print(f"{'Window':<20} {'Samples':>8} {'Model A RMSE':>13} {'Model B RMSE':>13} {'Winner':>10}")
    print("-"*70)
    
    model_b_wins = 0
    total_windows = len(comparisons)
    
    for comp in comparisons:
        winner_symbol = "✓" if comp['winner'] == "Model B" else "✓"
        print(f"{comp['window_name']:<20} {comp['n_samples']:>8} "
              f"{comp['model_a_rmse']:>13.6f} {comp['model_b_rmse']:>13.6f} "
              f"{comp['winner']:>9} {winner_symbol}")
        
        if comp['winner'] == "Model B":
            model_b_wins += 1
    
    print("-"*70)
    
    model_b_win_rate = (model_b_wins / total_windows) * 100 if total_windows > 0 else 0
    
    print(f"\nModel B Win Rate: {model_b_wins}/{total_windows} ({model_b_win_rate:.1f}%)")
    
    # Decision logic
    print("\n" + "="*70)
    print("DECISION LOGIC")
    print("="*70)
    
    print(f"\nModel A drift on validation set: {drift_pct_model_a:+.2f}%")
    print(f"Model B win rate: {model_b_win_rate:.1f}%")
    
    if model_b_win_rate >= 75 and abs(drift_pct_model_a) < 10:
        recommendation = "USE MODEL B (100%)"
        reason = "Model B wins ≥75% of windows AND Model A drift < 10%"
    else:
        recommendation = "USE MODEL A (90%)"
        if model_b_win_rate < 75:
            reason = f"Model B win rate ({model_b_win_rate:.1f}%) < 75% threshold"
        else:
            reason = f"Model A drift ({drift_pct_model_a:+.1f}%) exceeds 10% threshold"
    
    print(f"\nRECOMMENDATION: {recommendation}")
    print(f"Reason: {reason}")
    
    print("="*70)
    
    return recommendation


def main():
    print("="*70)
    print("Volatility Forecasting - Dual Model Monitoring Service")
    print("="*70)
    
    # Load dual model baseline
    print("\nLoading dual model baseline...")
    try:
        dual_baseline = load_dual_model_baseline()
        print(f"Model A trained on: {dual_baseline['model_a']['training_date_start']} to "
              f"{dual_baseline['model_a']['training_date_end']}")
        print(f"Model B trained on: {dual_baseline['model_b']['training_date_start']} to "
              f"{dual_baseline['model_b']['training_date_end']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load validation baseline for drift detection
    print("\nLoading validation baseline for drift detection...")
    try:
        val_baseline = load_validation_baseline()
        baseline_rmse = val_baseline['validation_rmse']
        print(f"Model A validation RMSE baseline: {baseline_rmse:.6f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load both models
    print("\nLoading Model A (90%)...")
    try:
        model_a = load_trained_model("model_90pct")
        print(f"✓ Model A loaded: ensemble with {len(model_a['feature_cols'])} features")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print("\nLoading Model B (100%)...")
    try:
        model_b = load_trained_model("model_100pct")
        print(f"✓ Model B loaded: ensemble with {len(model_b['feature_cols'])} features")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load master dataset
    print("\nLoading master dataset...")
    master_df = load_master_dataset()
    
    # ===================================================================
    # PART 1: DRIFT DETECTION (Model A on validation set)
    # ===================================================================
    print("\n" + "="*70)
    print("PART 1: DRIFT DETECTION")
    print("="*70)
    print("\nUsing Model A (90%) to detect drift on validation set...")
    
    # Get validation date range from baseline
    val_date_end = pd.to_datetime(dual_baseline['model_a']['training_date_end'])
    
    # Create validation window (data after Model A's training cutoff)
    val_df = master_df[master_df['date'] > val_date_end].copy()
    
    if len(val_df) == 0:
        print("⚠ No validation data available (all data was used in training)")
        print("  This is expected if Model B used 100% of available data")
    else:
        print(f"\nValidation window: {len(val_df)} samples")
        print(f"  Date range: {val_df['date'].min().date()} to {val_df['date'].max().date()}")
        
        # Make predictions with Model A
        val_pred = make_predictions(val_df, model_a)
        val_actual = val_df['rv_5d'].values
        
        # Calculate metrics
        val_metrics = calculate_metrics(val_actual, val_pred)
        
        # Detect drift
        alert_level, alert_message, rmse_vs_baseline = detect_performance_drift(
            val_metrics['rmse'], baseline_rmse
        )
        
        # Print results
        print(f"\nModel A on validation set:")
        print(f"  Current RMSE:  {val_metrics['rmse']:.6f}")
        print(f"  Baseline RMSE: {baseline_rmse:.6f}")
        print(f"  Drift: {rmse_vs_baseline:+.2f}%")
        print(f"\n{alert_message}")
    
    # ===================================================================
    # PART 2: DUAL MODEL COMPARISON
    # ===================================================================
    print("\n" + "="*70)
    print("PART 2: DUAL MODEL COMPARISON")
    print("="*70)
    print("\n⚠️  SAFETY: Explicit date filtering to exclude training data")
    print(f"   Model A trained up to: {dual_baseline['model_a']['training_date_end']}")
    print(f"   Model B trained up to: {dual_baseline['model_b']['training_date_end']}")
    
    # Get training cutoff dates from baseline
    model_a_cutoff = pd.to_datetime(dual_baseline['model_a']['training_date_end'])
    model_b_cutoff = pd.to_datetime(dual_baseline['model_b']['training_date_end'])
    
    print(f"\n   Using latest cutoff: {max(model_a_cutoff, model_b_cutoff).date()}")
    print(f"   Only testing on data AFTER this date (exclusive)")
    
    # Get data after Model B's training cutoff (truly new data for both models)
    latest_cutoff = max(model_a_cutoff, model_b_cutoff)
    new_data_df = master_df[master_df['date'] > latest_cutoff].copy()
    
    print(f"\nData available after cutoff:")
    if len(new_data_df) > 0:
        print(f"  ✓ {len(new_data_df)} samples available")
        print(f"    Date range: {new_data_df['date'].min().date()} to {new_data_df['date'].max().date()}")
    else:
        print(f"  ✗ No new data available yet")
        print(f"    Latest training cutoff: {latest_cutoff.date()}")
        print(f"    Latest data in dataset: {master_df['date'].max().date()}")
        print(f"    Wait for new data after cutoff for fair comparison")
    
    comparisons = []
    
    # Compare on multiple windows (with explicit date filtering in compare function)
    if len(new_data_df) > 0:
        # Window 1: All new data
        comp = compare_models_on_window(
            new_data_df, model_a, model_b, 
            f"All new data ({len(new_data_df)} samples)",
            model_a_cutoff, model_b_cutoff
        )
        if comp:
            comparisons.append(comp)
        
        # Window 2: Last 30 samples of new data (if available)
        if len(new_data_df) >= 30:
            last_30_new = new_data_df.tail(30)
            comp = compare_models_on_window(
                last_30_new, model_a, model_b, 
                "Last 30 of new data",
                model_a_cutoff, model_b_cutoff
            )
            if comp:
                comparisons.append(comp)
        
        # Window 3: Last 10 samples of new data (if available)
        if len(new_data_df) >= 10:
            last_10_new = new_data_df.tail(10)
            comp = compare_models_on_window(
                last_10_new, model_a, model_b, 
                "Last 10 of new data",
                model_a_cutoff, model_b_cutoff
            )
            if comp:
                comparisons.append(comp)
        
        # Window 4: Last 1 sample of new data
        if len(new_data_df) >= 1:
            last_1_new = new_data_df.tail(1)
            comp = compare_models_on_window(
                last_1_new, model_a, model_b, 
                "Latest new sample",
                model_a_cutoff, model_b_cutoff
            )
            if comp:
                comparisons.append(comp)
    else:
        print("\n⚠️  CANNOT COMPARE MODELS YET")
        print("   Reason: No new data after Model B's training cutoff")
        print("   Action: Wait for new data or retrain models to free up test data")
    
    # Print comparison summary
    if comparisons:
        recommendation = print_model_comparison_summary(comparisons, dual_baseline, rmse_vs_baseline if len(val_df) > 0 else 0.0)
        
        # Save monitoring record to history
        test_window_info = {
            'n_samples': len(new_data_df),
            'date_start': str(new_data_df['date'].min().date()) if len(new_data_df) > 0 else None,
            'date_end': str(new_data_df['date'].max().date()) if len(new_data_df) > 0 else None,
        }
        save_monitoring_record(
            dual_baseline, 
            comparisons, 
            rmse_vs_baseline if len(val_df) > 0 else 0.0,
            recommendation,
            test_window_info
        )
        
        # Load and analyze historical trends
        history = load_monitoring_history()
        analyze_monitoring_trends(history)
    
    print("\n" + "="*70)
    print("Monitoring Complete")
    print("="*70)
    print("\nNext steps:")
    print("  - If drift detected: Retrain models with latest data")
    print("  - If Model B recommended: Update production to use model_100pct.pkl")
    print("  - If Model A recommended: Continue using model_90pct.pkl")
    print("="*70)


def prepare_monitoring_dataset(master_df, window_pct=0.1):
    """
    Prepare the monitoring dataset: latest 10% of master dataset.
    
    Args:
        master_df: Master dataset with all features and actuals
        window_pct: Percentage of data to use for monitoring (default 0.1 = 10%)
    
    Returns:
        DataFrame with features and actual RV for monitoring window
    """
    #ensure sorted by date
    master_df = master_df.sort_values('date').reset_index(drop=True)
    
    total_samples = len(master_df)
    window_start_idx = int(total_samples * (1 - window_pct))
    
    monitoring_df = master_df.iloc[window_start_idx:].copy()
    
    print(f"\nmonitoring window (latest {int(window_pct*100)}%):")
    print(f"  samples: {len(monitoring_df)}")
    print(f"  date range: {monitoring_df['date'].min().date()} to {monitoring_df['date'].max().date()}")
    print(f"  coverage: {len(monitoring_df)/total_samples*100:.1f}% of all data")
    
    return monitoring_df


if __name__ == "__main__":
    main()
