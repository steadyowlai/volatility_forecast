"""
Training Service

Trains stacking ensemble (XGBoost + LightGBM + Ridge meta-learner) to forecast 
5-day realized volatility (RV_5d) for SPY.

Uses configuration from final_model/model_config.json (exported from notebook research).
Logs to MLflow for production tracking.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn


#configuration
MASTER_DATASET = Path("data/master_dataset.parquet")
OUTPUT_DIR = Path("data/predictions")
MODELS_DIR = Path("models")
MODEL_CONFIG = Path("final_model/model_config.json")
BENCHMARK_CONFIG = Path("final_model/benchmark_results.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "volatility-forecasting"
MODEL_NAME = "volatility-forecaster-ensemble"


def load_master_dataset():
    """
    Load pre-computed master dataset from prepare_dataset service.
    
    This dataset contains all features and labels, ensuring consistency
    between training and monitoring.
    """
    if not MASTER_DATASET.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {MASTER_DATASET}. "
            f"Run prepare_dataset service first: docker-compose up prepare_dataset"
        )
    
    print(f"loading master dataset from {MASTER_DATASET}...")
    df = pd.read_parquet(MASTER_DATASET)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"loaded {len(df)} samples")
    print(f"date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"features: {len([c for c in df.columns if c not in ['date', 'rv_5d']])}")
    
    return df


def load_dataset():
    """
    Load pre-computed master dataset from prepare_dataset service.
    
    Returns:
        DataFrame with date, features, and rv_5d label
    """
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    df = load_master_dataset()
    
    print(f"\nFinal dataset: {len(df)} rows")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def load_model_config():
    """
    Load stacking ensemble configuration from final_model/model_config.json
    
    Returns:
        tuple: (config dict, benchmark dict)
    """
    if not MODEL_CONFIG.exists():
        raise FileNotFoundError(
            f"Model config not found at {MODEL_CONFIG}. "
            f"Run notebook 2_model_selection.ipynb section 9 to export config."
        )
    
    print(f"\nloading model config from {MODEL_CONFIG}")
    with open(MODEL_CONFIG) as f:
        config = json.load(f)
    
    print(f"model architecture: {config['model_architecture']}")
    print(f"base models: {[m['name'] for m in config['base_models']]}")
    print(f"meta learner: {config['meta_learner']['type']}")
    
    #load benchmarks if available
    benchmarks = None
    if BENCHMARK_CONFIG.exists():
        with open(BENCHMARK_CONFIG) as f:
            benchmarks = json.load(f)
        print(f"expected val rmse: {benchmarks['expected_performance']['val_rmse']:.6f}")
    
    return config, benchmarks


def walk_forward_split(df, train_size=0.9):
    """
    time-series aware train/validation split
    
    production uses 90/10 split (was 80/20 in research)
    train on first 90% to maximize training data
    validate on last 10% (most recent data for monitoring)
    
    never use future data to predict the past
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    split_idx = int(len(df) * train_size)
    
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    
    #extract date ranges for tracking
    train_start = train['date'].min()
    train_end = train['date'].max()
    val_start = val['date'].min()
    val_end = val['date'].max()
    
    print(f"\nTrain/Val split: {int(train_size*100)}/{int((1-train_size)*100)}")
    print(f"Train: {len(train)} rows from {train_start.date()} to {train_end.date()}")
    print(f"Val: {len(val)} rows from {val_start.date()} to {val_end.date()}")
    
    #return date ranges for mlflow logging
    date_info = {
        'train_start_date': str(train_start.date()),
        'train_end_date': str(train_end.date()),
        'val_start_date': str(val_start.date()),
        'val_end_date': str(val_end.date()),
        'n_train_samples': len(train),
        'n_val_samples': len(val),
        'train_size_pct': train_size * 100
    }
    
    return train, val, date_info


def train_ensemble_model(X_train, y_train, X_val, y_val, config, feature_cols, model_name="ensemble"):
    """
    Train stacking ensemble (XGBoost + LightGBM + Ridge).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (can be None for 100% training)
        y_val: Validation labels (can be None for 100% training)
        config: Model configuration
        feature_cols: List of feature column names
        model_name: Name for logging
    
    Returns:
        dict with models and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    if X_val is not None:
        print(f"Validation samples: {len(X_val)}")
    
    # Step 1: Train XGBoost
    print("\n" + "-"*60)
    print("Training base model 1: XGBoost")
    print("-"*60)
    
    xgb_config = config['base_models'][0]
    xgb_params = xgb_config['params'].copy()
    xgb_params['objective'] = 'reg:squarederror'
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    if X_val is not None:
        xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    else:
        xgb_model.fit(X_train, y_train, verbose=False)
    
    xgb_train_pred = xgb_model.predict(X_train)
    train_rmse_xgb = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
    print(f"Train RMSE: {train_rmse_xgb:.6f}")
    
    if X_val is not None:
        xgb_val_pred = xgb_model.predict(X_val)
        val_rmse_xgb = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
        print(f"Val RMSE: {val_rmse_xgb:.6f}")
    
    # Step 2: Train LightGBM
    print("\n" + "-"*60)
    print("Training base model 2: LightGBM")
    print("-"*60)
    
    lgbm_config = config['base_models'][1]
    lgbm_params = lgbm_config['params'].copy()
    
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    if X_val is not None:
        lgbm_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=-1)
    else:
        lgbm_model.fit(X_train, y_train, verbose=-1)
    
    lgbm_train_pred = lgbm_model.predict(X_train)
    train_rmse_lgbm = np.sqrt(mean_squared_error(y_train, lgbm_train_pred))
    print(f"Train RMSE: {train_rmse_lgbm:.6f}")
    
    if X_val is not None:
        lgbm_val_pred = lgbm_model.predict(X_val)
        val_rmse_lgbm = np.sqrt(mean_squared_error(y_val, lgbm_val_pred))
        print(f"Val RMSE: {val_rmse_lgbm:.6f}")
    
    # Step 3: Train meta-learner (Ridge)
    print("\n" + "-"*60)
    print("Training meta-learner: Ridge")
    print("-"*60)
    
    meta_config = config['meta_learner']
    meta_params = meta_config['params'].copy()
    print(f"Alpha: {meta_params['alpha']}")
    
    # Stack base predictions
    X_meta_train = np.column_stack([xgb_train_pred, lgbm_train_pred])
    
    meta_model = Ridge(**meta_params)
    meta_model.fit(X_meta_train, y_train)
    
    # Final ensemble predictions
    ensemble_train_pred = meta_model.predict(X_meta_train)
    
    train_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_train, ensemble_train_pred)),
        'mae': mean_absolute_error(y_train, ensemble_train_pred),
        'r2': r2_score(y_train, ensemble_train_pred)
    }
    
    print(f"\n{model_name} Training Performance:")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  MAE:  {train_metrics['mae']:.6f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    
    val_metrics = None
    ensemble_val_pred = None
    
    if X_val is not None:
        X_meta_val = np.column_stack([xgb_val_pred, lgbm_val_pred])
        ensemble_val_pred = meta_model.predict(X_meta_val)
        
        val_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, ensemble_val_pred)),
            'mae': mean_absolute_error(y_val, ensemble_val_pred),
            'r2': r2_score(y_val, ensemble_val_pred)
        }
        
        print(f"\n{model_name} Validation Performance:")
        print(f"  RMSE: {val_metrics['rmse']:.6f}")
        print(f"  MAE:  {val_metrics['mae']:.6f}")
        print(f"  R²:   {val_metrics['r2']:.4f}")
        
        # Improvement analysis
        xgb_improvement = ((val_rmse_xgb - val_metrics['rmse']) / val_rmse_xgb) * 100
        lgbm_improvement = ((val_rmse_lgbm - val_metrics['rmse']) / val_rmse_lgbm) * 100
        
        print(f"\nImprovement over base models:")
        print(f"  vs XGBoost:  {xgb_improvement:+.2f}%")
        print(f"  vs LightGBM: {lgbm_improvement:+.2f}%")
    
    return {
        'xgb_model': xgb_model,
        'lgbm_model': lgbm_model,
        'meta_model': meta_model,
        'feature_cols': feature_cols,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_pred': ensemble_train_pred,
        'val_pred': ensemble_val_pred
    }
    """
    Train stacking ensemble with XGBoost + LightGBM + Ridge meta-learner
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        config: Model configuration dict from load_model_config()
    
    Returns:
        tuple: (xgb_model, lgbm_model, meta_model, feature_cols, metrics)
    """
    print("\n" + "="*60)
    print("Training Stacking Ensemble")
def save_predictions(train_df, val_df, train_preds, val_preds):
    """Save predictions with dates for later analysis"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    #train
    train_results = pd.DataFrame({
        'date': train_df['date'],
        'actual': train_df['rv_5d'],
        'predicted': train_preds,
        'split': 'train'
    })
    
    #validation
    val_results = pd.DataFrame({
        'date': val_df['date'],
        'actual': val_df['rv_5d'],
        'predicted': val_preds,
        'split': 'val'
    })
    
    #combine and save
    all_results = pd.concat([train_results, val_results], ignore_index=True)
    outpath = OUTPUT_DIR / "predictions.parquet"
    all_results.to_parquet(outpath, index=False)
    
    print(f"\nsaved predictions to {outpath}")
    
    return outpath


def log_to_mlflow(xgb_model, lgbm_model, meta_model, feature_cols, train_metrics, val_metrics, config, benchmarks=None, date_info=None):
    """
    log stacking ensemble to mlflow with date tracking
    
    tracks training dates for rolling window monitoring
    """
    print("\n" + "="*60)
    print("Logging to MLflow")
    print("="*60)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"stacking-ensemble-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        #log model architecture
        mlflow.log_param('model_architecture', config['model_architecture'])
        mlflow.log_param('n_base_models', len(config['base_models']))
        mlflow.log_param('meta_learner', config['meta_learner']['type'])
        
        #log date ranges for tracking rolling window
        if date_info:
            mlflow.log_params({
                'train_start_date': date_info['train_start_date'],
                'train_end_date': date_info['train_end_date'],
                'val_start_date': date_info['val_start_date'],
                'val_end_date': date_info['val_end_date'],
                'n_train_samples': date_info['n_train_samples'],
                'n_val_samples': date_info['n_val_samples'],
                'train_size_pct': date_info['train_size_pct']
            })
            print(f"\nlogging date ranges:")
            print(f"train: {date_info['train_start_date']} to {date_info['train_end_date']}")
            print(f"val: {date_info['val_start_date']} to {date_info['val_end_date']}")
        
        #log timestamp
        mlflow.log_param('training_timestamp', datetime.now().isoformat())
        
        #log base model hyperparameters
        for i, base_model in enumerate(config['base_models']):
            for param_name, param_value in base_model['params'].items():
                if param_name not in ['random_state', 'n_jobs', 'verbose']:
                    mlflow.log_param(f"{base_model['name']}_{param_name}", param_value)
        
        #log meta-learner hyperparameters
        for param_name, param_value in config['meta_learner']['params'].items():
            if param_name not in ['random_state']:
                mlflow.log_param(f"meta_{param_name}", param_value)
        
        #log train metrics
        mlflow.log_metrics({
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2']
        })
        
        #log validation metrics
        mlflow.log_metrics({
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2']
        })
        
        #log feature info
        mlflow.log_param('n_features', len(feature_cols))
        mlflow.log_param('features', ','.join(feature_cols))
        
        #log benchmark comparison if available
        if benchmarks:
            expected_rmse = benchmarks['expected_performance']['val_rmse']
            actual_rmse = val_metrics['rmse']
            rmse_diff = actual_rmse - expected_rmse
            rmse_diff_pct = (rmse_diff / expected_rmse) * 100
            
            mlflow.log_metrics({
                'expected_val_rmse': expected_rmse,
                'rmse_diff': rmse_diff,
                'rmse_diff_pct': rmse_diff_pct
            })
            
            print(f"\nbenchmark comparison:")
            print(f"expected val rmse: {expected_rmse:.6f}")
            print(f"actual val rmse: {actual_rmse:.6f}")
            print(f"difference: {rmse_diff:+.6f} ({rmse_diff_pct:+.2f}%)")
            
            #check for performance degradation
            if rmse_diff_pct > 10:
                print(f"\nwarning: model performance degraded by {rmse_diff_pct:.2f}%")
                print("consider investigating data quality or retuning hyperparameters")
                mlflow.set_tag("performance_alert", "degraded")
            elif rmse_diff_pct > 5:
                print(f"\nnote: model performance slightly worse by {rmse_diff_pct:.2f}%")
                mlflow.set_tag("performance_alert", "monitor")
            else:
                print("\nperformance within expected range")
                mlflow.set_tag("performance_alert", "ok")
        
        #log models as sklearn pipeline-like dict
        #mlflow doesn't have native stacking support, so we log as sklearn artifact
        ensemble_artifact = {
            'xgb_model': xgb_model,
            'lgbm_model': lgbm_model,
            'meta_model': meta_model,
            'feature_cols': feature_cols
        }
        
        mlflow.sklearn.log_model(
            ensemble_artifact,
            "ensemble",
            registered_model_name=MODEL_NAME
        )
        
        #save model to local filesystem for prediction service
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import pickle
        
        # Save as Model A (90% model)
        model_path_90 = MODELS_DIR / "model_90pct.pkl"
        with open(model_path_90, 'wb') as f:
            pickle.dump(ensemble_artifact, f)
        print(f"saved Model A (90%) to {model_path_90}")
        
        # Also save as latest_ensemble for backward compatibility
        model_path = MODELS_DIR / "latest_ensemble.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble_artifact, f)
        print(f"saved model to {model_path}")
        
        #save validation baseline for monitoring service
        import json
        baseline_path = MODELS_DIR / "validation_baseline.json"
        baseline_data = {
            'validation_rmse': val_metrics['rmse'],
            'validation_mae': val_metrics['mae'],
            'validation_r2': val_metrics['r2'],
            'timestamp': datetime.now().isoformat(),
            'train_val_split': config.get('training', {}).get('train_val_split', 0.9),
            'date_range': {
                'train_start': str(date_info['train_start_date']),
                'train_end': str(date_info['train_end_date']),
                'val_start': str(date_info['val_start_date']),
                'val_end': str(date_info['val_end_date'])
            }
        }
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        print(f"saved validation baseline to {baseline_path}")
        
        #log predictions
        pred_path = OUTPUT_DIR / "predictions.parquet"
        if pred_path.exists():
            mlflow.log_artifact(pred_path)
        
        #log config files
        mlflow.log_artifact(MODEL_CONFIG)
        if BENCHMARK_CONFIG.exists():
            mlflow.log_artifact(BENCHMARK_CONFIG)
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nmlflow run id: {run_id}")
        print(f"model registered as: {MODEL_NAME}")
    
    print("\nmlflow logging complete")


def main():
    print("\n" + "="*60)
    print("Volatility Forecasting - Training Service")
    print("="*60)
    
    #load model configuration
    config, benchmarks = load_model_config()
    
    #load dataset (from prepare_dataset service)
    df = load_dataset()
    
    #time-series split with date tracking
    train_df, val_df, date_info = walk_forward_split(df, train_size=0.9)
    
    #============================================================
    # MODEL A: Train on 90% data (with validation baseline)
    #============================================================
    print("\n" + "="*60)
    print("MODEL A: Training on 90% data")
    print("="*60)
    
    #train stacking ensemble
    xgb_model, lgbm_model, meta_model, feature_cols, train_metrics, val_metrics, train_preds, val_preds = train_stacking_ensemble(
        train_df, val_df, config
    )
    
    #save predictions
    save_predictions(train_df, val_df, train_preds, val_preds)
    
    #log to mlflow with date tracking
    log_to_mlflow(
        xgb_model,
        lgbm_model,
        meta_model,
        feature_cols,
        train_metrics,
        val_metrics,
        config,
        benchmarks,
        date_info
    )
    
    print(f"\nModel A (90%) validation performance:")
    print(f"  rmse: {val_metrics['rmse']:.6f}")
    print(f"  mae:  {val_metrics['mae']:.6f}")
    print(f"  r2:   {val_metrics['r2']:.4f}")
    
    #============================================================
    # MODEL B: Train on 100% data (all available)
    #============================================================
    print("\n" + "="*60)
    print("MODEL B: Training on 100% data")
    print("="*60)
    
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    
    #prepare full dataset
    X_full = df[[c for c in df.columns if c not in ['date', 'rv_5d']]].values
    y_full = df['rv_5d'].values
    
    #get same hyperparameters from config
    xgb_config = config['base_models'][0]  # XGBoost
    lgbm_config = config['base_models'][1]  # LightGBM
    meta_config = config['meta_learner']
    
    xgb_params = xgb_config['params'].copy()
    xgb_params['objective'] = 'reg:squarederror'
    
    lgbm_params = lgbm_config['params'].copy()
    ridge_params = meta_config['params'].copy()
    
    print(f"\nTraining on {len(df)} samples (100% of data)")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    #train base models
    print("\nTraining XGBoost (Model B)...")
    xgb_model_b = xgb.XGBRegressor(**xgb_params)
    xgb_model_b.fit(X_full, y_full, verbose=False)
    
    print("Training LightGBM (Model B)...")
    lgbm_model_b = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model_b.fit(X_full, y_full, verbose=-1)
    
    #train meta-learner
    print("Training meta-learner...")
    xgb_pred_full = xgb_model_b.predict(X_full)
    lgbm_pred_full = lgbm_model_b.predict(X_full)
    X_meta_full = np.column_stack([xgb_pred_full, lgbm_pred_full])
    
    meta_model_b = Ridge(**ridge_params)
    meta_model_b.fit(X_meta_full, y_full)
    
    #save Model B
    import pickle
    ensemble_artifact_b = {
        'xgb_model': xgb_model_b,
        'lgbm_model': lgbm_model_b,
        'meta_model': meta_model_b,
        'feature_cols': feature_cols
    }
    
    model_path_100 = MODELS_DIR / "model_100pct.pkl"
    with open(model_path_100, 'wb') as f:
        pickle.dump(ensemble_artifact_b, f)
    print(f"\n✓ Model B (100%) saved to {model_path_100}")
    
    #evaluate Model B on validation set (for comparison)
    X_val = val_df[[c for c in val_df.columns if c not in ['date', 'rv_5d']]].values
    y_val = val_df['rv_5d'].values
    
    xgb_pred_val_b = xgb_model_b.predict(X_val)
    lgbm_pred_val_b = lgbm_model_b.predict(X_val)
    X_meta_val_b = np.column_stack([xgb_pred_val_b, lgbm_pred_val_b])
    y_pred_val_b = meta_model_b.predict(X_meta_val_b)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse_val_b = np.sqrt(mean_squared_error(y_val, y_pred_val_b))
    mae_val_b = mean_absolute_error(y_val, y_pred_val_b)
    r2_val_b = r2_score(y_val, y_pred_val_b)
    
    print(f"\nModel B (100%) performance on validation set:")
    print(f"  rmse: {rmse_val_b:.6f} (in-sample)")
    print(f"  mae:  {mae_val_b:.6f}")
    print(f"  r2:   {r2_val_b:.4f}")
    
    #save dual model baseline
    import json
    dual_baseline = {
        'timestamp': datetime.now().isoformat(),
        'model_a': {
            'training_samples': len(train_df),
            'training_date_start': str(date_info['train_start_date']),
            'training_date_end': str(date_info['train_end_date']),
            'validation_rmse': float(val_metrics['rmse']),
            'validation_mae': float(val_metrics['mae']),
            'validation_r2': float(val_metrics['r2']),
            'validation_samples': len(val_df)
        },
        'model_b': {
            'training_samples': len(df),
            'training_date_start': str(df['date'].min().date()),
            'training_date_end': str(df['date'].max().date()),
            'validation_rmse': float(rmse_val_b),
            'validation_mae': float(mae_val_b),
            'validation_r2': float(r2_val_b),
            'note': 'Model B metrics on validation set are in-sample'
        }
    }
    
    dual_baseline_path = MODELS_DIR / "dual_model_baseline.json"
    with open(dual_baseline_path, 'w') as f:
        json.dump(dual_baseline, f, indent=2)
    print(f"✓ Dual baseline saved to {dual_baseline_path}")
    
    print("\n" + "="*60)
    print("Training Complete - Dual Model Strategy")
    print("="*60)
    print(f"\nModel A (90%): RMSE = {val_metrics['rmse']:.6f} (out-of-sample)")
    print(f"Model B (100%): RMSE = {rmse_val_b:.6f} (in-sample)")
    print(f"\nBoth models saved. Monitor service will compare them.")
    print("="*60)
    print(f"  mae:  {val_metrics['mae']:.6f}")
    print(f"  r2:   {val_metrics['r2']:.4f}")
    print(f"\nartifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
