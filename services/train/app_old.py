"""
train service - dual model strategy

trains two models:
- model A (90%): conservative, has true validation for drift detection
- model B (all-30): aggressive, uses max data but holds out last 30 for testing

both are stacking ensembles: xgboost + lightgbm + ridge metalearner

loads master dataset from prepare_dataset service
config from final_model/model_config.json (exported from notebooks)
logs model A to mlflow for tracking
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local experiment tracker
from experiment_tracker import experiment_run

# Storage abstraction layer
from storage import Storage

# Initialize storage
storage = Storage()

def get_training_date():
    """Get current date in YYYY-MM-DD format for date-based storage"""
    return datetime.now().strftime("%Y-%m-%d")


def append_validation_history(model_name, metrics, feature_importance=None, date_ranges=None):
    """
    Append validation metrics and feature importance to history files
    
    Args:
        model_name: e.g. 'xgboost_90pct', 'lightgbm_100pct', etc.
        metrics: dict with keys like train_rmse, val_rmse, train_mae, val_mae, etc.
        feature_importance: dict mapping feature names to importance scores (optional)
        date_ranges: dict with training_start_date, training_end_date, val_start_date, val_end_date (optional)
    """
    timestamp = datetime.now().isoformat()
    training_date = get_training_date()
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    metrics = convert_to_python_types(metrics)
    if feature_importance is not None:
        feature_importance = convert_to_python_types(feature_importance)
    
    # Append validation metrics
    validation_file = f"{TRAIN_DATA_DIR}/validation/{model_name}_validation.json"
    
    validation_entry = {
        'timestamp': timestamp,
        'training_date': training_date,
        **metrics
    }
    
    # Add date ranges if provided
    if date_ranges:
        validation_entry.update({
            'training_start_date': date_ranges.get('training_start_date'),
            'training_end_date': date_ranges.get('training_end_date'),
            'val_start_date': date_ranges.get('val_start_date'),
            'val_end_date': date_ranges.get('val_end_date')
        })
    
    # Read existing history or create new
    if os.path.exists(validation_file):
        with open(validation_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(validation_entry)
    
    # Write back
    os.makedirs(f"{TRAIN_DATA_DIR}/validation", exist_ok=True)
    with open(validation_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"appended to {validation_file}")
    
    # Append feature importance if provided
    if feature_importance is not None:
        importance_file = f"{TRAIN_DATA_DIR}/validation/{model_name}_feature_importance.json"
        
        importance_entry = {
            'timestamp': timestamp,
            'training_date': training_date,
            'feature_importance': feature_importance
        }
        
        # Read existing history or create new
        if os.path.exists(importance_file):
            with open(importance_file, 'r') as f:
                imp_history = json.load(f)
        else:
            imp_history = []
        
        imp_history.append(importance_entry)
        
        # Write back
        with open(importance_file, 'w') as f:
            json.dump(imp_history, f, indent=2)
        
        print(f"appended to {importance_file}")


#config
MASTER_DATASET = "data/master_dataset.parquet"
MODELS_DIR = "data/models"              # Models only
TRAIN_DATA_DIR = "data/train"           # Training metadata and logs
EXPERIMENTS_DIR = "data/train"          # Experiment logs
MODEL_CONFIG = "final_model/model_config.json"
BENCHMARK_CONFIG = "final_model/benchmark_results.json"
EXPERIMENT_NAME = "training"


def load_master_dataset():
    """load master dataset from prepare_dataset service"""
    if not storage.exists(MASTER_DATASET):
        raise FileNotFoundError(
            f"master dataset not found at {MASTER_DATASET}. "
            f"run prepare_dataset service first: docker-compose up prepare_dataset"
        )
    
    print(f"loading master dataset from {MASTER_DATASET}")
    df = storage.read_parquet(MASTER_DATASET)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"loaded {len(df)} samples")
    print(f"date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"features: {len([c for c in df.columns if c not in ['date', 'rv_5d']])}")
    
    return df


def load_model_config():
    """load model config from final_model/model_config.json"""
    if not storage.exists(MODEL_CONFIG):
        raise FileNotFoundError(
            f"model config not found at {MODEL_CONFIG}. "
            f"run notebook 2_model_selection.ipynb section 9 to export config"
        )
    
    print(f"\nloading model config from {MODEL_CONFIG}")
    config = storage.read_json(MODEL_CONFIG)
    
    print(f"architecture: {config['model_architecture']}")
    print(f"base models: {[m['name'] for m in config['base_models']]}")
    print(f"meta learner: {config['meta_learner']['type']}")
    
    #load benchmarks if available
    benchmarks = None
    if storage.exists(BENCHMARK_CONFIG):
        benchmarks = storage.read_json(BENCHMARK_CONFIG)
        print(f"expected val rmse: {benchmarks['expected_performance']['val_rmse']:.6f}")
    
    return config, benchmarks


def split_dataset(df, train_size=0.9):
    """time-series split for train/val"""
    df = df.sort_values('date').reset_index(drop=True)
    
    split_idx = int(len(df) * train_size)
    
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    
    split_info = {
        'train_samples': len(train),
        'val_samples': len(val),
        'train_start_date': str(train['date'].min().date()),
        'train_end_date': str(train['date'].max().date()),
        'val_start_date': str(val['date'].min().date()),
        'val_end_date': str(val['date'].max().date()),
        'train_size_pct': train_size * 100
    }
    
    print(f"\ntrain/val split {int(train_size*100)}/{int((1-train_size)*100)}")
    print(f"train: {len(train)} samples from {split_info['train_start_date']} to {split_info['train_end_date']}")
    print(f"val: {len(val)} samples from {split_info['val_start_date']} to {split_info['val_end_date']}")
    
    return train, val, split_info


def train_ensemble(X_train, y_train, X_val, y_val, config, model_name="Ensemble"):
    """train stacking ensemble: xgboost + lightgbm + ridge"""
    print(f"\n{'='*60}")
    print(f"training {model_name}")
    print(f"{'='*60}")
    print(f"training samples: {len(X_train)}")
    if X_val is not None:
        print(f"validation samples: {len(X_val)}")
    
    #step 1: train xgboost
    print("\n" + "-"*60)
    print("training xgboost")
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
    print(f"train rmse: {train_rmse_xgb:.6f}")
    
    if X_val is not None:
        xgb_val_pred = xgb_model.predict(X_val)
        val_rmse_xgb = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
        print(f"val rmse: {val_rmse_xgb:.6f}")
    
    #step 2: train lightgbm
    print("\n" + "-"*60)
    print("training lightgbm")
    print("-"*60)
    
    lgbm_config = config['base_models'][1]
    lgbm_params = lgbm_config['params'].copy()
    
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    if X_val is not None:
        lgbm_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], 
                      callbacks=[lgb.log_evaluation(0)])
    else:
        lgbm_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
    
    lgbm_train_pred = lgbm_model.predict(X_train)
    train_rmse_lgbm = np.sqrt(mean_squared_error(y_train, lgbm_train_pred))
    print(f"train rmse: {train_rmse_lgbm:.6f}")
    
    if X_val is not None:
        lgbm_val_pred = lgbm_model.predict(X_val)
        val_rmse_lgbm = np.sqrt(mean_squared_error(y_val, lgbm_val_pred))
        print(f"Val RMSE:   {val_rmse_lgbm:.6f}")
    
    # === Step 3: Train Meta-Learner (Ridge) ===
    print("\n" + "-"*60)
    print("Training Meta-Learner (Ridge)")
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
        
        print(f"\n{model_name} validation performance")
        print(f"rmse {val_metrics['rmse']:.6f} mae {val_metrics['mae']:.6f} r2 {val_metrics['r2']:.4f}")
        
        #improvement vs base models
        xgb_improvement = ((val_rmse_xgb - val_metrics['rmse']) / val_rmse_xgb) * 100
        lgbm_improvement = ((val_rmse_lgbm - val_metrics['rmse']) / val_rmse_lgbm) * 100
        
        print(f"\nimprovement over base models")
        print(f"vs xgboost {xgb_improvement:+.2f}% vs lightgbm {lgbm_improvement:+.2f}%")
    
    return {
        'xgb_model': xgb_model,
        'lgbm_model': lgbm_model,
        'meta_model': meta_model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_pred': ensemble_train_pred,
        'val_pred': ensemble_val_pred,
        'xgb_train_rmse': train_rmse_xgb,
        'xgb_val_rmse': val_rmse_xgb if X_val is not None else None,
        'lgbm_train_rmse': train_rmse_lgbm,
        'lgbm_val_rmse': val_rmse_lgbm if X_val is not None else None
    }


def save_models(models, feature_cols, split_name):
    """
    Save all models separately (xgboost, lightgbm, ensemble) in date-based folders
    split_name: '90pct' or '100pct'
    """
    training_date = get_training_date()
    date_folder = f"{MODELS_DIR}/date={training_date}"
    os.makedirs(date_folder, exist_ok=True)
    
    # Save individual base models
    xgb_artifact = {
        'model': models['xgb_model'],
        'feature_cols': feature_cols
    }
    xgb_path = f"{date_folder}/xgboost_{split_name}.pkl"
    storage.write_pickle(xgb_artifact, xgb_path)
    print(f"saved {xgb_path}")
    
    lgbm_artifact = {
        'model': models['lgbm_model'],
        'feature_cols': feature_cols
    }
    lgbm_path = f"{date_folder}/lightgbm_{split_name}.pkl"
    storage.write_pickle(lgbm_artifact, lgbm_path)
    print(f"saved {lgbm_path}")
    
    # Save ensemble (all 3 models together)
    ensemble_artifact = {
        'xgb_model': models['xgb_model'],
        'lgbm_model': models['lgbm_model'],
        'meta_model': models['meta_model'],
        'feature_cols': feature_cols
    }
    ensemble_path = f"{date_folder}/ensemble_{split_name}.pkl"
    storage.write_pickle(ensemble_artifact, ensemble_path)
    print(f"saved {ensemble_path}")
    
    # Also save to "latest" for easy access
    latest_folder = f"{MODELS_DIR}/latest"
    os.makedirs(latest_folder, exist_ok=True)
    
    storage.write_pickle(xgb_artifact, f"{latest_folder}/xgboost_{split_name}.pkl")
    storage.write_pickle(lgbm_artifact, f"{latest_folder}/lightgbm_{split_name}.pkl")
    storage.write_pickle(ensemble_artifact, f"{latest_folder}/ensemble_{split_name}.pkl")
    
    print(f"also saved to {latest_folder}/ for easy access")
    
    return ensemble_path



def log_training_experiment(models, feature_cols, train_metrics, val_metrics, config, benchmarks, split_info):
    """Log training experiment using simple JSON tracker"""
    print("\n" + "="*60)
    print("logging training experiment")
    print("="*60)
    
    with experiment_run(EXPERIMENT_NAME, run_name="dual-model-training") as tracker:
        # Log model architecture params
        tracker.log_params({
            'model_architecture': config.get('model_architecture', 'stacking_ensemble'),
            'n_base_models': len(config.get('base_models', [])),
            'n_features': len(feature_cols),
            'n_train_samples': split_info['train_samples'],
            'n_val_samples': split_info['val_samples']
        })
        
        # Log date ranges
        tracker.log_params({
            'train_start_date': split_info['train_start_date'],
            'train_end_date': split_info['train_end_date'],
            'val_start_date': split_info['val_start_date'],
            'val_end_date': split_info['val_end_date']
        })
        
        # Log hyperparameters
        for base_model in config.get('base_models', []):
            if isinstance(base_model, dict):
                model_name = base_model.get('name', 'unknown')
                model_params = base_model.get('params', {})
                for param_name, param_value in model_params.items():
                    tracker.log_param(f'{model_name}_{param_name}', param_value)
        
        # Log meta-learner params
        if 'meta_learner' in config:
            meta_params = config['meta_learner'].get('params', {})
            for param_name, param_value in meta_params.items():
                tracker.log_param(f'meta_{param_name}', param_value)
        
        # Log training metrics
        tracker.log_metrics({
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2']
        })
        
        # Benchmark comparison
        if benchmarks:
            expected_rmse = benchmarks.get('expected_performance', {}).get('val_rmse')
            if expected_rmse:
                actual_rmse = val_metrics['rmse']
                diff = actual_rmse - expected_rmse
                diff_pct = (diff / expected_rmse) * 100
                
                tracker.log_metrics({
                    'expected_val_rmse': expected_rmse,
                    'rmse_diff': diff,
                    'rmse_diff_pct': diff_pct
                })
                
                print(f"\nbenchmark comparison")
                print(f"expected {expected_rmse:.6f} actual {actual_rmse:.6f} diff {diff:+.6f} ({diff_pct:+.2f}%)")
        
        # Log artifact paths
        tracker.log_artifact(f"{MODELS_DIR}/model_90pct.pkl", "Ensemble 90% split")
        tracker.log_artifact(MODEL_CONFIG, "Model configuration")
        if storage.exists(BENCHMARK_CONFIG):
            tracker.log_artifact(BENCHMARK_CONFIG, "Benchmark results")
        
        print(f"\n✅ Experiment logged to {EXPERIMENTS_DIR}")
        print("✅ Training complete")


def main():
    """main training pipeline"""
    print("\n" + "="*60)
    print("volatility forecasting dual model training")
    print("="*60)
    
    #load config and data
    config, benchmarks = load_model_config()
    df = load_master_dataset()
    
    #prep features
    feature_cols = [c for c in df.columns if c not in ['date', 'rv_5d']]
    print(f"\nfeatures {len(feature_cols)}: {feature_cols[:5]}...")
    
    #split dataset
    train_df, val_df, split_info = split_dataset(df, train_size=0.9)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['rv_5d'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['rv_5d'].values
    X_full = df[feature_cols].values
    y_full = df['rv_5d'].values
    
    #model A: train on 90% with validation
    print("\n" + "="*70)
    print("model A: training on 90% with validation baseline")
    print("="*70)
    
    model_90pct = train_ensemble(X_train, y_train, X_val, y_val, config, "model A 90pct")
    
    #save model A
    save_models(model_90pct, feature_cols, "90pct")

    # Prepare date ranges for validation history
    date_ranges_90pct = {
        "training_start_date": str(train_df["date"].min()),
        "training_end_date": str(train_df["date"].max()),
        "val_start_date": str(val_df["date"].min()),
        "val_end_date": str(val_df["date"].max())
    }

    # Log validation history for each model
    xgb_importance = dict(zip(feature_cols, model_90pct["xgb_model"].feature_importances_))
    lgbm_importance = dict(zip(feature_cols, model_90pct["lgbm_model"].feature_importances_))

    append_validation_history(
        "xgboost_90pct",
        {
            "train_rmse": model_90pct["xgb_train_rmse"],
            "val_rmse": model_90pct["xgb_val_rmse"],
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        },
        xgb_importance,
        date_ranges_90pct
    )

    append_validation_history(
        "lightgbm_90pct",
        {
            "train_rmse": model_90pct["lgbm_train_rmse"],
            "val_rmse": model_90pct["lgbm_val_rmse"],
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        },
        lgbm_importance,
        date_ranges_90pct
    )

    append_validation_history(
        "ensemble_90pct",
        {
            **model_90pct["train_metrics"],
            **{f"val_{k}": v for k, v in model_90pct["val_metrics"].items()},
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        },
        None,
        date_ranges_90pct
    )
    
    #save predictions
    #log training experiment
    log_training_experiment(
        model_90pct, feature_cols,
        model_90pct['train_metrics'], model_90pct['val_metrics'],
        config, benchmarks, split_info
    )
    
    #model B: train on all data except last 30
    print("\n" + "="*70)
    print("model B: training on all data except last 30 samples")
    print("="*70)
    
    #hold out last 30 for immediate monitoring
    HOLDOUT_SAMPLES = 30
    
    if len(df) > HOLDOUT_SAMPLES:
        #split: everything except last 30 for training, last 30 for testing
        train_b_df = df.iloc[:-HOLDOUT_SAMPLES].copy()
        test_b_df = df.iloc[-HOLDOUT_SAMPLES:].copy()
        
        X_train_b = train_b_df[feature_cols].values
        y_train_b = train_b_df['rv_5d'].values
        X_test_b = test_b_df[feature_cols].values
        y_test_b = test_b_df['rv_5d'].values
        
        print(f"\nmodel B split")
        print(f"training {len(train_b_df)} samples from {train_b_df['date'].min().date()} to {train_b_df['date'].max().date()}")
        print(f"testing {len(test_b_df)} samples from {test_b_df['date'].min().date()} to {test_b_df['date'].max().date()}")
        
        #train model B with test set for validation metrics
        model_100pct = train_ensemble(X_train_b, y_train_b, X_test_b, y_test_b, config, "model B all-30")
        
        #save model B
        save_models(model_100pct, feature_cols, "100pct")

        # Prepare date ranges for validation history
        date_ranges_100pct = {
            "training_start_date": str(train_b_df["date"].min()),
            "training_end_date": str(train_b_df["date"].max()),
            "val_start_date": str(test_b_df["date"].min()),
            "val_end_date": str(test_b_df["date"].max())
        }

        # Log validation history for each model (trained on full data)
        xgb_importance = dict(zip(feature_cols, model_100pct["xgb_model"].feature_importances_))
        lgbm_importance = dict(zip(feature_cols, model_100pct["lgbm_model"].feature_importances_))

        append_validation_history(
            "xgboost_100pct",
            {
                "train_rmse": model_100pct["xgb_train_rmse"],
                "val_rmse": model_100pct["xgb_val_rmse"],
                "train_samples": len(X_train_b),
                "val_samples": len(X_test_b)
            },
            xgb_importance,
            date_ranges_100pct
        )

        append_validation_history(
            "lightgbm_100pct",
            {
                "train_rmse": model_100pct["lgbm_train_rmse"],
                "val_rmse": model_100pct["lgbm_val_rmse"],
                "train_samples": len(X_train_b),
                "val_samples": len(X_test_b)
            },
            lgbm_importance,
            date_ranges_100pct
        )

        append_validation_history(
            "ensemble_100pct",
            {
                **model_100pct["train_metrics"],
                **{f"val_{k}": v for k, v in model_100pct["val_metrics"].items()},
                "train_samples": len(X_train_b),
                "val_samples": len(X_test_b)
            },
            None,
            date_ranges_100pct
        )
        
        #model B eval metrics (true out-of-sample on last 30)
        print(f"\nmodel B all-30 on last 30 samples (true out-of-sample)")
        print(f"rmse {model_100pct['val_metrics']['rmse']:.6f} mae {model_100pct['val_metrics']['mae']:.6f} r2 {model_100pct['val_metrics']['r2']:.4f}")
        print(f"note: these are TRUE out-of-sample metrics, model never saw this data")
    else:
        print(f"\nwarning: dataset too small {len(df)} samples to hold out {HOLDOUT_SAMPLES}")
        print(f"training model B on 100pct data instead")
        
        model_100pct = train_ensemble(X_full, y_full, None, None, config, "model B 100pct")
        save_models(model_100pct, feature_cols, "100pct")

        # Prepare date ranges for validation history (using full dataset)
        full_df = pd.concat([train_df, val_df])
        date_ranges_100pct = {
            "training_start_date": str(full_df["date"].min()),
            "training_end_date": str(full_df["date"].max()),
            "val_start_date": None,  # No validation set in this case
            "val_end_date": None
        }

        # Log validation history for each model (trained on full data)
        xgb_importance = dict(zip(feature_cols, model_100pct["xgb_model"].feature_importances_))
        lgbm_importance = dict(zip(feature_cols, model_100pct["lgbm_model"].feature_importances_))

        append_validation_history(
            "xgboost_100pct",
            {
                "train_rmse": model_100pct["xgb_train_rmse"],
                "val_rmse": None,  # No validation set
                "train_samples": len(X_full),
                "val_samples": 0
            },
            xgb_importance,
            date_ranges_100pct
        )

        append_validation_history(
            "lightgbm_100pct",
            {
                "train_rmse": model_100pct["lgbm_train_rmse"],
                "val_rmse": None,  # No validation set
                "train_samples": len(X_full),
                "val_samples": 0
            },
            lgbm_importance,
            date_ranges_100pct
        )

        append_validation_history(
            "ensemble_100pct",
            {
                **model_100pct["train_metrics"],
                "val_rmse": None,
                "val_mae": None,
                "val_r2": None,
                "train_samples": len(X_full),
                "val_samples": 0
            },
            None,
            date_ranges_100pct
        )
        
        #eval on validation set (insample for model B)
        X_meta_val_b = np.column_stack([
            model_100pct['xgb_model'].predict(X_val),
            model_100pct['lgbm_model'].predict(X_val)
        ])
        y_pred_val_b = model_100pct['meta_model'].predict(X_meta_val_b)
        
        val_rmse_b = np.sqrt(mean_squared_error(y_val, y_pred_val_b))
        val_mae_b = mean_absolute_error(y_val, y_pred_val_b)
        val_r2_b = r2_score(y_val, y_pred_val_b)    #summary
    print("\n" + "="*70)
    print("training complete dual model strategy")
    print("="*70)
    print(f"\nmodel A 90pct (out-of-sample validation)")
    print(f"training {split_info['train_samples']} validation {split_info['val_samples']}")
    print(f"val_rmse {model_90pct['val_metrics']['rmse']:.6f} (true validation)")
    
    if len(df) > HOLDOUT_SAMPLES:
        print(f"\nmodel B all-30 (holdout for testing)")
        print(f"training {len(train_b_df)} testing {len(test_b_df)}")
        print(f"test_rmse {model_100pct['val_metrics']['rmse']:.6f} (true out-of-sample)")
    else:
        print(f"\nmodel B 100pct (all data)")
        print(f"training {len(df)}")
        print(f"val_rmse {val_rmse_b:.6f} (insample)")
    
    print(f"\nartifacts saved")
    print(f"data/models/xgboost_90pct.pkl")
    print(f"data/models/lightgbm_90pct.pkl")
    print(f"data/models/ensemble_90pct.pkl")
    print(f"data/models/xgboost_100pct.pkl")
    print(f"data/models/lightgbm_100pct.pkl")
    print(f"data/models/ensemble_100pct.pkl")
    print(f"data/train/validation_baseline.json")
    print(f"data/train/dual_model_baseline.json")
    print(f"data/train/validation/predictions.parquet")
    
    print("="*70)


if __name__ == "__main__":
    main()
