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
import mlflow
import mlflow.sklearn

#config
MASTER_DATASET = Path("data/master_dataset.parquet")
OUTPUT_DIR = Path("data/predictions")
MODELS_DIR = Path("models")
MODEL_CONFIG = Path("final_model/model_config.json")
BENCHMARK_CONFIG = Path("final_model/benchmark_results.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "volatility-forecasting"
MODEL_NAME = "volatility-forecaster-ensemble"


def load_master_dataset():
    """load master dataset from prepare_dataset service"""
    if not MASTER_DATASET.exists():
        raise FileNotFoundError(
            f"master dataset not found at {MASTER_DATASET}. "
            f"run prepare_dataset service first: docker-compose up prepare_dataset"
        )
    
    print(f"loading master dataset from {MASTER_DATASET}")
    df = pd.read_parquet(MASTER_DATASET)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"loaded {len(df)} samples")
    print(f"date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"features: {len([c for c in df.columns if c not in ['date', 'rv_5d']])}")
    
    return df


def load_model_config():
    """load model config from final_model/model_config.json"""
    if not MODEL_CONFIG.exists():
        raise FileNotFoundError(
            f"model config not found at {MODEL_CONFIG}. "
            f"run notebook 2_model_selection.ipynb section 9 to export config"
        )
    
    print(f"\nloading model config from {MODEL_CONFIG}")
    with open(MODEL_CONFIG) as f:
        config = json.load(f)
    
    print(f"architecture: {config['model_architecture']}")
    print(f"base models: {[m['name'] for m in config['base_models']]}")
    print(f"meta learner: {config['meta_learner']['type']}")
    
    #load benchmarks if available
    benchmarks = None
    if BENCHMARK_CONFIG.exists():
        with open(BENCHMARK_CONFIG) as f:
            benchmarks = json.load(f)
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
            'mae': mean_absolute_error(y_val, ensemble_val_pred)),
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
        'val_pred': ensemble_val_pred
    }


def save_model(models, feature_cols, filename):
    """save model pkl"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        'xgb_model': models['xgb_model'],
        'lgbm_model': models['lgbm_model'],
        'meta_model': models['meta_model'],
        'feature_cols': feature_cols
    }
    
    model_path = MODELS_DIR / filename
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)
    
    print(f"saved {model_path}")
    return model_path


def save_predictions(train_df, val_df, train_preds, val_preds):
    """save predictions parquet"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_results = pd.DataFrame({
        'date': train_df['date'],
        'actual': train_df['rv_5d'],
        'predicted': train_preds,
        'split': 'train'
    })
    
    val_results = pd.DataFrame({
        'date': val_df['date'],
        'actual': val_df['rv_5d'],
        'predicted': val_preds,
        'split': 'val'
    })
    
    all_results = pd.concat([train_results, val_results], ignore_index=True)
    outpath = OUTPUT_DIR / "predictions.parquet"
    all_results.to_parquet(outpath, index=False)
    
    print(f"saved predictions {outpath}")


def save_baselines(model_a_metrics, model_b_metrics, split_info_a, split_info_b=None):
    """
    save baseline metrics for monitoring
    
    maintains two file types:
    1. current baseline json - latest training run for monitor
    2. historical jsonl - append-only history of all runs
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    #validation baseline for model A - current
    validation_baseline = {
        'validation_rmse': model_a_metrics['val_metrics']['rmse'],
        'validation_mae': model_a_metrics['val_metrics']['mae'],
        'validation_r2': model_a_metrics['val_metrics']['r2'],
        'timestamp': datetime.now().isoformat(),
        'train_val_split': 0.9,
        'date_range': {
            'train_start': split_info_a['train_start_date'],
            'train_end': split_info_a['train_end_date'],
            'val_start': split_info_a['val_start_date'],
            'val_end': split_info_a['val_end_date']
        }
    }
    
    baseline_path = MODELS_DIR / "validation_baseline.json"
    with open(baseline_path, 'w') as f:
        json.dump(validation_baseline, f, indent=2)
    print(f"saved validation baseline {baseline_path}")
    
    #dual model baseline for monitor comparison - current
    dual_baseline = {
        'timestamp': datetime.now().isoformat(),
        'model_a': {
            'training_samples': split_info_a['train_samples'],
            'training_date_start': split_info_a['train_start_date'],
            'training_date_end': split_info_a['train_end_date'],
            'validation_rmse': float(model_a_metrics['val_metrics']['rmse']),
            'validation_mae': float(model_a_metrics['val_metrics']['mae']),
            'validation_r2': float(model_a_metrics['val_metrics']['r2']),
            'validation_samples': split_info_a['val_samples']
        }
    }
    
    #model B info
    if split_info_b:
        #model B has own train/test split
        dual_baseline['model_b'] = {
            'training_samples': split_info_b['training_samples'],
            'training_date_start': split_info_b['training_date_start'],
            'training_date_end': split_info_b['training_date_end'],
            'test_samples': split_info_b['test_samples'],
            'test_date_start': split_info_b['test_date_start'],
            'test_date_end': split_info_b['test_date_end'],
            'test_rmse': float(model_b_metrics['val_rmse']) if 'val_rmse' in model_b_metrics else None,
            'test_mae': float(model_b_metrics['val_mae']) if 'val_mae' in model_b_metrics else None,
            'test_r2': float(model_b_metrics['val_r2']) if 'val_r2' in model_b_metrics else None,
            'note': 'model B held out last 30 samples for testing'
        }
    else:
        #model B used 100% of data (fallback)
        dual_baseline['model_b'] = {
            'training_samples': split_info_a['train_samples'] + split_info_a['val_samples'],
            'training_date_start': split_info_a['train_start_date'],
            'training_date_end': split_info_a['val_end_date'],
            'validation_rmse': float(model_b_metrics['val_rmse_insample']) if 'val_rmse_insample' in model_b_metrics else None,
            'validation_mae': float(model_b_metrics['val_mae_insample']) if 'val_mae_insample' in model_b_metrics else None,
            'validation_r2': float(model_b_metrics['val_r2_insample']) if 'val_r2_insample' in model_b_metrics else None,
            'note': 'model B metrics on val set are insample'
        }
    
    dual_path = MODELS_DIR / "dual_model_baseline.json"
    with open(dual_path, 'w') as f:
        json.dump(dual_baseline, f, indent=2)
    print(f"saved dual baseline {dual_path}")
    
    #historical record (append-only)
    training_history_path = MODELS_DIR / "training_history.jsonl"
    
    history_record = {
        'timestamp': datetime.now().isoformat(),
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'model_a': dual_baseline['model_a'],
        'model_b': dual_baseline['model_b']
    }
    
    #append to jsonl
    with open(training_history_path, 'a') as f:
        f.write(json.dumps(history_record) + '\n')
    
    print(f"appended training record {training_history_path}")


def log_to_mlflow(models, feature_cols, train_metrics, val_metrics, config, benchmarks, split_info):
    """log model A to mlflow"""
    print("\n" + "="*60)
    print("logging to mlflow")
    print("="*60)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"ensemble-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        #log params
        mlflow.log_param('model_architecture', config['model_architecture'])
        mlflow.log_param('n_base_models', len(config['base_models']))
        mlflow.log_param('n_features', len(feature_cols))
        mlflow.log_param('n_train_samples', split_info['train_samples'])
        mlflow.log_param('n_val_samples', split_info['val_samples'])
        
        #log metrics
        mlflow.log_metrics({
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2']
        })
        
        #log date ranges
        mlflow.log_param('train_start_date', split_info['train_start_date'])
        mlflow.log_param('train_end_date', split_info['train_end_date'])
        mlflow.log_param('val_start_date', split_info['val_start_date'])
        mlflow.log_param('val_end_date', split_info['val_end_date'])
        
        #benchmark comparison
        if benchmarks:
            expected_rmse = benchmarks['expected_performance']['val_rmse']
            actual_rmse = val_metrics['rmse']
            diff = actual_rmse - expected_rmse
            diff_pct = (diff / expected_rmse) * 100
            
            mlflow.log_metrics({
                'expected_val_rmse': expected_rmse,
                'rmse_diff': diff,
                'rmse_diff_pct': diff_pct
            })
            
            print(f"\nbenchmark comparison")
            print(f"expected {expected_rmse:.6f} actual {actual_rmse:.6f} diff {diff:+.6f} ({diff_pct:+.2f}%)")
        
        #log model
        ensemble_artifact = {
            'xgb_model': models['xgb_model'],
            'lgbm_model': models['lgbm_model'],
            'meta_model': models['meta_model'],
            'feature_cols': feature_cols
        }
        
        mlflow.sklearn.log_model(
            ensemble_artifact,
            "ensemble",
            registered_model_name=MODEL_NAME
        )
        
        #log config files
        mlflow.log_artifact(MODEL_CONFIG)
        if BENCHMARK_CONFIG.exists():
            mlflow.log_artifact(BENCHMARK_CONFIG)
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nmlflow run_id {run_id}")
        print(f"model registered as {MODEL_NAME}")


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
    
    model_a = train_ensemble(X_train, y_train, X_val, y_val, config, "model A 90pct")
    
    #save model A
    save_model(model_a, feature_cols, "model_90pct.pkl")
    save_model(model_a, feature_cols, "latest_ensemble.pkl")  #backward compat
    
    #save predictions
    save_predictions(train_df, val_df, model_a['train_pred'], model_a['val_pred'])
    
    #log to mlflow
    log_to_mlflow(
        model_a, feature_cols,
        model_a['train_metrics'], model_a['val_metrics'],
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
        model_b = train_ensemble(X_train_b, y_train_b, X_test_b, y_test_b, config, "model B all-30")
        
        #save model B
        save_model(model_b, feature_cols, "model_100pct.pkl")
        
        #model B eval metrics (true out-of-sample on last 30)
        print(f"\nmodel B all-30 on last 30 samples (true out-of-sample)")
        print(f"rmse {model_b['val_metrics']['rmse']:.6f} mae {model_b['val_metrics']['mae']:.6f} r2 {model_b['val_metrics']['r2']:.4f}")
        print(f"note: these are TRUE out-of-sample metrics, model never saw this data")
        
        #save baselines for monitoring
        model_b_eval = {
            'val_rmse': model_b['val_metrics']['rmse'],
            'val_mae': model_b['val_metrics']['mae'],
            'val_r2': model_b['val_metrics']['r2']
        }
        
        #split info for model B
        split_info_b = {
            'training_samples': len(train_b_df),
            'training_date_start': str(train_b_df['date'].min().date()),
            'training_date_end': str(train_b_df['date'].max().date()),
            'test_samples': len(test_b_df),
            'test_date_start': str(test_b_df['date'].min().date()),
            'test_date_end': str(test_b_df['date'].max().date())
        }
        
        save_baselines(model_a, model_b_eval, split_info, split_info_b)
    else:
        print(f"\nwarning: dataset too small {len(df)} samples to hold out {HOLDOUT_SAMPLES}")
        print(f"training model B on 100pct data instead")
        
        model_b = train_ensemble(X_full, y_full, None, None, config, "model B 100pct")
        save_model(model_b, feature_cols, "model_100pct.pkl")
        
        #eval on validation set (insample for model B)
        X_meta_val_b = np.column_stack([
            model_b['xgb_model'].predict(X_val),
            model_b['lgbm_model'].predict(X_val)
        ])
        y_pred_val_b = model_b['meta_model'].predict(X_meta_val_b)
        
        val_rmse_b = np.sqrt(mean_squared_error(y_val, y_pred_val_b))
        val_mae_b = mean_absolute_error(y_val, y_pred_val_b)
        val_r2_b = r2_score(y_val, y_pred_val_b)
        
        model_b_eval = {
            'val_rmse_insample': val_rmse_b,
            'val_mae_insample': val_mae_b,
            'val_r2_insample': val_r2_b
        }
        save_baselines(model_a, model_b_eval, split_info, None)
    
    #summary
    print("\n" + "="*70)
    print("training complete dual model strategy")
    print("="*70)
    print(f"\nmodel A 90pct (out-of-sample validation)")
    print(f"training {split_info['train_samples']} validation {split_info['val_samples']}")
    print(f"val_rmse {model_a['val_metrics']['rmse']:.6f} (true validation)")
    
    if len(df) > HOLDOUT_SAMPLES:
        print(f"\nmodel B all-30 (holdout for testing)")
        print(f"training {len(train_b_df)} testing {len(test_b_df)}")
        print(f"test_rmse {model_b['val_metrics']['rmse']:.6f} (true out-of-sample)")
    else:
        print(f"\nmodel B 100pct (all data)")
        print(f"training {len(df)}")
        print(f"val_rmse {val_rmse_b:.6f} (insample)")
    
    print(f"\nartifacts saved")
    print(f"models/model_90pct.pkl")
    print(f"models/model_100pct.pkl")
    print(f"models/validation_baseline.json")
    print(f"models/dual_model_baseline.json")
    print(f"data/predictions/predictions.parquet")
    
    print("="*70)


if __name__ == "__main__":
    main()
