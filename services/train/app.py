"""
Training Service (XGBoost)

Trains a single XGBoost model to forecast 5-day realized volatility (RV_5d).
Uses all rows except the last 30 from data/master_dataset.parquet.
Hyperparameters are loaded from final_model/model_config.json.

Saves:
- data/models/date=YYYY-MM-DD/xgboost.pkl
- data/models/latest/xgboost.pkl (overwrites previous latest)
- data/train/train_history.json (JSONL append)
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from storage import get_storage


# paths
MASTER_DATASET = "data/master_dataset.parquet"
MODEL_CONFIG = "final_model/model_config.json"
MODELS_ROOT = "data/models"
TRAIN_HISTORY = "data/train/train_history.json"


def compute_metrics(y_true, y_pred):
	return {
		"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred))
	}


def load_model_config(storage):
	"""Load model name + hyperparameters from final_model/model_config.json."""
	if not storage.exists(MODEL_CONFIG):
		raise FileNotFoundError(
			f"model_config.json not found at {MODEL_CONFIG}. Run the export notebook first."
		)

	config = storage.read_json(MODEL_CONFIG)
	hyper_block = config.get("hyperparams") or {}
	model_name = hyper_block.get("model_name") or "xgboost"
	hyperparams = hyper_block.get("hyperparams") or {}

	return model_name, hyperparams


def load_dataset(storage):
	"""Load master_dataset.parquet and sort by date."""
	if not storage.exists(MASTER_DATASET):
		raise FileNotFoundError(
			f"master_dataset.parquet not found at {MASTER_DATASET}. Run prepare_dataset first."
		)

	df = storage.read_parquet(MASTER_DATASET)
	df["date"] = pd.to_datetime(df["date"])
	df = df.sort_values("date").reset_index(drop=True)
	return df


def prepare_training_data(df):
	"""
	Prepare clean training data and test set, compute feature means.
	
	Drops (in order):
	1. Rows with null rv_5d (labels not available yet)
	2. Last 30 rows (test holdout)
	3. Rows with null features from train only (insufficient lookback data)
	
	Returns:
	- train_df: Clean training dataframe
	- test_df: Test holdout (last 30 rows, may have nulls)
	- feature_means: Dict of mean values for each feature (for prediction imputation)
	- feature_cols: List of feature column names
	"""
	print("\nPreparing training data:")
	original_rows = len(df)
	
	# Step 1: Drop rows with null rv_5d
	before_rv_drop = len(df)
	df_clean = df.dropna(subset=['rv_5d']).copy()
	rv_dropped = before_rv_drop - len(df_clean)
	print(f"  Dropped {rv_dropped} rows with null rv_5d: {len(df_clean)} rows remaining")
	
	# Step 2: Split last 30 rows as test, rest as train
	if len(df_clean) <= 30:
		raise ValueError("Not enough data after dropping null rv_5d to create test holdout")
	test_df = df_clean.iloc[-30:].copy()
	train_df = df_clean.iloc[:-30].copy()
	print(f"  Split: {len(train_df)} train, {len(test_df)} test (last 30 rows)")
	
	# Step 3: Drop rows with ANY null feature from train only
	feature_cols = [c for c in train_df.columns if c not in ['date', 'rv_5d']]
	before_feature_drop = len(train_df)
	train_df = train_df.dropna(subset=feature_cols)
	feature_dropped = before_feature_drop - len(train_df)
	print(f"  Dropped {feature_dropped} rows with null features from train: {len(train_df)} rows remaining")
	
	print(f"  Total train dropped: {original_rows - len(train_df) - len(test_df)} rows")
	print(f"  Final: {len(train_df)} train, {len(test_df)} test")
	
	# Compute feature means from training data only (for prediction imputation)
	feature_means = train_df[feature_cols].mean().to_dict()
	print(f"  Computed means for {len(feature_means)} features")
	
	return train_df, test_df, feature_means, feature_cols


def train_xgboost(train_df, feature_cols, hyperparams):
	"""Fit XGBoost on the training subset and return model + metrics."""
	X_train = train_df[feature_cols].values
	y_train = train_df["rv_5d"].values

	params = hyperparams.copy()
	params.setdefault("objective", "reg:squarederror")
	params.setdefault("n_jobs", -1)
	params.setdefault("eval_metric", "rmse")

	model = XGBRegressor(**params)
	model.fit(X_train, y_train, verbose=False)

	preds = model.predict(X_train)
	train_metrics = compute_metrics(y_train, preds)

	return model, train_metrics


def evaluate_test_set(model, test_df, feature_cols, feature_means):
	"""Evaluate model on test set with feature imputation for missing values."""
	X_test = test_df[feature_cols].copy()
	y_test = test_df["rv_5d"].values
	
	# Impute missing features using training means
	missing_mask = X_test.isnull()
	if missing_mask.any().any():
		for col in feature_cols:
			if missing_mask[col].any():
				X_test.loc[missing_mask[col], col] = feature_means[col]
		print(f"\nTest set: imputed missing features using training means")
	
	y_pred = model.predict(X_test.values)
	test_metrics = compute_metrics(y_test, y_pred)
	
	return test_metrics


def clear_prefix(storage, prefix):
	"""Remove existing files under a storage prefix (e.g., data/models/latest/)."""
	files = storage.list_files(prefix)
	for path in files:
		storage.delete(path)


def save_model(storage, model_payload, feature_means, test_metrics, trained_date):
	"""Persist model, feature_means, and benchmark results to dated folder and latest folder."""
	dated_prefix = f"{MODELS_ROOT}/date={trained_date}"
	latest_prefix = f"{MODELS_ROOT}/latest"

	clear_prefix(storage, f"{latest_prefix}/")

	storage.write_pickle(model_payload, f"{dated_prefix}/xgboost.pkl")
	storage.write_pickle(model_payload, f"{latest_prefix}/xgboost.pkl")
	
	# Save feature means for prediction imputation
	storage.write_json(feature_means, f"{dated_prefix}/feature_means.json")
	storage.write_json(feature_means, f"{latest_prefix}/feature_means.json")
	
	# Save benchmark results (test set performance)
	benchmark = {
		"trained_date": trained_date,
		"test_rmse": test_metrics["rmse"],
		"test_mae": test_metrics["mae"],
		"test_r2": test_metrics["r2"],
		"test_rows": 30
	}
	storage.write_json(benchmark, f"{dated_prefix}/benchmark_results.json")
	storage.write_json(benchmark, f"{latest_prefix}/benchmark_results.json")


def append_train_history(storage, entry):
	storage.append_jsonl(entry, TRAIN_HISTORY)


def main():
	print("=" * 60)
	print("Training Service (XGBoost)")
	print("=" * 60)

	storage = get_storage()
	model_name, hyperparams = load_model_config(storage)
	print(f"model_name: {model_name}")
	print(f"hyperparameters: {list(hyperparams.keys())}")

	df = load_dataset(storage)
	print(f"Loaded master_dataset: {len(df)} rows")
	
	if len(df) <= 35:
		raise ValueError("master_dataset must have more than 35 rows to train.")

	train_df, test_df, feature_means, feature_cols = prepare_training_data(df)
	print(f"\nTraining on {len(train_df)} clean rows")
	print(f"Date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")

	model, train_metrics = train_xgboost(train_df, feature_cols, hyperparams)
	
	# Evaluate on test set (last 30 rows)
	test_metrics = evaluate_test_set(model, test_df, feature_cols, feature_means)

	trained_date = datetime.utcnow().strftime("%Y-%m-%d")
	model_payload = {
		"model_name": model_name,
		"model": model,
		"feature_cols": feature_cols,
		"hyperparams": hyperparams,
		"trained_at": trained_date
	}

	save_model(storage, model_payload, feature_means, test_metrics, trained_date)

	history_entry = {
		"training_date": trained_date,
		"data_start_date": train_df["date"].min().strftime("%Y-%m-%d"),
		"data_end_date": train_df["date"].max().strftime("%Y-%m-%d"),
		"model_name": model_name,
		"train_rmse": train_metrics["rmse"],
		"train_mae": train_metrics["mae"],
		"train_r2": train_metrics["r2"],
		"test_rmse": test_metrics["rmse"],
		"test_mae": test_metrics["mae"],
		"test_r2": test_metrics["r2"],
		"train_rows": len(train_df),
		"test_rows": len(test_df),
		"feature_means_path": f"{MODELS_ROOT}/date={trained_date}/feature_means.json"
	}
	append_train_history(storage, history_entry)

	print("\ntrain metrics:")
	print(f"  rmse: {train_metrics['rmse']:.6f}")
	print(f"  mae:  {train_metrics['mae']:.6f}")
	print(f"  r2:   {train_metrics['r2']:.4f}")
	
	print("\ntest metrics:")
	print(f"  rmse: {test_metrics['rmse']:.6f}")
	print(f"  mae:  {test_metrics['mae']:.6f}")
	print(f"  r2:   {test_metrics['r2']:.4f}")

	print("\nSaved:")
	print(f"  model (dated): {MODELS_ROOT}/date={trained_date}/xgboost.pkl")
	print(f"  model (latest): {MODELS_ROOT}/latest/xgboost.pkl")
	print(f"  feature_means (dated): {MODELS_ROOT}/date={trained_date}/feature_means.json")
	print(f"  feature_means (latest): {MODELS_ROOT}/latest/feature_means.json")
	print(f"  benchmark (dated): {MODELS_ROOT}/date={trained_date}/benchmark_results.json")
	print(f"  benchmark (latest): {MODELS_ROOT}/latest/benchmark_results.json")
	print(f"  history: {TRAIN_HISTORY}")


if __name__ == "__main__":
	main()
