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


def train_xgboost(train_df, hyperparams):
	"""Fit XGBoost on the training subset and return model + metrics."""
	feature_cols = [c for c in train_df.columns if c not in ["date", "rv_5d"]]
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

	return model, feature_cols, train_metrics


def clear_prefix(storage, prefix):
	"""Remove existing files under a storage prefix (e.g., data/models/latest/)."""
	files = storage.list_files(prefix)
	for path in files:
		storage.delete(path)


def save_model(storage, model_payload, trained_date):
	"""Persist model to dated folder and latest folder."""
	dated_prefix = f"{MODELS_ROOT}/date={trained_date}"
	latest_prefix = f"{MODELS_ROOT}/latest"

	clear_prefix(storage, f"{latest_prefix}/")

	storage.write_pickle(model_payload, f"{dated_prefix}/xgboost.pkl")
	storage.write_pickle(model_payload, f"{latest_prefix}/xgboost.pkl")


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
	if len(df) <= 30:
		raise ValueError("master_dataset must have more than 30 rows to train.")

	train_df = df.iloc[:-30].copy()
	print(f"train rows: {len(train_df)}")
	print(f"train date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")

	model, feature_cols, train_metrics = train_xgboost(train_df, hyperparams)

	trained_date = datetime.utcnow().strftime("%Y-%m-%d")
	model_payload = {
		"model_name": model_name,
		"model": model,
		"feature_cols": feature_cols,
		"hyperparams": hyperparams,
		"trained_at": trained_date
	}

	save_model(storage, model_payload, trained_date)

	history_entry = {
		"training_date": trained_date,
		"data_start_date": train_df["date"].min().strftime("%Y-%m-%d"),
		"data_end_date": train_df["date"].max().strftime("%Y-%m-%d"),
		"model_name": model_name,
		"train_rmse": train_metrics["rmse"],
		"train_mae": train_metrics["mae"],
		"train_r2": train_metrics["r2"],
		"train_rows": len(train_df)
	}
	append_train_history(storage, history_entry)

	print("\ntrain metrics:")
	print(f"  rmse: {train_metrics['rmse']:.6f}")
	print(f"  mae:  {train_metrics['mae']:.6f}")
	print(f"  r2:   {train_metrics['r2']:.4f}")

	print("\nSaved:")
	print(f"  model (dated): {MODELS_ROOT}/date={trained_date}/xgboost.pkl")
	print(f"  model (latest): {MODELS_ROOT}/latest/xgboost.pkl")
	print(f"  history: {TRAIN_HISTORY}")


if __name__ == "__main__":
	main()
