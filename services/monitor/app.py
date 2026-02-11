import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from storage import get_storage


MASTER_DATASET = "data/master_dataset.parquet"
PREDICTIONS = "data/predict/latest/predictions.parquet"
BENCHMARK = "data/models/latest/benchmark_results.json"
MONITOR_OUTPUT = "data/monitor/latest"


def compute_metrics(y_true, y_pred):
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	mae = float(mean_absolute_error(y_true, y_pred))
	r2 = float(r2_score(y_true, y_pred))
	mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
	tolerance = 0.20
	within_tolerance = np.abs((y_pred - y_true) / y_true) <= tolerance
	hit_rate = float(np.mean(within_tolerance))
	return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "hit_rate": hit_rate}


def classify_regime(volatility):
	if volatility < 0.015:
		return "LOW", "LOW"
	elif volatility < 0.03:
		return "NORMAL", "NORMAL"
	elif volatility < 0.05:
		return "ELEVATED", "ELEVATED"
	else:
		return "HIGH", "HIGH"


def detect_trend(recent_values):
	if len(recent_values) < 10:
		return "STABLE", 0.0
	avg_recent = float(np.mean(recent_values[-5:]))
	avg_prior = float(np.mean(recent_values[-10:-5]))
	if avg_prior == 0:
		return "STABLE", 0.0
	pct_change = ((avg_recent - avg_prior) / avg_prior) * 100
	if pct_change > 10:
		return "INCREASING", pct_change
	elif pct_change < -10:
		return "DECREASING", pct_change
	else:
		return "STABLE", pct_change


def generate_insights(current_pred, trend, trend_pct, recent_metrics, regime, percentile, benchmark_comparison):
	insights = []
	if trend == "INCREASING" and abs(trend_pct) > 15:
		insights.append({"type": "TREND", "severity": "WARNING", "message": f"Volatility trending higher over past 5 days"})
	elif trend == "DECREASING" and abs(trend_pct) > 15:
		insights.append({"type": "TREND", "severity": "INFO", "message": f"Volatility trending lower over past 5 days"})
	if regime == "HIGH":
		insights.append({"type": "REGIME", "severity": "ALERT", "message": f"Volatility in HIGH regime expect larger market swings"})
	elif regime == "LOW" and percentile is not None and percentile < 10:
		insights.append({"type": "REGIME", "severity": "INFO", "message": f"Volatility unusually low regime shift possible"})
	
	# Check RMSE vs benchmark
	if benchmark_comparison and benchmark_comparison.get("status"):
		status = benchmark_comparison["status"]
		if status == "DEGRADED":
			test_rmse = benchmark_comparison["test_rmse"]
			prod_rmse = benchmark_comparison["production_rmse"]
			diff_pct = benchmark_comparison["rmse_diff_pct"]
			insights.append({
				"type": "RMSE", 
				"severity": "ALERT", 
				"message": f"RMSE degraded: benchmark={test_rmse:.6f} production={prod_rmse:.6f} ({diff_pct:+.1f}%) - retrain model"
			})
		elif status == "ACCEPTABLE":
			test_rmse = benchmark_comparison["test_rmse"]
			prod_rmse = benchmark_comparison["production_rmse"]
			diff_pct = benchmark_comparison["rmse_diff_pct"]
			insights.append({
				"type": "RMSE", 
				"severity": "WARNING", 
				"message": f"RMSE acceptable: benchmark={test_rmse:.6f} production={prod_rmse:.6f} ({diff_pct:+.1f}%) - monitor closely"
			})
	return insights


def compute_rolling_metrics(merged_df, window=30):
	merged_df = merged_df.sort_values('date').copy()
	merged_df['date_str'] = merged_df['date'].dt.strftime('%Y-%m-%d')
	rolling_results = []
	for i in range(window, len(merged_df) + 1):
		window_df = merged_df.iloc[i-window:i]
		y_true = window_df['rv_5d'].values
		y_pred = window_df['prediction'].values
		metrics = compute_metrics(y_true, y_pred)
		rolling_results.append({"end_date": window_df['date_str'].iloc[-1], "window_size": window, "rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"], "mape": metrics["mape"], "hit_rate": metrics["hit_rate"]})
	return rolling_results


def main():
	print("=" * 60)
	print("Monitor Service")
	print("=" * 60)
	storage = get_storage()
	print("\nLoading data...")
	master_df = storage.read_parquet(MASTER_DATASET)
	master_df['date'] = pd.to_datetime(master_df['date'])
	master_df = master_df.sort_values('date').reset_index(drop=True)
	predictions_df = storage.read_parquet(PREDICTIONS)
	predictions_df['date'] = pd.to_datetime(predictions_df['date'])
	benchmark = storage.read_json(BENCHMARK)
	print(f"Loaded master dataset: {len(master_df)} rows")
	print(f"Loaded predictions: {len(predictions_df)} rows")
	print(f"Loaded benchmark: Test RMSE = {benchmark['test_rmse']:.6f}")
	merged_df = predictions_df.merge(master_df[['date', 'rv_5d']], on='date', how='left')
	with_actuals = merged_df[merged_df['rv_5d'].notna()].copy()
	without_actuals = merged_df[merged_df['rv_5d'].isna()].copy()
	print(f"\nPredictions with actuals: {len(with_actuals)}")
	print(f"Predictions without actuals: {len(without_actuals)}")
	overall_metrics = None
	if len(with_actuals) > 0:
		y_true = with_actuals['rv_5d'].values
		y_pred = with_actuals['prediction'].values
		overall_metrics = compute_metrics(y_true, y_pred)
		print(f"\nOverall Performance:")
		print(f"  RMSE: {overall_metrics['rmse']:.6f}")
		print(f"  MAE:  {overall_metrics['mae']:.6f}")
		print(f"  MAPE: {overall_metrics['mape']:.1f} percent")
		print(f"  Hit Rate: {overall_metrics['hit_rate']*100:.1f} percent")
	recent_30d_metrics = None
	if len(with_actuals) >= 30:
		recent_30 = with_actuals.tail(30)
		y_true = recent_30['rv_5d'].values
		y_pred = recent_30['prediction'].values
		recent_30d_metrics = compute_metrics(y_true, y_pred)
	elif len(with_actuals) > 0:
		y_true = with_actuals['rv_5d'].values
		y_pred = with_actuals['prediction'].values
		recent_30d_metrics = compute_metrics(y_true, y_pred)
	rolling_metrics = []
	if len(with_actuals) >= 30:
		rolling_metrics = compute_rolling_metrics(with_actuals, window=30)
		print(f"\nRolling 30 day windows: {len(rolling_metrics)}")
	
	# Compute benchmark comparison first
	benchmark_comparison = {"test_rmse": benchmark['test_rmse'], "production_rmse": overall_metrics['rmse'] if overall_metrics else None, "trained_date": benchmark.get('trained_date', 'unknown')}
	if overall_metrics:
		rmse_diff = overall_metrics['rmse'] - benchmark['test_rmse']
		rmse_diff_pct = (rmse_diff / benchmark['test_rmse']) * 100
		if abs(rmse_diff_pct) < 10:
			status = "GOOD"
		elif abs(rmse_diff_pct) < 30:
			status = "ACCEPTABLE"
		else:
			status = "DEGRADED"
		benchmark_comparison.update({"rmse_diff": rmse_diff, "rmse_diff_pct": rmse_diff_pct, "status": status})
		print(f"\nBenchmark Comparison:")
		print(f"  Test RMSE: {benchmark['test_rmse']:.6f}")
		print(f"  Production RMSE: {overall_metrics['rmse']:.6f}")
		print(f"  Status: {status}")
	
	current_forecast = None
	if len(predictions_df) > 0:
		latest = predictions_df.iloc[-1]
		current_pred = float(latest['prediction'])
		regime, regime_emoji = classify_regime(current_pred)
		trend = "UNKNOWN"
		trend_pct = 0.0
		if len(predictions_df) >= 10:
			recent_preds = predictions_df['prediction'].values[-10:]
			trend, trend_pct = detect_trend(recent_preds)
		percentile = None
		if len(with_actuals) > 0:
			historical_rv = with_actuals['rv_5d'].values
			percentile = float((historical_rv < current_pred).mean() * 100)
		insights = generate_insights(current_pred, trend, trend_pct, recent_30d_metrics, regime, percentile, benchmark_comparison)
		current_forecast = {"date": latest['date'].strftime('%Y-%m-%d'), "volatility": current_pred, "volatility_pct": f"{current_pred*100:.2f} percent", "regime": regime, "regime_emoji": regime_emoji, "trend": trend, "trend_change_pct": trend_pct, "percentile": percentile, "confidence": "MEDIUM", "insights": insights}
		print(f"\nCurrent Forecast: {current_forecast['volatility_pct']} {regime}")
		print(f"  Trend: {trend}")
		print(f"  Insights: {len(insights)}")
	
	output = {"metadata": {"generated_at": datetime.utcnow().isoformat() + "Z", "total_predictions": len(predictions_df), "predictions_with_actuals": len(with_actuals), "predictions_without_actuals": len(without_actuals), "model_trained_date": benchmark.get('trained_date', 'unknown')}, "current_forecast": current_forecast, "latest_predictions": [{"date": row['date'].strftime('%Y-%m-%d'), "prediction": float(row['prediction']), "prediction_pct": f"{row['prediction']*100:.2f} percent"} for _, row in without_actuals.tail(5).iterrows()] if len(without_actuals) > 0 else [], "predictions_vs_actuals": [{"date": row['date'].strftime('%Y-%m-%d'), "actual_rv_5d": float(row['rv_5d']), "prediction": float(row['prediction']), "error": float(row['prediction'] - row['rv_5d']), "error_pct": f"{((row['prediction'] - row['rv_5d']) / row['rv_5d'] * 100):+.1f} percent"} for _, row in with_actuals.tail(30).iterrows()] if len(with_actuals) > 0 else [], "performance": {"overall": overall_metrics, "recent_30d": recent_30d_metrics, "rolling_30d": rolling_metrics, "benchmark_comparison": benchmark_comparison}}
	monitor_file = f"{MONITOR_OUTPUT}/monitor.json"
	storage.write_json(output, monitor_file)
	print(f"\nSaved: {monitor_file}")
	summary = {"metadata": output["metadata"], "current_forecast": output["current_forecast"], "latest_predictions": output["latest_predictions"][:5], "performance": {"overall": overall_metrics, "recent_30d": recent_30d_metrics, "benchmark_comparison": benchmark_comparison}}
	summary_file = f"{MONITOR_OUTPUT}/summary.json"
	storage.write_json(summary, summary_file)
	print(f"Saved: {summary_file}")
	alerts = {"generated_at": output["metadata"]["generated_at"], "current_forecast_date": current_forecast["date"] if current_forecast else None, "insights": current_forecast["insights"] if current_forecast else [], "benchmark_comparison": benchmark_comparison}
	alerts_file = f"{MONITOR_OUTPUT}/alerts.json"
	storage.write_json(alerts, alerts_file)
	print(f"Saved: {alerts_file}")
	print("\n" + "=" * 60)
	print("Monitor completed successfully")
	print("=" * 60)


if __name__ == "__main__":
	main()
