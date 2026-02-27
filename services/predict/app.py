"""
Prediction Service (XGBoost)

Loads the latest trained model from data/models/latest/xgboost.pkl,
scores data/predict_dataset.parquet, and writes predictions to:
- data/predict/date=YYYY-MM-DD/predictions.parquet
- data/predict/latest/predictions.parquet
"""

from datetime import datetime

import pandas as pd

from storage import Storage


# Storage
storage = Storage()

# Paths
MODEL_PATH = "data/models/latest/xgboost.pkl"
FEATURE_MEANS_PATH = "data/models/latest/feature_means.json"
PREDICT_DATASET = "data/predict_dataset.parquet"
PREDICT_ROOT = "data/predict"


def get_run_date():
    return datetime.utcnow().strftime("%Y-%m-%d")


def load_model():
    if not storage.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Latest model not found at {MODEL_PATH}. Run train service first."
        )
    model_payload = storage.read_pickle(MODEL_PATH)

    if "model" not in model_payload or "feature_cols" not in model_payload:
        raise ValueError("Model payload missing required keys: model, feature_cols")

    return model_payload


def load_feature_means():
    """Load feature means for imputing missing values during prediction."""
    if not storage.exists(FEATURE_MEANS_PATH):
        raise FileNotFoundError(
            f"feature_means.json not found at {FEATURE_MEANS_PATH}. Run train service first."
        )
    
    feature_means = storage.read_json(FEATURE_MEANS_PATH)
    print(f"Loaded feature means for {len(feature_means)} features")
    return feature_means


def impute_missing_features(predict_df, feature_means, feature_cols):
    """
    Impute missing feature values with training set means.
    Reports which features were imputed and how many rows were affected.
    """
    imputed_count = 0
    imputed_features = []
    
    for col in feature_cols:
        if col not in predict_df.columns:
            # Column missing entirely - add it with mean value
            if col in feature_means:
                predict_df[col] = feature_means[col]
                imputed_features.append(f"{col} (all rows)")
                imputed_count += len(predict_df)
        else:
            # Check for null values in existing column
            null_mask = predict_df[col].isnull()
            null_count = null_mask.sum()
            
            if null_count > 0 and col in feature_means:
                predict_df.loc[null_mask, col] = feature_means[col]
                imputed_features.append(f"{col} ({null_count} rows)")
                imputed_count += null_count
    
    if imputed_features:
        print(f"\n⚠️  Imputed {imputed_count} missing values:")
        for feature in imputed_features:
            print(f"     {feature}")
    else:
        print("\n✅ No missing features - all data complete")
    
    return predict_df


def load_predict_dataset():
    if not storage.exists(PREDICT_DATASET):
        raise FileNotFoundError(
            f"predict_dataset.parquet not found at {PREDICT_DATASET}. Run prepare_dataset first."
        )

    df = storage.read_parquet(PREDICT_DATASET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_predictions(model_payload, predict_df, feature_means):
    """Make predictions with automatic feature imputation if needed."""
    feature_cols = model_payload["feature_cols"]
    model = model_payload["model"]
    
    # Impute any missing features before prediction
    predict_df = impute_missing_features(predict_df, feature_means, feature_cols)

    X = predict_df[feature_cols].values
    preds = model.predict(X)

    output = pd.DataFrame({
        "date": predict_df["date"],
        "prediction": preds,
        "model_name": model_payload.get("model_name", "xgboost")
    })
    output["prediction_timestamp"] = datetime.utcnow().isoformat() + "Z"
    return output


def clear_prefix(prefix):
    files = storage.list_files(prefix)
    for path in files:
        storage.delete(path)


def save_predictions(pred_df, run_date):
    dated_prefix = f"{PREDICT_ROOT}/date={run_date}"
    latest_prefix = f"{PREDICT_ROOT}/latest"

    clear_prefix(f"{latest_prefix}/")

    storage.write_parquet(pred_df, f"{dated_prefix}/predictions.parquet")
    storage.write_parquet(pred_df, f"{latest_prefix}/predictions.parquet")


def main():
    print("=" * 60)
    print("Prediction Service (XGBoost)")
    print("=" * 60)

    model_payload = load_model()
    feature_means = load_feature_means()
    predict_df = load_predict_dataset()

    print(f"predict rows: {len(predict_df)}")
    print(
        f"predict date range: {predict_df['date'].min().date()} to {predict_df['date'].max().date()}"
    )

    pred_df = make_predictions(model_payload, predict_df, feature_means)
    run_date = get_run_date()
    save_predictions(pred_df, run_date)

    print("\n✅ Saved:")
    print(f"  dated: {PREDICT_ROOT}/date={run_date}/predictions.parquet")
    print(f"  latest: {PREDICT_ROOT}/latest/predictions.parquet")


if __name__ == "__main__":
    main()


def lambda_handler(event, context):
    """AWS Lambda entry point."""
    try:
        main()
        response = {
            'statusCode': 200,
            'body': '{"message": "Prediction completed successfully"}'
        }
        # Chain: trigger vf-monitor asynchronously
        try:
            import boto3
            boto3.client('lambda', region_name='us-east-1').invoke(
                FunctionName='vf-monitor',
                InvocationType='Event'
            )
            print("Triggered vf-monitor")
        except Exception as chain_err:
            print(f"WARNING: Failed to trigger vf-monitor: {chain_err}")
        return response
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': f'{{"message": "Prediction failed: {str(e)}"}}'
        }
