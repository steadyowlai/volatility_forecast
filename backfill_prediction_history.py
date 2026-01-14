"""
One-time backfill script to create prediction_history.jsonl from existing predictions.

This reads all existing prediction partitions and writes them to the consolidated history file.
After this runs once, the predict service will continue appending new predictions automatically.
"""

import json
import pandas as pd
import glob
from pathlib import Path

# Find all prediction partition files
prediction_files = sorted(glob.glob("data/predictions/date=*/prediction.parquet"))

print(f"Found {len(prediction_files)} existing predictions")
print("=" * 60)

# Read and consolidate
history_records = []

for file_path in prediction_files:
    try:
        df = pd.read_parquet(file_path)
        
        # Extract the single row (each partition has 1 prediction)
        record = {
            'prediction_date': str(df['prediction_date'].iloc[0]),
            'predicted_volatility': float(df['predicted_volatility'].iloc[0]),
            'features_date': str(df['features_date'].iloc[0]),
            'model_path': str(df['model_path'].iloc[0]),
            'prediction_timestamp': df['prediction_timestamp'].iloc[0].isoformat() + 'Z'
        }
        
        history_records.append(record)
        print(f"✓ {record['prediction_date']}: {record['predicted_volatility']:.6f}")
        
    except Exception as e:
        print(f"✗ Failed to read {file_path}: {e}")

# Write to prediction_history.jsonl
history_file = Path("data/predictions/prediction_history.jsonl")

with open(history_file, 'w') as f:
    for record in history_records:
        f.write(json.dumps(record) + '\n')

print("=" * 60)
print(f"✅ Created {history_file}")
print(f"   Total predictions: {len(history_records)}")
print(f"   Date range: {history_records[0]['prediction_date']} to {history_records[-1]['prediction_date']}")
print("\nFuture predictions will be automatically appended to this file.")
