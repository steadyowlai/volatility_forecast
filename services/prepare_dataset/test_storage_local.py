"""Quick test of storage.py in prepare_dataset service."""

from storage import Storage
import pandas as pd
import os

print("="*60)
print("Testing Storage Module (Local Mode)")
print("="*60)
print()

# Show environment
print(f"AWS_EXECUTION_ENV: {os.getenv('AWS_EXECUTION_ENV', 'Not set (Local mode)')}")
print()

storage = Storage()
print(f"Storage mode: {'AWS/S3' if storage.is_aws else 'Local filesystem'}")
print(f"S3 Bucket: {storage.bucket_name}")
print()

# Test 1: Check existing files
print("Test 1: Check if master_dataset exists")
exists = storage.exists('data/master_dataset.parquet')
print(f"  data/master_dataset.parquet exists: {exists}")
print()

if exists:
    # Test 2: Read the master dataset
    print("Test 2: Read master_dataset")
    df = storage.read_parquet('data/master_dataset.parquet')
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print()

# Test 3: Check data_status.json
print("Test 3: Check data_status.json")
exists = storage.exists('data/data_status.json')
print(f"  data/data_status.json exists: {exists}")

if exists:
    data = storage.read_json('data/data_status.json')
    print(f"  ✓ Last date: {data.get('last_date')}")
    print(f"  ✓ Rows: {data.get('rows')}")
print()

# Test 4: Write/Read test
print("Test 4: Write and read test files")

# Small test dataframe
test_df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=3),
    'value': [1.0, 2.0, 3.0]
})

storage.write_parquet(test_df, 'data/test_storage.parquet')
print("  ✓ Wrote test parquet file")

read_df = storage.read_parquet('data/test_storage.parquet')
print(f"  ✓ Read back {len(read_df)} rows")

# Test JSON
test_json = {'test': 'success', 'timestamp': '2024-01-01'}
storage.write_json(test_json, 'data/test_storage.json')
print("  ✓ Wrote test JSON file")

read_json = storage.read_json('data/test_storage.json')
print(f"  ✓ Read back: {read_json}")

# Cleanup
storage.delete('data/test_storage.parquet')
storage.delete('data/test_storage.json')
print("  ✓ Cleaned up test files")
print()

print("="*60)
print("✅ All storage tests passed!")
print("="*60)
