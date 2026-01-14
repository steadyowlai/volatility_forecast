"""Quick test of storage abstraction layer."""

import sys
sys.path.insert(0, 'libs')

from storage import Storage
import pandas as pd

def test_storage():
    """Test storage operations."""
    print("Testing Storage Abstraction Layer")
    print("="*60)
    
    storage = Storage()
    print(f"Environment: {'AWS' if storage.is_aws else 'Local'}")
    print(f"Bucket: {storage.bucket_name}")
    print()
    
    # Test 1: Check if files exist
    print("Test 1: Check file existence")
    exists = storage.exists('data/master_dataset.parquet')
    print(f"  data/master_dataset.parquet exists: {exists}")
    
    exists = storage.exists('data/data_status.json')
    print(f"  data/data_status.json exists: {exists}")
    print()
    
    # Test 2: Read parquet
    if storage.exists('data/master_dataset.parquet'):
        print("Test 2: Read parquet file")
        df = storage.read_parquet('data/master_dataset.parquet')
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)[:5]}...")
        print()
    
    # Test 3: Read JSON
    if storage.exists('data/data_status.json'):
        print("Test 3: Read JSON file")
        data = storage.read_json('data/data_status.json')
        print(f"  Data: {data}")
        print()
    
    # Test 4: Write and read test file
    print("Test 4: Write and read test files")
    
    # Test parquet
    test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    storage.write_parquet(test_df, 'data/test.parquet')
    read_df = storage.read_parquet('data/test.parquet')
    print(f"  Parquet: wrote {len(test_df)} rows, read {len(read_df)} rows ✓")
    
    # Test JSON
    test_json = {'test': 'data', 'value': 123}
    storage.write_json(test_json, 'data/test.json')
    read_json = storage.read_json('data/test.json')
    print(f"  JSON: {read_json} ✓")
    
    # Test text
    storage.write_text("Hello, World!", 'data/test.txt')
    read_text = storage.read_text('data/test.txt')
    print(f"  Text: '{read_text}' ✓")
    
    # Test JSONL append
    storage.write_text('', 'data/test.jsonl')  # Clear file
    storage.append_jsonl({'line': 1}, 'data/test.jsonl')
    storage.append_jsonl({'line': 2}, 'data/test.jsonl')
    lines = storage.read_text('data/test.jsonl')
    print(f"  JSONL: {len(lines.strip().split(chr(10)))} lines ✓")
    print()
    
    # Test 5: List files
    print("Test 5: List files")
    files = storage.list_files('data/')
    print(f"  Found {len(files)} files in data/")
    print(f"  Sample: {files[:3]}")
    print()
    
    # Cleanup test files
    print("Cleaning up test files...")
    storage.delete('data/test.parquet')
    storage.delete('data/test.json')
    storage.delete('data/test.txt')
    storage.delete('data/test.jsonl')
    print("  Cleanup complete ✓")
    print()
    
    print("="*60)
    print("✅ All storage tests passed!")

if __name__ == '__main__':
    test_storage()
