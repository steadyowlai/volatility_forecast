"""
Quick scan: Check all feature partitions for basic issues.
"""
import pandas as pd
from pathlib import Path


def scan_all_partitions():
    """Scan all feature partitions and report any issues."""
    features_path = Path("data/features.L1")
    
    if not features_path.exists():
        print("❌ No features directory found")
        return False
    
    partitions = sorted(features_path.glob("date=*/features.parquet"))
    
    if not partitions:
        print("❌ No feature partitions found")
        return False
    
    print(f"Scanning {len(partitions)} feature partitions...")
    print("=" * 70)
    
    valid_count = 0
    bad_partitions = []
    
    for partition_file in partitions:
        date_str = partition_file.parent.name.replace("date=", "")
        
        try:
            df = pd.read_parquet(partition_file)
            
            # Check 1: Must be 1 row
            if len(df) != 1:
                bad_partitions.append((date_str, f"Has {len(df)} rows (expected 1)"))
                continue
            
            # Check 2: Must be 21 columns
            if len(df.columns) != 21:
                bad_partitions.append((date_str, f"Has {len(df.columns)} columns (expected 21)"))
                continue
            
            # Check 3: Must NOT have 'symbol' column (raw curated data)
            if 'symbol' in df.columns:
                bad_partitions.append((date_str, "Contains raw curated data (has 'symbol' column)"))
                continue
            
            valid_count += 1
            
        except Exception as e:
            bad_partitions.append((date_str, f"Error: {str(e)[:50]}"))
    
    print("=" * 70)
    print(f"\n✅ Valid partitions: {valid_count}")
    print(f"❌ Bad partitions: {len(bad_partitions)}")
    
    if bad_partitions:
        print(f"\nBad partitions (first 10):")
        for date, reason in bad_partitions[:10]:
            print(f"  - {date}: {reason}")
        if len(bad_partitions) > 10:
            print(f"  ... and {len(bad_partitions) - 10} more")
        
        print(f"\n⚠️  Run this to delete bad partitions:")
        print(f"     docker-compose run --rm features python3 << 'EOF'")
        print(f"import shutil")
        print(f"from pathlib import Path")
        for date, _ in bad_partitions:
            print(f"shutil.rmtree(Path('data/features.L1/date={date}'))")
        print(f"EOF")
        return False
    else:
        print(f"\n✅ All partitions are valid!")
        return True


if __name__ == "__main__":
    success = scan_all_partitions()
    exit(0 if success else 1)
