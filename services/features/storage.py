"""
Storage abstraction layer for local and S3 storage.

This module provides a unified interface for file operations that works
with both local filesystem (development) and S3 (production).

Environment detection:
- Local: When AWS_EXECUTION_ENV is not set
- AWS Lambda: When AWS_EXECUTION_ENV is set

Usage:
    from storage import Storage
    
    storage = Storage()
    
    # Read files
    df = storage.read_parquet('data/master_dataset.parquet')
    data = storage.read_json('data/curated.market/_manifest.json')
    
    # Write files
    storage.write_parquet(df, 'data/curated.market/date=2010-01-04/data.parquet')
    storage.write_json(data, 'data/curated.market/_manifest.json')
    
    # List partitions
    dates = storage.list_partitions('data/curated.market')
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, List
from io import BytesIO


class Storage:
    """Unified storage interface for local and S3."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize storage.
        
        Args:
            bucket_name: S3 bucket name (only used in AWS environment)
        """
        self.is_aws = os.getenv('AWS_EXECUTION_ENV') is not None
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET', 'volatility-forecast')
        
        if self.is_aws:
            import boto3
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None
            self.local_root = Path('/app')  # Docker path
    
    def _get_local_path(self, path: str) -> Path:
        """Convert path to local filesystem path."""
        # Remove leading slash if present
        path = path.lstrip('/')
        return self.local_root / path
    
    def _get_s3_key(self, path: str) -> str:
        """Convert path to S3 key."""
        # Remove leading slash if present
        return path.lstrip('/')
    
    def exists(self, path: str) -> bool:
        """
        Check if file exists.
        
        Args:
            path: File path (e.g., 'data/master_dataset.parquet')
            
        Returns:
            True if file exists, False otherwise
        """
        if self.is_aws:
            try:
                self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=self._get_s3_key(path)
                )
                return True
            except:
                return False
        else:
            return self._get_local_path(path).exists()
    
    def read_parquet(self, path: str) -> pd.DataFrame:
        """
        Read parquet file.
        
        Args:
            path: File path (e.g., 'data/master_dataset.parquet')
            
        Returns:
            DataFrame
        """
        if self.is_aws:
            key = self._get_s3_key(path)
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return pd.read_parquet(BytesIO(obj['Body'].read()))
        else:
            return pd.read_parquet(self._get_local_path(path))
    
    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        """
        Write parquet file.
        
        Args:
            df: DataFrame to write
            path: File path (e.g., 'data/master_dataset.parquet')
        """
        if self.is_aws:
            buffer = BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._get_s3_key(path),
                Body=buffer.getvalue()
            )
        else:
            local_path = self._get_local_path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(local_path, index=False)
    
    def read_json(self, path: str) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Args:
            path: File path (e.g., 'data/curated.market/_manifest.json')
            
        Returns:
            Dictionary
        """
        if self.is_aws:
            key = self._get_s3_key(path)
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return json.loads(obj['Body'].read().decode('utf-8'))
        else:
            with open(self._get_local_path(path), 'r') as f:
                return json.load(f)
    
    def write_json(self, data: Dict[str, Any], path: str) -> None:
        """
        Write JSON file.
        
        Args:
            data: Dictionary to write
            path: File path (e.g., 'data/curated.market/_manifest.json')
        """
        if self.is_aws:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._get_s3_key(path),
                Body=json.dumps(data, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
        else:
            local_path = self._get_local_path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def list_partitions(self, prefix: str) -> List[str]:
        """
        List partition folders (e.g., date=2010-01-04/).
        
        Args:
            prefix: Path prefix (e.g., 'data/curated.market')
            
        Returns:
            List of date strings (e.g., ['2010-01-04', '2010-01-05'])
        """
        if self.is_aws:
            # List objects with prefix and delimiter to get "folders"
            # Uses pagination to handle > 1000 partitions
            prefix_key = self._get_s3_key(prefix)
            if not prefix_key.endswith('/'):
                prefix_key += '/'

            dates = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix_key,
                Delimiter='/'
            )
            for page in pages:
                for common_prefix in page.get('CommonPrefixes', []):
                    folder_name = common_prefix['Prefix'].rstrip('/').split('/')[-1]
                    if folder_name.startswith('date='):
                        date_str = folder_name.replace('date=', '')
                        dates.append(date_str)

            return sorted(dates)
        else:
            # Local filesystem
            local_path = self._get_local_path(prefix)
            if not local_path.exists():
                return []
            
            dates = []
            for folder in local_path.iterdir():
                if folder.is_dir() and folder.name.startswith('date='):
                    date_str = folder.name.replace('date=', '')
                    dates.append(date_str)
            
            return sorted(dates)
    
    def list_files(self, prefix: str) -> List[str]:
        """
        List files with given prefix.
        
        Args:
            prefix: Path prefix (e.g., 'data/features.L1/')
            
        Returns:
            List of file paths
        """
        if self.is_aws:
            prefix_key = self._get_s3_key(prefix)
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix_key
            )
            if 'Contents' not in response:
                return []
            return [obj['Key'] for obj in response['Contents']]
        else:
            local_path = self._get_local_path(prefix)
            if not local_path.exists():
                return []
            # Return paths relative to local_root
            return [str(p.relative_to(self.local_root)) 
                    for p in local_path.rglob('*') if p.is_file()]
    
    def delete(self, path: str) -> None:
        """
        Delete file.
        
        Args:
            path: File path to delete
        """
        if self.is_aws:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=self._get_s3_key(path)
            )
        else:
            local_path = self._get_local_path(path)
            if local_path.exists():
                local_path.unlink()


# Convenience function for getting storage instance
_storage_instance = None

def get_storage() -> Storage:
    """Get singleton storage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = Storage()
    return _storage_instance
