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
    data = storage.read_json('data/data_status.json')
    
    # Write files
    storage.write_parquet(df, 'data/master_dataset.parquet')
    storage.write_json(data, 'data/data_status.json')
    
    # Check existence
    if storage.exists('data/master_dataset.parquet'):
        ...
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional
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
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET', 'volatility-forecast-data')
        
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
            path: File path (e.g., 'data/data_status.json')
            
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
            path: File path (e.g., 'data/data_status.json')
        """
        if self.is_aws:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._get_s3_key(path),
                Body=json.dumps(data, indent=2).encode('utf-8')
            )
        else:
            local_path = self._get_local_path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def read_pickle(self, path: str) -> Any:
        """
        Read pickle file.
        
        Args:
            path: File path (e.g., 'models/latest_ensemble.pkl')
            
        Returns:
            Unpickled object
        """
        import pickle
        
        if self.is_aws:
            key = self._get_s3_key(path)
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return pickle.loads(obj['Body'].read())
        else:
            with open(self._get_local_path(path), 'rb') as f:
                return pickle.load(f)
    
    def write_pickle(self, obj: Any, path: str) -> None:
        """
        Write pickle file.
        
        Args:
            obj: Object to pickle
            path: File path (e.g., 'models/latest_ensemble.pkl')
        """
        import pickle
        
        if self.is_aws:
            buffer = BytesIO()
            pickle.dump(obj, buffer)
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._get_s3_key(path),
                Body=buffer.getvalue()
            )
        else:
            local_path = self._get_local_path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                pickle.dump(obj, f)
    
    def list_files(self, prefix: str) -> list:
        """
        List files with given prefix.
        
        Args:
            prefix: Path prefix (e.g., 'data/features.L1/')
            
        Returns:
            List of file paths
        """
        if self.is_aws:
            prefix = self._get_s3_key(prefix)
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
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
    
    def read_text(self, path: str) -> str:
        """
        Read text file.
        
        Args:
            path: File path
            
        Returns:
            File contents as string
        """
        if self.is_aws:
            key = self._get_s3_key(path)
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return obj['Body'].read().decode('utf-8')
        else:
            with open(self._get_local_path(path), 'r') as f:
                return f.read()
    
    def write_text(self, content: str, path: str) -> None:
        """
        Write text file.
        
        Args:
            content: Text content to write
            path: File path
        """
        if self.is_aws:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._get_s3_key(path),
                Body=content.encode('utf-8')
            )
        else:
            local_path = self._get_local_path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'w') as f:
                f.write(content)
    
    def append_jsonl(self, data: Dict[str, Any], path: str) -> None:
        """
        Append JSON line to JSONL file.
        
        Args:
            data: Dictionary to append
            path: File path (e.g., 'models/training_history.jsonl')
        """
        line = json.dumps(data) + '\n'
        
        if self.is_aws:
            # Read existing content
            try:
                existing = self.read_text(path)
            except:
                existing = ''
            
            # Append new line
            new_content = existing + line
            self.write_text(new_content, path)
        else:
            local_path = self._get_local_path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'a') as f:
                f.write(line)


# Convenience function for getting storage instance
_storage_instance = None

def get_storage() -> Storage:
    """Get singleton storage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = Storage()
    return _storage_instance
