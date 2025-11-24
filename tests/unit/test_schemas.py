"""
unit tests for libs/schemas.py

test pandera schema validation with valid and invalid data
"""

import pytest
import pandas as pd
import numpy as np
import pandera as pa
from pathlib import Path
import sys

#add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from libs.schemas import raw_market_schema, curated_market_daily_schema


class TestRawMarketSchema:
    """tests for raw_market_schema validation"""
    
    def test_raw_market_schema_valid_data(self):
        """valid data should pass raw_market_schema"""
        df = pd.DataFrame({
            'symbol': ['SPY', 'SPY'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000000.0, 1100000.0]
        })
        
        #should not raise
        validated = raw_market_schema.validate(df)
        assert len(validated) == 2
    
    def test_raw_market_schema_rejects_missing_column(self):
        """missing required column should be rejected"""
        df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': pd.to_datetime(['2024-01-01']),
            'open': [100.0],
            'high': [102.0],
            'low': [99.0],
            #missing close column
            'volume': [1000000.0]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            raw_market_schema.validate(df)
    
    def test_raw_market_schema_rejects_wrong_type(self):
        """wrong data types should be rejected"""
        df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': pd.to_datetime(['2024-01-01']),
            'open': ['not_a_number'],  #should be float
            'high': [102.0],
            'low': [99.0],
            'close': [101.0],
            'volume': [1000000.0]
        })
        
        with pytest.raises((pa.errors.SchemaError, ValueError)):
            raw_market_schema.validate(df)
    
    def test_raw_market_schema_multiple_symbols(self):
        """validation with multiple symbols"""
        df = pd.DataFrame({
            'symbol': ['SPY', '^VIX', 'TLT'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-01']),
            'open': [100.0, 20.0, 90.0],
            'high': [102.0, 22.0, 92.0],
            'low': [99.0, 19.0, 89.0],
            'close': [101.0, 21.0, 91.0],
            'volume': [1000000.0, 0.0, 500000.0]
        })
        
        #should pass
        validated = raw_market_schema.validate(df)
        assert len(validated) == 3
    
    def test_raw_market_schema_accepts_zero_volume(self):
        """zero volume for VIX should be accepted"""
        df = pd.DataFrame({
            'symbol': ['^VIX'],
            'date': pd.to_datetime(['2024-01-01']),
            'open': [20.0],
            'high': [22.0],
            'low': [19.0],
            'close': [21.0],
            'volume': [0.0]  #VIX has zero volume
        })
        
        #should pass
        validated = raw_market_schema.validate(df)
        assert validated['volume'].iloc[0] == 0.0


class TestCuratedMarketDailySchema:
    """tests for curated_market_daily_schema validation"""
    
    def test_curated_schema_valid_data(self):
        """valid data should pass curated_market_daily_schema"""
        df = pd.DataFrame({
            'symbol': ['SPY', 'SPY'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'close': [100.0, 102.0],
            'ret': [np.nan, 0.02],  #first return is NaN
            'adj_close': [100.0, 102.0]
        })
        
        #should not raise
        validated = curated_market_daily_schema.validate(df)
        assert len(validated) == 2
    
    def test_curated_schema_allows_null_ret(self):
        """first return can be null"""
        df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': pd.to_datetime(['2024-01-01']),
            'close': [100.0],
            'ret': [np.nan],  #null for first day
            'adj_close': [100.0]
        })
        
        #should not raise
        validated = curated_market_daily_schema.validate(df)
        assert pd.isna(validated['ret'].iloc[0])
    
    def test_curated_schema_rejects_missing_column(self):
        """missing required column should be rejected"""
        df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': pd.to_datetime(['2024-01-01']),
            'close': [100.0],
            #missing ret column
            'adj_close': [100.0]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            curated_market_daily_schema.validate(df)
    
    def test_curated_schema_multiple_symbols(self):
        """validation with multiple symbols"""
        df = pd.DataFrame({
            'symbol': ['SPY', 'SPY', '^VIX', '^VIX'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02']),
            'close': [100.0, 102.0, 20.0, 21.0],
            'ret': [np.nan, 0.02, np.nan, 0.05],
            'adj_close': [100.0, 102.0, 20.0, 21.0]
        })
        
        #should pass
        validated = curated_market_daily_schema.validate(df)
        assert len(validated) == 4
    
    def test_curated_schema_rejects_wrong_type(self):
        """wrong data types should be rejected"""
        df = pd.DataFrame({
            'symbol': [123],  #should be string
            'date': pd.to_datetime(['2024-01-01']),
            'close': [100.0],
            'ret': [np.nan],
            'adj_close': [100.0]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            curated_market_daily_schema.validate(df)
    
    def test_curated_schema_validates_datetime(self):
        """date column must be datetime"""
        df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': ['2024-01-01'],  #string not datetime
            'close': [100.0],
            'ret': [np.nan],
            'adj_close': [100.0]
        })
        
        with pytest.raises(pa.errors.SchemaError):
            curated_market_daily_schema.validate(df)


class TestSchemaEdgeCases:
    """edge cases for both schemas"""
    
    def test_empty_dataframe_raw(self):
        """empty dataframe should be handled"""
        df = pd.DataFrame({
            'symbol': [],
            'date': pd.to_datetime([]),
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        })
        
        #empty dataframes should pass schema validation
        validated = raw_market_schema.validate(df)
        assert len(validated) == 0
    
    def test_empty_dataframe_curated(self):
        """empty dataframe should be handled"""
        df = pd.DataFrame({
            'symbol': [],
            'date': pd.to_datetime([]),
            'close': [],
            'ret': [],
            'adj_close': []
        })
        
        #empty dataframes should pass schema validation
        validated = curated_market_daily_schema.validate(df)
        assert len(validated) == 0
    
    def test_large_volume_values(self):
        """very large volume values should be accepted"""
        df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': pd.to_datetime(['2024-01-01']),
            'open': [100.0],
            'high': [102.0],
            'low': [99.0],
            'close': [101.0],
            'volume': [999_999_999_999.0]  #very large volume
        })
        
        #should pass
        validated = raw_market_schema.validate(df)
        assert validated['volume'].iloc[0] == 999_999_999_999.0
    
    def test_negative_returns_allowed(self):
        """negative returns should be allowed in curated schema"""
        df = pd.DataFrame({
            'symbol': ['SPY', 'SPY'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'close': [100.0, 95.0],
            'ret': [np.nan, -0.05],  #negative return
            'adj_close': [100.0, 95.0]
        })
        
        #should pass
        validated = curated_market_daily_schema.validate(df)
        assert validated['ret'].iloc[1] == pytest.approx(-0.05)
