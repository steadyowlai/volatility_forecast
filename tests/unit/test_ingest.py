"""
Unit tests for services/ingest/app.py

Tests all functions in the ingest service with mocking for external dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path so we can import services
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.ingest.app import (
    download_one,
    build_raw,
    build_curated,
    write_partitions
)
from libs.schemas import raw_market_schema, curated_market_daily_schema


class TestDownloadOne:
    """Tests for download_one() function"""
    
    def test_download_one_success(self, mocker, mock_yfinance_response):
        """Test successful download of a single ticker"""
        # Mock yf.download to avoid hitting real API
        mocker.patch('yfinance.download', return_value=mock_yfinance_response)
        
        result = download_one('SPY', period='1mo')
        
        # Check basic structure
        assert len(result) == 10
        assert 'symbol' in result.columns
        assert 'date' in result.columns
        assert 'adj close' in result.columns
        
        # Check symbol is correct
        assert (result['symbol'] == 'SPY').all()
        
        # Check expected columns exist
        expected_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
        assert set(result.columns) == set(expected_cols)
    
    def test_download_one_empty_response(self, mocker):
        """Test handling of empty response from yfinance"""
        mocker.patch('yfinance.download', return_value=pd.DataFrame())
        
        with pytest.raises(RuntimeError, match="No data downloaded"):
            download_one('INVALID', period='1mo')
    
    def test_download_one_vix_no_adj_close(self, mocker, mock_yfinance_vix_response):
        """Test handling of tickers without adj close (like VIX)"""
        mocker.patch('yfinance.download', return_value=mock_yfinance_vix_response)
        
        result = download_one('^VIX', period='1mo')
        
        # Should have adj close column
        assert 'adj close' in result.columns
        
        # Adj close should equal close for VIX
        assert (result['adj close'] == result['close']).all()
        
        # Symbol should be ^VIX
        assert (result['symbol'] == '^VIX').all()
    
    def test_download_one_flattens_multiindex(self, mocker, mock_yfinance_response):
        """Test that MultiIndex columns are flattened correctly"""
        mocker.patch('yfinance.download', return_value=mock_yfinance_response)
        
        result = download_one('SPY', period='1mo')
        
        # Columns should be flat (not MultiIndex)
        assert not isinstance(result.columns, pd.MultiIndex)
        
        # Column names should be lowercase strings
        assert all(isinstance(col, str) for col in result.columns)
        assert all(col.islower() for col in result.columns)


class TestBuildRaw:
    """Tests for build_raw() function"""
    
    def test_build_raw_schema_compliance(self):
        """Test that build_raw produces schema-compliant data"""
        input_df = pd.DataFrame({
            'symbol': ['SPY', 'SPY'],
            'date': ['2024-01-01', '2024-01-02'],
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'adj close': [101.0, 102.0],
            'volume': [1000000, 1100000]
        })
        
        result = build_raw(input_df)
        
        # Should match raw_market_schema columns
        expected_cols = {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'}
        assert set(result.columns) == expected_cols
        
        # Volume should be float
        assert result['volume'].dtype == float
        
        # Date should be datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        
        # Should pass schema validation
        raw_market_schema.validate(result)
    
    def test_build_raw_preserves_data_integrity(self):
        """Test that no data is lost in transformation"""
        input_df = pd.DataFrame({
            'symbol': ['SPY', '^VIX', 'TLT'],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'open': [100.0, 20.0, 90.0],
            'high': [102.0, 22.0, 92.0],
            'low': [99.0, 19.0, 89.0],
            'close': [101.0, 21.0, 91.0],
            'adj close': [101.0, 21.0, 91.0],
            'volume': [1000000, 0, 500000]
        })
        
        result = build_raw(input_df)
        
        # No rows should be lost
        assert len(result) == len(input_df)
        
        # Symbols should be preserved
        assert result['symbol'].tolist() == ['SPY', '^VIX', 'TLT']
        
        # Close prices should be preserved
        assert result['close'].tolist() == [101.0, 21.0, 91.0]
    
    def test_build_raw_multiple_dates(self):
        """Test build_raw with multiple dates"""
        input_df = pd.DataFrame({
            'symbol': ['SPY'] * 5,
            'date': pd.date_range('2024-01-01', periods=5),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'adj close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000000] * 5
        })
        
        result = build_raw(input_df)
        
        assert len(result) == 5
        assert pd.api.types.is_datetime64_any_dtype(result['date'])


class TestBuildCurated:
    """Tests for build_curated() function"""
    
    def test_build_curated_computes_returns(self):
        """Test that returns are computed correctly"""
        input_df = pd.DataFrame({
            'symbol': ['SPY', 'SPY', 'SPY'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'adj close': [101.0, 102.0, 103.0],
            'volume': [1000000, 1100000, 1200000]
        })
        
        result = build_curated(input_df)
        
        # First return should be NaN
        assert pd.isna(result['ret'].iloc[0])
        
        # Second return: (102-101)/101 ≈ 0.0099
        assert result['ret'].iloc[1] == pytest.approx(0.0099, rel=1e-3)
        
        # Third return: (103-102)/102 ≈ 0.0098
        assert result['ret'].iloc[2] == pytest.approx(0.0098, rel=1e-3)
    
    def test_build_curated_per_symbol_returns(self):
        """Test that returns are computed per symbol (not across symbols)"""
        input_df = pd.DataFrame({
            'symbol': ['SPY', 'SPY', '^VIX', '^VIX'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
            'open': [100.0, 101.0, 20.0, 21.0],
            'high': [102.0, 103.0, 22.0, 23.0],
            'low': [99.0, 100.0, 19.0, 20.0],
            'close': [101.0, 102.0, 21.0, 22.0],
            'adj close': [101.0, 102.0, 21.0, 22.0],
            'volume': [1000000, 1100000, 0, 0]
        })
        
        result = build_curated(input_df)
        
        # First return for each symbol should be NaN
        spy_data = result[result['symbol'] == 'SPY'].sort_values('date')
        vix_data = result[result['symbol'] == '^VIX'].sort_values('date')
        
        assert pd.isna(spy_data['ret'].iloc[0])
        assert pd.isna(vix_data['ret'].iloc[0])
        
        # Second returns should be computed correctly per symbol
        assert not pd.isna(spy_data['ret'].iloc[1])
        assert not pd.isna(vix_data['ret'].iloc[1])
    
    def test_build_curated_schema_compliance(self):
        """Test that build_curated produces schema-compliant data"""
        input_df = pd.DataFrame({
            'symbol': ['SPY', 'SPY'],
            'date': ['2024-01-01', '2024-01-02'],
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'adj close': [101.0, 102.0],
            'volume': [1000000, 1100000]
        })
        
        result = build_curated(input_df)
        
        # Should pass schema validation
        curated_market_daily_schema.validate(result)
        
        # Check columns
        expected_cols = {'symbol', 'date', 'close', 'ret', 'adj_close'}
        assert set(result.columns) == expected_cols
    
    def test_build_curated_sorted_by_symbol_date(self):
        """Test that output is sorted by symbol, date"""
        input_df = pd.DataFrame({
            'symbol': ['^VIX', 'SPY', '^VIX', 'SPY'],
            'date': ['2024-01-02', '2024-01-02', '2024-01-01', '2024-01-01'],
            'open': [21.0, 101.0, 20.0, 100.0],
            'high': [23.0, 103.0, 22.0, 102.0],
            'low': [20.0, 100.0, 19.0, 99.0],
            'close': [22.0, 102.0, 21.0, 101.0],
            'adj close': [22.0, 102.0, 21.0, 101.0],
            'volume': [0, 1100000, 0, 1000000]
        })
        
        result = build_curated(input_df)
        
        # Should be sorted by symbol, then date
        assert result['symbol'].iloc[0] == 'SPY'
        assert result['date'].iloc[0] == pd.Timestamp('2024-01-01')


class TestWritePartitions:
    """Tests for write_partitions() function"""
    
    def test_write_partitions_creates_directories(self, tmp_path):
        """Test that partitions are created correctly"""
        test_df = pd.DataFrame({
            'symbol': ['SPY', 'SPY'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'close': [100.0, 101.0]
        })
        
        write_partitions(test_df, tmp_path, 'test')
        
        # Check directories exist
        assert (tmp_path / 'date=2024-01-01').exists()
        assert (tmp_path / 'date=2024-01-02').exists()
        
        # Check files exist
        assert (tmp_path / 'date=2024-01-01' / 'test.parquet').exists()
        assert (tmp_path / 'date=2024-01-02' / 'test.parquet').exists()
    
    def test_write_partitions_correct_data(self, tmp_path):
        """Test that data is correctly partitioned by date"""
        test_df = pd.DataFrame({
            'symbol': ['SPY', 'SPY', '^VIX'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01']),
            'close': [100.0, 101.0, 20.0]
        })
        
        write_partitions(test_df, tmp_path, 'test')
        
        # Read back and verify
        jan1_data = pd.read_parquet(tmp_path / 'date=2024-01-01' / 'test.parquet')
        jan2_data = pd.read_parquet(tmp_path / 'date=2024-01-02' / 'test.parquet')
        
        # Jan 1 should have SPY and VIX
        assert len(jan1_data) == 2
        assert set(jan1_data['symbol']) == {'SPY', '^VIX'}
        
        # Jan 2 should have only SPY
        assert len(jan2_data) == 1
        assert jan2_data['symbol'].iloc[0] == 'SPY'
    
    def test_write_partitions_preserves_data(self, tmp_path):
        """Test that data values are preserved in partitions"""
        test_df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': pd.to_datetime(['2024-01-01']),
            'close': [123.45],
            'volume': [9876543.0]
        })
        
        write_partitions(test_df, tmp_path, 'test')
        
        # Read back
        data = pd.read_parquet(tmp_path / 'date=2024-01-01' / 'test.parquet')
        
        # Values should match
        assert data['close'].iloc[0] == pytest.approx(123.45)
        assert data['volume'].iloc[0] == pytest.approx(9876543.0)
    
    def test_write_partitions_handles_multiple_dates(self, tmp_path):
        """Test writing partitions for many dates"""
        dates = pd.date_range('2024-01-01', periods=30)
        test_df = pd.DataFrame({
            'symbol': ['SPY'] * 30,
            'date': dates,
            'close': range(100, 130)
        })
        
        write_partitions(test_df, tmp_path, 'test')
        
        # Check that 30 directories were created
        partition_dirs = list(tmp_path.glob('date=*'))
        assert len(partition_dirs) == 30
