"""
unit tests for services/features/app.py

Tests all feature engineering functions with various edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.features.app import (
    compute_spy_returns,
    compute_spy_volatility,
    compute_drawdown,
    compute_vix_features,
    compute_rsi,
    compute_correlations,
    compute_spreads,
    build_features
)


class TestComputeSpyReturns:
    """Tests for compute_spy_returns() function"""
    
    def test_compute_spy_returns_basic(self):
        """Test basic return computation"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': pd.date_range('2024-01-01', periods=10),
            'adj_close': [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0],
            'ret': [np.nan, 0.02, -0.0098, 0.0198, 0.0194, -0.0095, 0.0192, 0.0189, -0.0093, 0.0187]
        })
        
        result = compute_spy_returns(df)
        
        # Check that spy_ret_1d exists and has values
        assert 'spy_ret_1d' in result.columns
        assert not result['spy_ret_1d'].isna().all()
        
        # Check that we have the same number of rows
        assert len(result) == len(df)
    
    def test_compute_spy_returns_multiple_windows(self):
        """Test that all windows are computed"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'adj_close': 100 + np.cumsum(np.random.randn(100)),
            'ret': np.random.randn(100) * 0.01
        })
        
        result = compute_spy_returns(df)
        
        # Check all return columns exist
        for window in [1, 5, 10, 20, 60]:
            assert f'spy_ret_{window}d' in result.columns
    
    def test_compute_spy_returns_insufficient_data(self):
        """Test behavior with insufficient data for windows"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': pd.date_range('2024-01-01', periods=10),
            'adj_close': np.linspace(100, 105, 10),
            'ret': np.random.randn(10) * 0.01
        })
        
        result = compute_spy_returns(df)
        
        # 60-day returns should be all NaN with only 10 days
        assert result['spy_ret_60d'].isna().all()
        
        # 1-day returns should have only first NaN
        assert result['spy_ret_1d'].isna().sum() == 1
    
    def test_compute_spy_returns_filters_spy_only(self):
        """Test that only SPY data is used"""
        df = pd.DataFrame({
            'symbol': ['SPY', 'SPY', '^VIX', '^VIX'],
            'date': pd.date_range('2024-01-01', periods=2).tolist() * 2,
            'adj_close': [100.0, 102.0, 20.0, 21.0],
            'ret': [np.nan, 0.02, np.nan, 0.05]
        })
        
        result = compute_spy_returns(df)
        
        # Should only return SPY rows
        assert len(result) == 2
        
        # Date column should exist
        assert 'date' in result.columns
    
    def test_compute_spy_returns_log_returns(self):
        """Test that log returns are computed correctly"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 3,
            'date': pd.date_range('2024-01-01', periods=3),
            'adj_close': [100.0, 110.0, 121.0],
            'ret': [np.nan, 0.1, 0.1]
        })
        
        result = compute_spy_returns(df)
        
        # 1-day log return from 100 to 110: log(110/100) ≈ 0.0953
        assert result['spy_ret_1d'].iloc[1] == pytest.approx(np.log(110/100), rel=1e-3)


class TestComputeSpyVolatility:
    """Tests for compute_spy_volatility() function"""
    
    def test_compute_spy_volatility_formula(self):
        """Test that volatility uses sqrt(sum(r^2)) formula"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': pd.date_range('2024-01-01', periods=10),
            'adj_close': [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0],
            'ret': [np.nan, 0.02, -0.01, 0.02, 0.019, -0.01, 0.019, 0.019, -0.009, 0.019]
        })
        
        result = compute_spy_volatility(df)
        
        # Check 5-day vol column exists
        assert 'spy_vol_5d' in result.columns
        
        # Manually compute 5-day vol for row 5 (index 5)
        returns = df['ret'].iloc[1:6].values  # First 5 returns (skip NaN)
        expected_vol = np.sqrt(np.sum(returns**2))
        
        # Compare (allowing for numerical precision)
        assert result['spy_vol_5d'].iloc[5] == pytest.approx(expected_vol, rel=1e-2)
    
    def test_compute_spy_volatility_all_windows(self):
        """Test that all volatility windows are computed"""
        np.random.seed(42)
        df = pd.DataFrame({
            'symbol': ['SPY'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'adj_close': 100 + np.cumsum(np.random.randn(100)),
            'ret': np.concatenate([[np.nan], np.random.randn(99) * 0.01])
        })
        
        result = compute_spy_volatility(df)
        
        # Check all volatility windows exist
        for window in [5, 10, 20, 60]:
            assert f'spy_vol_{window}d' in result.columns
    
    def test_compute_spy_volatility_positive_values(self):
        """Test that volatility is always positive"""
        np.random.seed(42)
        df = pd.DataFrame({
            'symbol': ['SPY'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'adj_close': 100 + np.cumsum(np.random.randn(100)),
            'ret': np.concatenate([[np.nan], np.random.randn(99) * 0.01])
        })
        
        result = compute_spy_volatility(df)
        
        # All non-NaN volatility values should be >= 0
        for window in [5, 10, 20, 60]:
            col = f'spy_vol_{window}d'
            valid_values = result[col].dropna()
            assert (valid_values >= 0).all()


class TestComputeDrawdown:
    """Tests for compute_drawdown() function"""
    
    def test_compute_drawdown_no_drawdown(self):
        """Test drawdown is 0 when price only goes up"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 70,
            'date': pd.date_range('2024-01-01', periods=70),
            'adj_close': np.linspace(100, 110, 70)  # Monotonically increasing
        })
        
        result = compute_drawdown(df)
        
        # Drawdown should be 0 or very close to 0
        assert result['drawdown_60d'].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    
    def test_compute_drawdown_max_drawdown(self):
        """Test drawdown calculation with actual drop"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 70,
            'date': pd.date_range('2024-01-01', periods=70),
            'adj_close': [100]*30 + [90]*40  # Drop from 100 to 90
        })
        
        result = compute_drawdown(df)
        
        # After the drop, drawdown should be 0.1 (10% down from peak)
        assert result['drawdown_60d'].iloc[-1] == pytest.approx(0.1, rel=1e-3)
    
    def test_compute_drawdown_recovery(self):
        """Test that drawdown goes back to 0 after recovery"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 80,
            'date': pd.date_range('2024-01-01', periods=80),
            'adj_close': [100]*20 + [90]*20 + [100]*40  # Drop and recover
        })
        
        result = compute_drawdown(df)
        
        # At the end (after recovery), drawdown should be 0
        assert result['drawdown_60d'].iloc[-1] == pytest.approx(0.0, abs=1e-6)


class TestComputeVixFeatures:
    """Tests for compute_vix_features() function"""
    
    def test_compute_vix_features_basic(self):
        """Test VIX feature extraction"""
        df = pd.DataFrame({
            'symbol': ['^VIX', '^VIX', '^VIX3M', '^VIX3M'],
            'date': pd.date_range('2024-01-01', periods=2).tolist() * 2,
            'adj_close': [20.0, 21.0, 22.0, 23.0]
        })
        
        result = compute_vix_features(df)
        
        # Check columns exist
        assert 'vix' in result.columns
        assert 'vix3m' in result.columns
        assert 'vix_term' in result.columns
        
        # Check values
        assert len(result) == 2
    
    def test_compute_vix_term_structure_contango(self):
        """Test VIX term structure calculation (vix3m / vix) in contango"""
        df = pd.DataFrame({
            'symbol': ['^VIX', '^VIX3M'],
            'date': ['2024-01-01'] * 2,
            'adj_close': [20.0, 24.0]
        })
        
        result = compute_vix_features(df)
        
        # vix_term = 24/20 = 1.2 (contango)
        assert result['vix_term'].iloc[0] == pytest.approx(1.2)
    
    def test_compute_vix_term_structure_backwardation(self):
        """Test VIX term structure in backwardation"""
        df = pd.DataFrame({
            'symbol': ['^VIX', '^VIX3M'],
            'date': ['2024-01-01'] * 2,
            'adj_close': [25.0, 20.0]
        })
        
        result = compute_vix_features(df)
        
        # vix_term = 20/25 = 0.8 (backwardation)
        assert result['vix_term'].iloc[0] == pytest.approx(0.8)


class TestComputeRsi:
    """Tests for compute_rsi() function"""
    
    def test_compute_rsi_overbought(self):
        """Test RSI approaches 100 when all prices increase"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 20,
            'date': pd.date_range('2024-01-01', periods=20),
            'adj_close': np.linspace(100, 120, 20)  #all gains
        })
        
        result = compute_rsi(df, window=14)
        
        # RSI should be close to 100 (fully overbought)
        assert result['rsi_spy_14'].iloc[-1] > 95
    
    def test_compute_rsi_oversold(self):
        """Test RSI approaches 0 when all prices decrease"""
        df = pd.DataFrame({
            'symbol': ['SPY'] * 20,
            'date': pd.date_range('2024-01-01', periods=20),
            'adj_close': np.linspace(120, 100, 20)  # All losses
        })
        
        result = compute_rsi(df, window=14)
        
        # RSI should be close to 0 (fully oversold)
        assert result['rsi_spy_14'].iloc[-1] < 5
    
    def test_compute_rsi_range(self):
        """Test that RSI is always between 0 and 100"""
        np.random.seed(42)
        df = pd.DataFrame({
            'symbol': ['SPY'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'adj_close': 100 + np.cumsum(np.random.randn(100))  #random walk
        })
        
        result = compute_rsi(df, window=14)
        
        # RSI should always be 0-100
        valid_rsi = result['rsi_spy_14'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_compute_rsi_neutral(self):
        """Test RSI is around 50 for balanced gains/losses"""
        # Alternating up and down
        prices = [100]
        for i in range(50):
            if i % 2 == 0:
                prices.append(prices[-1] * 1.01)  # +1%
            else:
                prices.append(prices[-1] * 0.99)  # -1%
        
        df = pd.DataFrame({
            'symbol': ['SPY'] * len(prices),
            'date': pd.date_range('2024-01-01', periods=len(prices)),
            'adj_close': prices
        })
        
        result = compute_rsi(df, window=14)
        
        # RSI should be around 50 (neutral)
        assert 40 < result['rsi_spy_14'].iloc[-1] < 60


class TestComputeCorrelations:
    """Tests for compute_correlations() function"""
    
    def test_compute_correlations_perfect_positive(self):
        """Test correlation = 1 when assets move together"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50)
        returns = np.random.randn(50) * 0.01  # Variable returns
        
        df = pd.DataFrame({
            'symbol': ['SPY'] * 50 + ['TLT'] * 50 + ['HYG'] * 50,
            'date': dates.tolist() * 3,
            'ret': list(returns) + list(returns) + list(returns)  # Same returns for all = perfect correlation
        })
        
        result = compute_correlations(df)
        
        # Correlation should be close to 1
        assert result['corr_spy_tlt_20d'].iloc[-1] == pytest.approx(1.0, abs=0.1)
    
    def test_compute_correlations_negative(self):
        """Test negative correlation when assets move opposite"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50)
        returns = np.random.randn(50) * 0.01
        
        df = pd.DataFrame({
            'symbol': ['SPY'] * 50 + ['TLT'] * 50 + ['HYG'] * 50,
            'date': dates.tolist() * 3,
            'ret': list(returns) + list(-returns) + list(returns * 0.5)  # TLT opposite, HYG neutral
        })
        
        result = compute_correlations(df)
        
        # Correlation should be close to -1
        assert result['corr_spy_tlt_20d'].iloc[-1] == pytest.approx(-1.0, abs=0.1)
    
    def test_compute_correlations_all_windows(self):
        """Test that both 20d and 60d correlations are computed"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        
        df = pd.DataFrame({
            'symbol': ['SPY'] * 100 + ['TLT'] * 100 + ['HYG'] * 100,
            'date': dates.tolist() * 3,
            'ret': np.random.randn(300) * 0.01
        })
        
        result = compute_correlations(df)
        
        # Check all correlation columns exist
        expected_cols = ['corr_spy_tlt_20d', 'corr_spy_hyg_20d', 'corr_spy_tlt_60d', 'corr_spy_hyg_60d']
        for col in expected_cols:
            assert col in result.columns


class TestComputeSpreads:
    """Tests for compute_spreads() function"""
    
    def test_compute_spreads_basic(self):
        """Test HYG-TLT spread calculation"""
        df = pd.DataFrame({
            'symbol': ['HYG', 'HYG', 'TLT', 'TLT'],
            'date': ['2024-01-01', '2024-01-02'] * 2,
            'ret': [0.01, 0.02, 0.005, 0.01]
        })
        
        result = compute_spreads(df)
        
        # Spread for 2024-01-01: 0.01 - 0.005 = 0.005
        assert result['hyg_tlt_spread'].iloc[0] == pytest.approx(0.005)
        
        # Spread for 2024-01-02: 0.02 - 0.01 = 0.01
        assert result['hyg_tlt_spread'].iloc[1] == pytest.approx(0.01)
    
    def test_compute_spreads_negative_spread(self):
        """Test negative spreads (TLT outperforms HYG)"""
        df = pd.DataFrame({
            'symbol': ['HYG', 'TLT'],
            'date': ['2024-01-01'] * 2,
            'ret': [0.005, 0.02]  # TLT higher
        })
        
        result = compute_spreads(df)
        
        # Spread should be negative: 0.005 - 0.02 = -0.015
        assert result['hyg_tlt_spread'].iloc[0] == pytest.approx(-0.015)


class TestBuildFeatures:
    """Tests for build_features() integration function"""
    
    def test_build_features_integration(self, sample_market_data):
        """Test that build_features merges all feature groups correctly"""
        result = build_features(sample_market_data)
        
        # Check that key columns exist
        expected_cols = [
            'date',
            'spy_ret_1d', 'spy_ret_5d', 'spy_vol_5d', 'drawdown_60d',
            'vix', 'vix3m', 'vix_term', 'rsi_spy_14',
            'corr_spy_tlt_20d', 'hyg_tlt_spread', 'rv_vix_spread_20d'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_build_features_no_data_loss(self, sample_market_data):
        """Test that build_features doesn't lose dates"""
        result = build_features(sample_market_data)
        
        # Should have one row per date
        expected_dates = sample_market_data['date'].nunique()
        assert len(result) == expected_dates
    
    def test_build_features_rv_vix_spread(self, sample_market_data):
        """Test that RV-VIX spread is computed correctly"""
        result = build_features(sample_market_data)
        
        # RV-VIX spread should be: spy_vol_20d - vix
        # Check that it's computed correctly for non-NaN rows
        valid_rows = result[['spy_vol_20d', 'vix', 'rv_vix_spread_20d']].dropna()
        
        if len(valid_rows) > 0:
            computed_spread = valid_rows['spy_vol_20d'] - valid_rows['vix']
            assert np.allclose(computed_spread, valid_rows['rv_vix_spread_20d'], rtol=1e-5)
