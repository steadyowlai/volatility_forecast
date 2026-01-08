"""
Unit tests for services/train/app.py
Tests critical helper functions with mocking for external dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path so we can import services
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.train.app import (
    compute_rv_5d,
    walk_forward_split,
    load_model_config,
)


class TestComputeRV5d:
    """Tests for compute_rv_5d() function - computes 5-day forward realized volatility"""
    
    def test_compute_rv_5d_basic_calculation(self):
        """Test that RV_5d is computed correctly using the formula"""
        # Create simple SPY data with known returns
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        returns = [0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01, 0.005, -0.01]
        
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': dates,
            'ret': returns
        })
        
        result = compute_rv_5d(df)
        
        # Should have date and rv_5d columns
        assert 'date' in result.columns
        assert 'rv_5d' in result.columns
        assert len(result) == 10
        
        # RV_5d at day 0 = sqrt(ret[1]^2 + ret[2]^2 + ret[3]^2 + ret[4]^2 + ret[5]^2)
        # Using shift(-1) through shift(-5)
        expected_rv_0 = np.sqrt(
            returns[1]**2 + returns[2]**2 + returns[3]**2 + returns[4]**2 + returns[5]**2
        )
        
        assert np.isclose(result['rv_5d'].iloc[0], expected_rv_0, rtol=1e-5)
    
    def test_compute_rv_5d_last_values_nan(self):
        """Test that last 5 values are NaN (no future returns available)"""
        dates = pd.date_range('2024-01-01', periods=10)
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': dates,
            'ret': np.random.randn(10)
        })
        
        result = compute_rv_5d(df)
        
        # Last 5 values should be NaN (can't look forward)
        assert result['rv_5d'].iloc[-5:].isna().all()
        
        # First 5 values should have data
        assert result['rv_5d'].iloc[:5].notna().any()
    
    def test_compute_rv_5d_filters_spy_only(self):
        """Test that RV_5d only computed for SPY, not other symbols"""
        df = pd.DataFrame({
            'symbol': ['SPY', 'VIX', 'SPY', 'VIX', 'SPY', 'TLT'],
            'date': pd.date_range('2024-01-01', periods=6),
            'ret': [0.01, 0.02, -0.01, -0.02, 0.005, 0.01]
        })
        
        result = compute_rv_5d(df)
        
        # Should only have SPY rows (3 rows)
        assert len(result) == 3
        assert set(result.columns) == {'date', 'rv_5d'}
    
    def test_compute_rv_5d_zero_returns(self):
        """Test RV_5d with zero returns (should be zero)"""
        dates = pd.date_range('2024-01-01', periods=10)
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': dates,
            'ret': [0.0] * 10
        })
        
        result = compute_rv_5d(df)
        
        # First value should be 0 (sqrt of sum of zeros)
        assert result['rv_5d'].iloc[0] == 0.0
    
    def test_compute_rv_5d_positive_values_only(self):
        """Test that RV_5d is always non-negative (sqrt of squared values)"""
        dates = pd.date_range('2024-01-01', periods=20)
        df = pd.DataFrame({
            'symbol': ['SPY'] * 20,
            'date': dates,
            'ret': np.random.randn(20)  # Mix of positive and negative
        })
        
        result = compute_rv_5d(df)
        
        # All non-NaN values should be >= 0
        valid_values = result['rv_5d'].dropna()
        assert (valid_values >= 0).all()
    
    def test_compute_rv_5d_sorted_by_date(self):
        """Test that function works with unsorted data"""
        dates = pd.date_range('2024-01-01', periods=10)
        # Create data in random order
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': dates,
            'ret': np.random.randn(10)
        })
        # Shuffle it
        df = df.sample(frac=1).reset_index(drop=True)
        
        result = compute_rv_5d(df)
        
        # Should still compute without error
        assert len(result) == 10
        assert 'rv_5d' in result.columns


class TestWalkForwardSplit:
    """Tests for walk_forward_split() - time-series train/val split"""
    
    def test_walk_forward_split_default_ratio(self):
        """Test default 80/20 train/val split"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'feature1': np.random.randn(100),
            'rv_5d': np.random.randn(100)
        })
        
        train, val = walk_forward_split(df, train_size=0.8)
        
        # Check sizes
        assert len(train) == 80
        assert len(val) == 20
        
        # Check that we get back dataframes
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
    
    def test_walk_forward_split_custom_ratio(self):
        """Test custom train/val split ratios"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'value': range(100)
        })
        
        # 70/30 split
        train, val = walk_forward_split(df, train_size=0.7)
        assert len(train) == 70
        assert len(val) == 30
        
        # 90/10 split
        train, val = walk_forward_split(df, train_size=0.9)
        assert len(train) == 90
        assert len(val) == 10
    
    def test_walk_forward_split_time_ordering(self):
        """Test that train comes before val (no data leakage)"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'value': range(100)
        })
        
        train, val = walk_forward_split(df, train_size=0.8)
        
        # Train dates should all be before val dates
        assert train['date'].max() < val['date'].min()
        
        # Values should be sequential (no overlap)
        assert train['value'].max() < val['value'].min()
    
    def test_walk_forward_split_no_overlap(self):
        """Test that there's no overlap between train and val"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'id': range(100)
        })
        
        train, val = walk_forward_split(df, train_size=0.75)
        
        # No shared IDs
        train_ids = set(train['id'])
        val_ids = set(val['id'])
        assert len(train_ids.intersection(val_ids)) == 0
        
        # All IDs accounted for
        assert len(train_ids.union(val_ids)) == 100
    
    def test_walk_forward_split_preserves_columns(self):
        """Test that all columns are preserved in split"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randn(50)
        })
        
        train, val = walk_forward_split(df, train_size=0.6)
        
        # Same columns in both
        assert set(train.columns) == set(df.columns)
        assert set(val.columns) == set(df.columns)
    
    def test_walk_forward_split_sorts_by_date(self):
        """Test that function sorts by date before splitting"""
        # Create unsorted data
        dates = pd.date_range('2024-01-01', periods=20)
        df = pd.DataFrame({
            'date': dates,
            'value': range(20)
        })
        # Shuffle it
        df = df.sample(frac=1).reset_index(drop=True)
        
        train, val = walk_forward_split(df, train_size=0.75)
        
        # Should still maintain time ordering
        assert train['date'].is_monotonic_increasing
        assert val['date'].is_monotonic_increasing
        assert train['date'].max() < val['date'].min()
    
    def test_walk_forward_split_small_dataset(self):
        """Test split with very small dataset"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'value': range(10)
        })
        
        train, val = walk_forward_split(df, train_size=0.8)
        
        # 8/2 split
        assert len(train) == 8
        assert len(val) == 2
    
    def test_walk_forward_split_large_train_ratio(self):
        """Test with large train ratio (95%)"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'value': range(100)
        })
        
        train, val = walk_forward_split(df, train_size=0.95)
        
        assert len(train) == 95
        assert len(val) == 5
        assert train['date'].max() < val['date'].min()
    
    def test_walk_forward_split_maintains_index(self):
        """Test that train indices start at 0, val may be sequential from train"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'value': range(50)
        })
        
        train, val = walk_forward_split(df, train_size=0.8)
        
        # Train indices should start at 0
        assert train.index[0] == 0
        
        # Both should have continuous indices
        assert train.index.is_monotonic_increasing
        assert val.index.is_monotonic_increasing


class TestDataLoading:
    """Tests for data loading functions (would need fixtures/mocking in real tests)"""
    
    def test_load_features_structure(self):
        """
        Placeholder test for load_features()
        
        In a real test, you'd:
        1. Create fixture parquet files
        2. Mock the file paths
        3. Test that loading works correctly
        
        Skipping for now since it requires actual data files
        """
        pytest.skip("Requires fixture data - implement later")
    
    def test_load_curated_data_structure(self):
        """
        Placeholder test for load_curated_data()
        
        Same as above - needs fixture data
        """
        pytest.skip("Requires fixture data - implement later")


class TestPrepareDataset:
    """Tests for prepare_dataset() - integration of loading and merging"""
    
    def test_prepare_dataset_integration(self):
        """
        Placeholder test for full dataset preparation
        
        This would be an integration test that:
        1. Uses fixture data
        2. Tests the full load -> merge -> clean pipeline
        3. Validates output structure
        
        Mark as integration test since it touches multiple functions
        """
        pytest.skip("Integration test - implement after fixtures ready")


# Additional edge case tests
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_rv_5d_with_nan_returns(self):
        """Test RV_5d computation when returns contain NaN"""
        dates = pd.date_range('2024-01-01', periods=10)
        returns = [0.01, np.nan, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01, 0.005, -0.01]
        
        df = pd.DataFrame({
            'symbol': ['SPY'] * 10,
            'date': dates,
            'ret': returns
        })
        
        result = compute_rv_5d(df)
        
        # Should handle NaN gracefully
        assert 'rv_5d' in result.columns
        # Values with NaN in lookforward window should be NaN
        assert pd.isna(result['rv_5d'].iloc[0])  # First value has NaN in lookforward
    
    def test_walk_forward_split_single_row(self):
        """Test edge case: dataset with only 1 row"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=1),
            'value': [1]
        })
        
        train, val = walk_forward_split(df, train_size=0.8)
        
        # Should handle gracefully (0 train, 1 val or 1 train, 0 val)
        assert len(train) + len(val) == 1


class TestLoadModelConfig:
    """Tests for load_model_config() - loads hyperparameters from config file"""
    
    def test_load_model_config_returns_default_when_no_file(self, tmp_path, monkeypatch):
        """Test that default XGBoost config is returned when best_params.json doesn't exist"""
        # Mock CONFIG_FILE to point to non-existent file
        import services.train.app as train_app
        fake_config_path = tmp_path / "nonexistent" / "best_params.json"
        monkeypatch.setattr(train_app, 'CONFIG_FILE', fake_config_path)
        
        config = load_model_config()
        
        # Should return default config
        assert config['model_family'] == 'xgboost'
        assert config['model_type'] == 'XGBoost'
        assert 'params' in config
        
        # Check default params structure
        params = config['params']
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'n_estimators' in params
        assert params['max_depth'] == 6
        assert params['n_estimators'] == 300
    
    def test_load_model_config_loads_from_file(self, tmp_path, monkeypatch):
        """Test that config is loaded from best_params.json when it exists"""
        import services.train.app as train_app
        import json
        
        # Create fake config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "best_params.json"
        
        test_config = {
            'model_family': 'lightgbm',
            'model_type': 'LightGBM',
            'params': {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 200
            },
            'tuned_on': '2024-01-01T12:00:00'
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        # Mock CONFIG_FILE to point to our test file
        monkeypatch.setattr(train_app, 'CONFIG_FILE', config_file)
        
        config = load_model_config()
        
        # Should load the test config
        assert config['model_family'] == 'lightgbm'
        assert config['model_type'] == 'LightGBM'
        assert config['params']['num_leaves'] == 31
        assert config['params']['learning_rate'] == 0.1
        assert config['tuned_on'] == '2024-01-01T12:00:00'
    
    def test_load_model_config_handles_xgboost_config(self, tmp_path, monkeypatch):
        """Test loading XGBoost config from tuning results"""
        import services.train.app as train_app
        import json
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "best_params.json"
        
        xgboost_config = {
            'model_family': 'xgboost',
            'model_type': 'XGBoost',
            'params': {
                'max_depth': 8,
                'learning_rate': 0.03,
                'n_estimators': 500,
                'subsample': 0.9,
                'colsample_bytree': 0.7
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(xgboost_config, f)
        
        monkeypatch.setattr(train_app, 'CONFIG_FILE', config_file)
        
        config = load_model_config()
        
        assert config['model_family'] == 'xgboost'
        assert config['params']['max_depth'] == 8
        assert config['params']['learning_rate'] == 0.03
    
    def test_load_model_config_handles_random_forest_config(self, tmp_path, monkeypatch):
        """Test loading Random Forest config from tuning results"""
        import services.train.app as train_app
        import json
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "best_params.json"
        
        rf_config = {
            'model_family': 'random_forest',
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 5
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(rf_config, f)
        
        monkeypatch.setattr(train_app, 'CONFIG_FILE', config_file)
        
        config = load_model_config()
        
        assert config['model_family'] == 'random_forest'
        assert config['params']['n_estimators'] == 300
    
    def test_load_model_config_handles_lightgbm_config(self, tmp_path, monkeypatch):
        """Test loading LightGBM config from tuning results"""
        import services.train.app as train_app
        import json
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "best_params.json"
        
        lgbm_config = {
            'model_family': 'lightgbm',
            'model_type': 'LightGBM',
            'params': {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'min_child_samples': 20
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(lgbm_config, f)
        
        monkeypatch.setattr(train_app, 'CONFIG_FILE', config_file)
        
        config = load_model_config()
        
        assert config['model_family'] == 'lightgbm'
        assert config['params']['num_leaves'] == 31
        assert config['params']['learning_rate'] == 0.1
    
    def test_load_model_config_params_structure(self, tmp_path, monkeypatch):
        """Test that params dict has expected structure"""
        import services.train.app as train_app
        
        # Test with non-existent file (defaults)
        fake_path = tmp_path / "fake.json"
        monkeypatch.setattr(train_app, 'CONFIG_FILE', fake_path)
        
        config = load_model_config()
        
        # Params should be a dict
        assert isinstance(config['params'], dict)
        assert len(config['params']) > 0
        
        # Should have numeric hyperparameters
        for key, value in config['params'].items():
            assert isinstance(value, (int, float, str)), f"Param {key} should be numeric or string"
    
    def test_load_model_config_missing_tuned_on_field(self, tmp_path, monkeypatch):
        """Test that config without 'tuned_on' field still loads"""
        import services.train.app as train_app
        import json
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "best_params.json"
        
        # Config without tuned_on field
        minimal_config = {
            'model_family': 'xgboost',
            'model_type': 'XGBoost',
            'params': {'max_depth': 5}
        }
        
        with open(config_file, 'w') as f:
            json.dump(minimal_config, f)
        
        monkeypatch.setattr(train_app, 'CONFIG_FILE', config_file)
        
        config = load_model_config()
        
        # Should load successfully
        assert config['model_family'] == 'xgboost'
        assert config['params']['max_depth'] == 5
