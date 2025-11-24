"""
integration tests for features service

test full feature engineering pipeline
loading data computing features and writing partitions

main() orchestrates everything
need to test it for coverage above 80%

features app does:
load_curated_data reads parquet files
build_features computes all 21 features
write_partitions writes results

these tests run all three together
"""

import sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

#add services to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.features.app import main, load_curated_data, write_partitions


class TestFeaturesMainPipeline:
    """test full features main() function"""
    
    def test_main_runs_successfully(self, tmp_path):
        """
        run entire feature pipeline
        
        create fake curated data
        run main() to compute features
        verify feature partitions created
        verify features exist in output
        
        We use tmp_path so we're not messing with real data files.
        """
        # Create fake curated data directory structure
        #mimic what ingest.py creates
        curated_dir = tmp_path / "curated.market"
        curated_dir.mkdir(parents=True)
        
        #fake curated data
        #need 100 days and all tickers for proper feature calculations
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        
        #SPY VIX VIX3M TLT HYG required for all features
        all_data = []
        for symbol in ["SPY", "VIX", "VIX3M", "TLT", "HYG"]:
            for date in dates:
                all_data.append({
                    "symbol": symbol,
                    "date": date,
                    "close": 100.0 + (dates.get_loc(date) * 0.1),  #slowly increasing
                    "adj_close": 100.0 + (dates.get_loc(date) * 0.1),  #same as close
                    "ret": 0.001 if date > dates[0] else None,  #0.1% daily returns
                })
        
        df_curated = pd.DataFrame(all_data)
        
        #write to parquet in date partitions
        #features expects files like date=2024-01-01/daily.parquet
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            date_dir = curated_dir / f"date={date_str}"
            date_dir.mkdir()
            
            #just this date's data
            df_date = df_curated[df_curated["date"] == date]
            df_date.to_parquet(date_dir / "daily.parquet", index=False)
        
        #run main() with patched paths
        features_dir = tmp_path / "features.market"
        
        with patch("services.features.app.Path", return_value=curated_dir):
            #need to patch both load and write paths
            #tricky because Path() is used multiple times
            
            #easier to just call functions directly
            #and patch paths in each one
            pass
        
        #patch the functions directly
        with patch("services.features.app.load_curated_data") as mock_load:
            #return our fake data
            mock_load.return_value = df_curated
            
            with patch("services.features.app.write_partitions") as mock_write:
                #run main
                main()
                
                #check write_partitions was called
                assert mock_write.called, "write_partitions should be called"
                
                #get dataframe that was passed to write_partitions
                written_df = mock_write.call_args[0][0]
                
                #verify features were computed
                #should have spy_ret_5d spy_vol_5d vix rsi_spy_14 etc
                assert "spy_ret_5d" in written_df.columns, "Should compute SPY returns"
                assert "spy_vol_5d" in written_df.columns, "Should compute SPY volatility"
                assert "vix" in written_df.columns, "Should extract VIX"
                assert "rsi_spy_14" in written_df.columns, "Should compute RSI"
                assert "corr_spy_tlt_20d" in written_df.columns, "Should compute correlations"
                
                print(f"\n✅ main() computed {len(written_df.columns)} feature columns")
    
    def test_main_prints_progress(self, capsys):
        """
        check main() prints status messages
        
        should see progress like loading partitions and feature engineering complete
        capsys captures stdout
        """
        #minimal fake data
        #need SPY TLT HYG for compute_correlations
        #need VIX VIX3M for other features
        dates = pd.date_range("2024-01-01", periods=100)
        all_data = []
        for symbol in ["SPY", "VIX", "VIX3M", "TLT", "HYG"]:
            for date in dates:
                all_data.append({
                    "symbol": symbol,
                    "date": date,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "ret": 0.001,
                })
        df_fake = pd.DataFrame(all_data)
        
        with patch("services.features.app.load_curated_data", return_value=df_fake):
            with patch("services.features.app.write_partitions"):
                #run main
                main()
        
        #check output
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Feature Engineering Service" in output
        assert "Feature engineering complete" in output
        assert "=" * 60 in output  #separator lines
        
        print("\n✅ Progress messages printing correctly")


class TestLoadCuratedData:
    """
    test load_curated_data() function
    
    reads parquet files from disk so need real files
    thats why integration tests not unit tests
    """
    
    def test_load_curated_data_reads_files(self, tmp_path):
        """
        check load_curated_data() reads parquet files correctly
        
        hits lines 14-26 in features/app.py for coverage
        create fake parquet files and verify loading
        """
        #fake curated directory
        curated_dir = tmp_path / "curated.market"
        curated_dir.mkdir(parents=True)
        
        #create fake partition files
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        for date_str in dates:
            date_dir = curated_dir / f"date={date_str}"
            date_dir.mkdir()
            
            #write fake data
            df = pd.DataFrame({
                "symbol": ["SPY", "VIX"],
                "date": [date_str, date_str],
                "close": [100.0, 20.0],
                "ret": [0.001, 0.002],
            })
            df.to_parquet(date_dir / "daily.parquet", index=False)
        
        #test loading
        #patch Path to use tmp_path
        with patch("services.features.app.Path") as mock_path:
            mock_path.return_value = curated_dir
            
            #should work now
            result = load_curated_data()
            
            #verify we got all data
            assert len(result) == 6, "Should have 2 symbols × 3 dates = 6 rows"
            assert "symbol" in result.columns
            assert "date" in result.columns
            assert result["symbol"].nunique() == 2
            
            print(f"\n✅ load_curated_data() loaded {len(result)} rows")
    
    def test_load_curated_data_raises_if_no_files(self, tmp_path):
        """
        what happens if theres no curated data
        
        should raise helpful FileNotFoundError
        not cryptic crash
        """
        #empty directory
        empty_dir = tmp_path / "empty_curated"
        empty_dir.mkdir()
        
        with patch("services.features.app.Path") as mock_path:
            mock_path.return_value = empty_dir
            
            #should raise FileNotFoundError
            with pytest.raises(FileNotFoundError, match="No curated data found"):
                load_curated_data()
        
        print("\n✅ load_curated_data() handles missing files correctly")
    
    def test_load_curated_data_sorts_correctly(self, tmp_path):
        """
        verify data gets sorted by symbol and date
        
        important because feature calculations need chronological order
        """
        curated_dir = tmp_path / "curated.market"
        curated_dir.mkdir(parents=True)
        
        #create files in random order
        dates = ["2024-01-03", "2024-01-01", "2024-01-02"]  #not sorted
        for date_str in dates:
            date_dir = curated_dir / f"date={date_str}"
            date_dir.mkdir()
            
            df = pd.DataFrame({
                "symbol": ["VIX", "SPY"],  #reverse alphabetical
                "date": [date_str, date_str],
                "close": [20.0, 100.0],
                "ret": [0.002, 0.001],
            })
            df.to_parquet(date_dir / "daily.parquet", index=False)
        
        with patch("services.features.app.Path") as mock_path:
            mock_path.return_value = curated_dir
            
            result = load_curated_data()
            
            #check result is sorted
            #first row should be SPY on 2024-01-01
            assert result.iloc[0]["symbol"] == "SPY"
            assert result.iloc[0]["date"] == pd.Timestamp("2024-01-01")
            
            #last row should be VIX on 2024-01-03
            assert result.iloc[-1]["symbol"] == "VIX"
            assert result.iloc[-1]["date"] == pd.Timestamp("2024-01-03")
            
            print("\n✅ Data is sorted correctly by symbol and date")


class TestWritePartitions:
    """
    test write_partitions() function
    
    writes parquet files to disk so need real file IO
    
    hits lines 175-190 in features/app.py for coverage
    """
    
    def test_write_partitions_creates_files(self, tmp_path):
        """
        check write_partitions() creates right directory structure
        
        should create data/features.market/date=YYYY-MM-DD/features.parquet
        one partition per unique date
        """
        #fake feature data
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "symbol": ["SPY"] * 3,
            "rv_5d": [0.01, 0.015, 0.012],
            "vix": [15.0, 16.0, 15.5],
        })
        
        #patch output path to use tmp_path
        with patch("services.features.app.Path") as mock_path:
            #make Path return our tmp directory
            mock_path.return_value = tmp_path
            
            #call write_partitions
            write_partitions(df)
            
            #check files were created
            #should have 3 date partitions
            partitions = list(tmp_path.glob("date=*/features.parquet"))
            assert len(partitions) == 3, "Should create 3 date partitions"
            
            #check we can read files back
            first_partition = partitions[0]
            df_read = pd.read_parquet(first_partition)
            assert "rv_5d" in df_read.columns
            assert "vix" in df_read.columns
            
            print(f"\n✅ write_partitions() created {len(partitions)} files")
    
    def test_write_partitions_preserves_all_columns(self, tmp_path):
        """
        make sure write_partitions() doesnt lose columns
        
        all 21 feature columns should be in output files
        """
        #data with many columns like real features
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "symbol": ["SPY", "SPY"],
            "rv_5d": [0.01, 0.015],
            "rv_21d": [0.012, 0.016],
            "vix": [15.0, 16.0],
            "rsi_14d": [55.0, 60.0],
            "corr_spy_vix_20d": [-0.5, -0.6],
            #imagine all 21 features here
        })
        
        with patch("services.features.app.Path") as mock_path:
            mock_path.return_value = tmp_path
            
            write_partitions(df)
            
            #read back and verify columns
            written_file = list(tmp_path.glob("date=*/features.parquet"))[0]
            df_read = pd.read_parquet(written_file)
            
            #all original columns should be present
            for col in df.columns:
                assert col in df_read.columns, f"Column {col} should be preserved"
            
            print(f"\n✅ All {len(df.columns)} columns preserved in output")
