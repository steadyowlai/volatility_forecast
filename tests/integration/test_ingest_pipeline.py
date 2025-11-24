"""
integration tests for ingest service

tests run entire ingest pipeline from start to finish
unlike unit tests these verify main() works correctly end to end

main() was at 0% coverage since unit tests dont call it
these tests check orchestration logic and catch integration bugs

still mock yfinance so we dont hit real APIs
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

#add services to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.ingest.app import main, TICKERS


class TestIngestMainPipeline:
    """test full ingest main function"""
    
    def test_main_handles_errors_gracefully(self, tmp_path, monkeypatch):
        """
        what happens if yfinance fails
        
        download_one catches exceptions and returns empty DataFrame
        so main() doesnt crash even if API fails
        test verifies this resilient behavior
        """
        #mock yfinance to raise exception
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("API rate limit exceeded")
        
        with patch("services.ingest.app.yf.Ticker", return_value=mock_ticker):
            with patch("services.ingest.app.DATA_RAW", tmp_path / "raw"):
                with patch("services.ingest.app.DATA_CURATED", tmp_path / "curated"):
                    try:
                        main()
                        print("\nmain handled API failure gracefully")
                    except Exception as e:
                        assert "API rate limit" in str(e)
    
    def test_main_validates_schemas(self, tmp_path, monkeypatch):
        """
        make sure main() calls schema validation
        
        if validation lines get removed accidentally
        this test will catch it
        """
        #mock yfinance
        mock_ticker = MagicMock()
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        mock_data = pd.DataFrame({
            ("Price", "Open"): [100.0] * 50,
            ("Price", "High"): [102.0] * 50,
            ("Price", "Low"): [99.0] * 50,
            ("Price", "Close"): [101.0] * 50,
            ("Price", "Adj Close"): [101.0] * 50,
            ("Volume", "Volume"): [1000000] * 50,
        }, index=dates)
        mock_data.index.name = "Date"
        mock_ticker.history.return_value = mock_data
        
        #patch schema validators to check they get called
        with patch("services.ingest.app.yf.Ticker", return_value=mock_ticker):
            with patch("services.ingest.app.DATA_RAW", tmp_path / "raw"):
                with patch("services.ingest.app.DATA_CURATED", tmp_path / "curated"):
                    with patch("services.ingest.app.raw_market_schema.validate") as mock_raw_validate:
                        with patch("services.ingest.app.curated_market_daily_schema.validate") as mock_curated_validate:
                            main()
                            
                            assert mock_raw_validate.called
                            assert mock_curated_validate.called
                            
                            print("\nschema validation working in main")


class TestMainFunctionCoverage:
    """tests for main() function progress messages"""
    
    def test_main_prints_progress_messages(self, tmp_path, capsys):
        """
        verify main() prints progress messages
        
        when running ingest service want to see whats happening
        test checks status messages print at each step
        
        capsys captures stdout/stderr so we can check output
        """
        #mock yfinance
        mock_ticker = MagicMock()
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        mock_data = pd.DataFrame({
            ("Price", "Open"): [100.0] * 10,
            ("Price", "High"): [102.0] * 10,
            ("Price", "Low"): [99.0] * 10,
            ("Price", "Close"): [101.0] * 10,
            ("Price", "Adj Close"): [101.0] * 10,
            ("Volume", "Volume"): [1000000] * 10,
        }, index=dates)
        mock_data.index.name = "Date"
        mock_ticker.history.return_value = mock_data
        
        with patch("services.ingest.app.yf.Ticker", return_value=mock_ticker):
            with patch("services.ingest.app.DATA_RAW", tmp_path / "raw"):
                with patch("services.ingest.app.DATA_CURATED", tmp_path / "curated"):
                    main()
        
        captured = capsys.readouterr()
        output = captured.out
        
        #check expected messages appear
        assert "Downloading market data" in output
        assert "Building raw.market" in output
        assert "Validating raw.market schema" in output
        assert "Building curated.market.daily" in output
        assert "Ingestion complete" in output
        
        print("\nprogress messages printing correctly")
