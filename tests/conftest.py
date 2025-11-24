"""
shared test fixtures for reusable test data

fixtures let you define data once and reuse it in tests
just add fixture name to test function params and pytest passes it in automatically

example: def test_something(sample_spy_data):
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_spy_data():
    """
    100 days of fake SPY data
    
    most feature functions need SPY data so this fixture provides it
    gives realistic looking data for testing
    
    what it has:
    prices follow random walk with cumsum
    OHLC values in reasonable ranges
    log returns like we actually use
    first return is NaN since cant compute on day 1
    
    use when testing SPY features like returns volatility drawdown
    or when you need simple single ticker price data
    
    dont use when you need multiple tickers like VIX TLT HYG
    for that use sample_market_data instead

    """
    np.random.seed(42)  #reproducible random data
    dates = pd.date_range('2024-01-01', periods=100)
    
    #random walk starting at 100
    #*2 means about $2 daily moves
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    return pd.DataFrame({
        'symbol': ['SPY'] * 100,
        'date': dates,
        'open': prices * 0.99,     #open slightly below close
        'high': prices * 1.01,     #high slightly above
        'low': prices * 0.98,      #low a bit lower
        'close': prices,
        'adj_close': prices,       #for testing just use close
        'volume': np.random.randint(1_000_000, 10_000_000, 100).astype(float),
        'ret': np.concatenate([[np.nan], np.diff(np.log(prices))])  #log returns first is NaN
    })


@pytest.fixture
def sample_vix_data():
    """
    100 days of fake VIX data
    
    VIX is different from regular stocks:
    no adj_close column since its an index not a stock
    volume is always 0 for same reason
    prices typically 10 to 50 range
    smaller volatility 0.5 vs 2.0 for SPY
    
    use this when testing VIX specific features
    or testing code that handles missing adj_close column
    or zero volume
    
    real VIX data doesnt have adj_close so our code needs to handle that
    this fixture tests that edge case
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    
    #start at 20 which is VIX average
    #smaller moves 0.5 vs 2.0 for SPY
    vix_prices = 20 + np.cumsum(np.random.randn(100) * 0.5)
    #keep it in realistic range
    vix_prices = np.clip(vix_prices, 10, 50)
    
    return pd.DataFrame({
        'symbol': ['^VIX'] * 100,
        'date': dates,
        'open': vix_prices * 0.99,
        'high': vix_prices * 1.05,   #VIX can have bigger intraday swings
        'low': vix_prices * 0.95,
        'close': vix_prices,
        'adj_close': vix_prices,     #reality VIX doesnt have this but some tests need it
        'volume': [0.0] * 100,       #no volume for index
        'ret': np.concatenate([[np.nan], np.diff(np.log(vix_prices))])
    })


@pytest.fixture
def sample_market_data():
    """
    100 days of data for all tickers we use
    
    includes:
    SPY - S&P 500 ETF main one were forecasting
    VIX - fear gauge
    VIX3M - 3 month VIX for term structure
    TLT - long term treasury bonds
    HYG - high yield corp bonds
    
    why all these:
    feature engineering needs correlations between assets
    compute_correlations needs SPY TLT and HYG
    compute_vix_features needs VIX and VIX3M for term structure
    
    each ticker gets different behavior:
    SPY base 100 high vol 2.0 since stocks are volatile
    VIX base 20 medium vol 0.5 mean reverting
    VIX3M base 22 low vol 0.4 more stable than VIX
    TLT base 90 medium vol 1.5 bonds less volatile than stocks
    HYG base 80 low vol 1.0 middle ground
    
    use this when testing compute_correlations or compute_spreads or build_features
    or integration tests that need all tickers
    
    dont use for simple tests that only need SPY
    use sample_spy_data for those instead
    
    creates 500 rows which is fine for tests
    real data way bigger
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    
    #each ticker gets base price and volatility
    tickers_config = {
        'SPY': {'base': 100, 'vol': 2},
        '^VIX': {'base': 20, 'vol': 0.5},
        '^VIX3M': {'base': 22, 'vol': 0.4},
        'TLT': {'base': 90, 'vol': 1.5},
        'HYG': {'base': 80, 'vol': 1}
    }
    
    dfs = []
    for ticker, config in tickers_config.items():
        #random walk for each ticker
        prices = config['base'] + np.cumsum(np.random.randn(100) * config['vol'])
        
        df = pd.DataFrame({
            'symbol': ticker,
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'adj_close': prices,
            'volume': np.random.randint(100_000, 1_000_000, 100).astype(float) if ticker != '^VIX' and ticker != '^VIX3M' else [0.0] * 100,
            'ret': np.concatenate([[np.nan], np.diff(np.log(prices))])
        })
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    fake data directory for tests that write files
    
    some tests write parquet files to disk like write_partitions
    dont want to mess up real data directory with test files
    so create temp directory that gets deleted after test
    
    tmp_path is pytest builtin that gives you temp directory
    this just adds our expected subdirs
    
    use when testing write_partitions or any file IO tests
    
    tmp_path auto cleans up after test
    """
    (tmp_path / 'raw.market').mkdir(parents=True)
    (tmp_path / 'curated.market').mkdir(parents=True)
    (tmp_path / 'features.L1').mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_yfinance_response():
    """
    fake yfinance API response for mocking
    
    dont want tests hitting real Yahoo Finance API because:
    its slow with network latency
    sometimes API is down
    costs bandwidth
    can get rate limited
    tests should be deterministic same result every time
    
    returns DataFrame that looks like yf.Ticker().history():
    MultiIndex columns like ('Price', 'Open')
    DatetimeIndex for index
    10 days of fake data
    
    use when testing download_one or mocking yfinance
    pair with pytest mock like mock.patch('yf.Ticker', ...)
    
    note yfinance returns MultiIndex for columns
    this fixture matches that structure
    """
    dates = pd.date_range('2024-01-01', periods=10)
    data = {
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
        'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'Adj Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
    }
    
    df = pd.DataFrame(data, index=dates)
    
    #yfinance returns MultiIndex columns
    df.columns = pd.MultiIndex.from_tuples([
        ('Open', 'SPY'), ('High', 'SPY'), ('Low', 'SPY'),
        ('Close', 'SPY'), ('Adj Close', 'SPY'), ('Volume', 'SPY')
    ])
    
    return df


@pytest.fixture
def mock_yfinance_vix_response():
    """
    fake yfinance response for VIX
    
    VIX is different:
    NO Adj Close column since its an index not a stock
    volume always 0 cant trade index directly
    
    download_one has special handling for VIX
    when no Adj Close it uses Close instead
    this fixture tests that logic
    
    use when testing download_one with VIX data
    or testing code that handles missing Adj Close
    
    prices go up then down 20 to 26 to 22 to give it some movement
    """
    dates = pd.date_range('2024-01-01', periods=10)
    data = {
        'Open': [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 24.0, 23.0, 22.0, 21.0],
        'High': [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 26.0, 25.0, 24.0, 23.0],
        'Low': [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 23.0, 22.0, 21.0, 20.0],
        'Close': [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 25.0, 24.0, 23.0, 22.0],
        'Volume': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Always 0 for indices
    }
    
    df = pd.DataFrame(data, index=dates)
    
    #key difference NO Adj Close in MultiIndex
    df.columns = pd.MultiIndex.from_tuples([
        ('Open', '^VIX'), ('High', '^VIX'), ('Low', '^VIX'),
        ('Close', '^VIX'), ('Volume', '^VIX')
        #no Adj Close tuple
    ])
    
    return df
