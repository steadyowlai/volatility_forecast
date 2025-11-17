import pandera as pa
from pandera import Column, DataFrameSchema

# raw market: exactly as ingested from yfinance
raw_market_schema = DataFrameSchema({
    "symbol": Column(str),
    "date": Column(pa.DateTime),
    "open": Column(float),
    "high": Column(float),
    "low": Column(float),
    "close": Column(float),
    "volume": Column(float),  # can be float or int depending on ticker
})

# curated market: cleaned, standardized, canonical daily dataset
curated_market_daily_schema = DataFrameSchema({
    "symbol": Column(str),
    "date": Column(pa.DateTime),
    "close": Column(float),
    "ret": Column(float, nullable=True),   # null for the first day of each symbol
    "adj_close": Column(float),
})