import pandera as pa
from pandera import Column, DataFrameSchema

raw_market_schema = DataFrameSchema({
    "symbol": Column(str),
    "date": Column(pa.DateTime),
    "open": Column(float),
    "high": Column(float),
    "low": Column(float),
    "close": Column(float),
    "volume": Column(int),
})

curated_market_daily_schema = DataFrameSchema({
    "symbol": Column(str),
    "date": Column(pa.DateTime),
    "open": Column(float),
    "high": Column(float),
    "return": Column(float),
    "adj_close": Column(float),
})