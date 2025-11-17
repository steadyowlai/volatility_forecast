import argparse
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from libs.schemas import raw_market_schema, curated_market_daily_schema
from libs import __init__  # noqa: F401 (ensure package import works in Docker)
from libs.io import write_parquet_partitioned


def _normalize_yf(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Normalize yfinance output to a flat schema with columns:
    symbol, date, open, high, low, close, volume, adj_close
    """
    # Handle single-ticker and multi-ticker shapes
    records = []
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex columns: level 0 = Ticker, level 1 = Field
        for t in tickers:
            if t not in df.columns.get_level_values(0):
                continue
            sub = df[t].copy()
            sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
            sub.reset_index(inplace=True)
            sub.rename(columns={"index": "date"}, inplace=True)
            sub.insert(0, "symbol", t)
            records.append(sub)
        if not records:
            return pd.DataFrame(columns=[
                "symbol", "date", "open", "high", "low", "close", "volume", "adj_close",
            ])
        out = pd.concat(records, ignore_index=True)
    else:
        # Single ticker: columns like Open, High, Low, Close, Adj Close, Volume
        t = tickers[0]
        sub = df.copy()
        sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
        sub.reset_index(inplace=True)
        sub.rename(columns={"index": "date"}, inplace=True)
        sub.insert(0, "symbol", t)
        out = sub

    # Enforce column order and types
    expected_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    # adj_close might be missing for certain tickers; fill via close as fallback
    if "adj_close" not in out.columns:
        out["adj_close"] = out.get("close")

    # Some tickers (like ^VIX) can have missing volume; default to 0 when absent
    if "volume" not in out.columns:
        out["volume"] = 0

    # Ensure datetime and dtypes
    out["date"] = pd.to_datetime(out["date"], utc=False)
    out = out[["symbol", "date", "open", "high", "low", "close", "volume", "adj_close"]]

    # Cast numeric types
    num_cols = ["open", "high", "low", "close", "adj_close"]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype("int64")

    # Drop rows with no price data
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    return out


def build_curated_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.sort_values(["symbol", "date"], inplace=True)
    df["return"] = df.groupby("symbol")["close"].pct_change()

    curated = df[["symbol", "date", "open", "high", "return", "adj_close"]].copy()
    curated = curated.dropna(subset=["return"])  # drop first row per symbol
    # Ensure types
    curated["date"] = pd.to_datetime(curated["date"], utc=False)
    for c in ["open", "high", "return", "adj_close"]:
        curated[c] = pd.to_numeric(curated[c], errors="coerce")
    curated = curated.dropna().reset_index(drop=True)
    return curated


def ingest(tickers: List[str], start: str, end: str | None) -> None:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
        interval="1d",
    )

    raw = _normalize_yf(data, tickers)
    # Validate against schema
    raw = raw_market_schema.validate(raw, lazy=True)

    # Write raw, partitioned by symbol
    write_parquet_partitioned(raw, root_path="data/raw.market", partition_col="symbol")

    # Build curated and validate
    curated = build_curated_from_raw(raw)
    curated = curated_market_daily_schema.validate(curated, lazy=True)
    write_parquet_partitioned(curated, root_path="data/curated.market_daily", partition_col="symbol")

    # Summary
    symbols = sorted(raw["symbol"].unique().tolist())
    start_d = raw["date"].min()
    end_d = raw["date"].max()
    print(
        f"Ingest complete: {len(raw):,} rows for {symbols} from {start_d.date()} to {end_d.date()}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest market data via yfinance")
    p.add_argument(
        "--tickers",
        type=str,
        default="SPY,^VIX,^VIX3M,TLT,HYG",
        help="Comma-separated list of tickers",
    )
    p.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date YYYY-MM-DD",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (optional)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tks = [t.strip() for t in args.tickers.split(",") if t.strip()]
    # Basic validation of dates
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        if args.end:
            datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        raise SystemExit(f"Invalid date format: {e}")

    ingest(tks, args.start, args.end)
