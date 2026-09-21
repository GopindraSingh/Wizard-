"""
====================================================================
NSE INTRADAY 09:45 SHORT-BIASED SCANNER - V2
====================================================================

V2 OBJECTIVES
--------------
Designed around the 09:45 IST decision point.

Signal candles:

    09:15 - 09:30
    09:30 - 09:45

Decision:

    End of 09:45 candle

V2 IMPROVEMENTS
----------------

1. Historical indicator warm-up
   --------------------------------
   EMA9 / EMA20 / RSI / MACD / ATR are NOT calculated from only
   the first two candles of the current session.

   Historical 15-minute candles are loaded before the target date.

2. Session VWAP
   --------------------------------
   VWAP resets at 09:15 every trading day.

3. Same-time-of-day relative volume
   --------------------------------
   Today's 09:15-09:30 volume is compared with historical
   09:15-09:30 volume.

   Today's 09:30-09:45 volume is compared with historical
   09:30-09:45 volume.

4. Market-relative strength
   --------------------------------
   Stock performance is compared with NIFTY 50 performance.

5. Volatility normalization
   --------------------------------
   Movement is normalized using ATR.

6. Reduced correlated scoring
   --------------------------------
   VWAP, EMA, MACD, price action etc. are grouped into feature
   families instead of blindly adding points for every indicator.

7. Continuous scoring
   --------------------------------
   Most features contribute smoothly rather than through arbitrary
   threshold jumps.

8. Exhaustion model
   --------------------------------
   RSI, VWAP extension, EMA extension, opening collapse,
   lower wick and momentum deceleration are treated as exhaustion
   features.

9. Transaction costs
   --------------------------------
   Historical evaluation includes configurable slippage and costs.

10. 1-minute post-entry evaluation
    --------------------------------
    If enabled and available, target/stop sequencing is evaluated
    using 1-minute candles instead of assuming the order inside a
    15-minute candle.

11. Better statistics
    --------------------------------
    Expectancy
    Win rate
    Profit factor
    Average win
    Average loss
    MFE
    MAE
    Drawdown
    Net return

IMPORTANT
---------
This remains a RESEARCH / SCANNING SYSTEM.

It does NOT guarantee future returns.

Yahoo Finance intraday history is also not a suitable long-term
institutional backtesting database. For serious historical research,
replace the data layer with a reliable NSE intraday provider.

====================================================================
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ====================================================================
# CONFIGURATION
# ====================================================================

NIFTY_500_URL = (
    "https://nsearchives.nseindia.com/content/indices/"
    "ind_nifty500list.csv"
)

NIFTY_50_URL = (
    "https://nsearchives.nseindia.com/content/indices/"
    "ind_nifty50list.csv"
)

MARKET_TZ = "Asia/Kolkata"

MARKET_OPEN = "09:15"
OPENING_END = "09:30"
DECISION_TIME = "09:45"
MARKET_CLOSE = "15:30"

SIGNAL_INTERVAL = "15m"

# Yahoo's 15m intraday availability is limited.
# Keep this reasonably small.
WARMUP_DAYS = 45

# Number of previous sessions used for relative volume.
RVOL_LOOKBACK_DAYS = 20

# Minimum number of historical observations required
# before relative volume is considered reliable.
MIN_RVOL_OBSERVATIONS = 8


# ====================================================================
# UNIVERSE
# ====================================================================

# The original script removed NIFTY 50 from NIFTY 500.
# Preserve that behavior by default.

EXCLUDE_NIFTY50 = True

# NOTE:
# Historical scans using today's index constituents still have
# survivorship bias.
#
# A production research system should replace build_universe()
# with a historical constituent database.


# ====================================================================
# LIQUIDITY
# ====================================================================

MIN_TURNOVER_CR = 2.0


# ====================================================================
# INDICATORS
# ====================================================================

EMA_FAST = 9
EMA_SLOW = 20

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_PERIOD = 14


# ====================================================================
# SIGNAL THRESHOLDS
# ====================================================================

SHORT_TRADEABLE_SCORE = 60.0

SHORT_HIGH_CONVICTION_SCORE = 75.0

LONG_TRADEABLE_SCORE = 70.0

LONG_HIGH_CONVICTION_SCORE = 80.0


# ====================================================================
# MOVEMENT REQUIREMENTS
# ====================================================================

MIN_SHORT_TOTAL_MOVE = 0.30

MIN_LONG_TOTAL_MOVE = 0.50

MIN_OPENING_SHORT_MOVE = 0.25

MIN_OPENING_LONG_MOVE = 0.35


# ====================================================================
# EXHAUSTION
# ====================================================================

RSI_OVERSOLD = 30.0

RSI_EXTREME_OVERSOLD = 22.0

MAX_SHORT_VWAP_DISTANCE = 3.0

MAX_SHORT_EMA20_DISTANCE = 3.5

MAX_OPENING_COLLAPSE = 3.0


# ====================================================================
# RISK / OUTCOME
# ====================================================================

TARGET_PCT = 1.00

STOP_PCT = 0.75

# Estimated round-trip trading cost.
#
# This is intentionally configurable.
#
# Do NOT assume this is your exact Zerodha/Upstox/etc cost.
# Replace with your actual cost model.

ROUND_TRIP_COST_PCT = 0.10

# Additional slippage per side.
SLIPPAGE_PER_SIDE_PCT = 0.05

TOTAL_SLIPPAGE_PCT = (
    SLIPPAGE_PER_SIDE_PCT * 2
)


# ====================================================================
# HISTORICAL EVALUATION
# ====================================================================

USE_1M_EVALUATION = True

# If 1-minute data is unavailable, optionally fall back to 15m.
ALLOW_15M_FALLBACK = True


# ====================================================================
# OUTPUT
# ====================================================================

TOP_SHORTS_TO_SHOW = 5

TOP_LONGS_TO_SHOW = 3

EXPORT_RESULTS = True

EXPORT_FILENAME = (
    "nse_0945_scanner_v2_results.csv"
)


# ====================================================================
# DOWNLOAD
# ====================================================================

BATCH_SIZE = 40

DOWNLOAD_TIMEOUT = 20

BATCH_DELAY = 0.8


# ====================================================================
# BENCHMARK
# ====================================================================

NIFTY50_TICKER = "^NSEI"


# ====================================================================
# DATA CLASS
# ====================================================================

@dataclass
class CostModel:

    round_trip_cost_pct: float = (
        ROUND_TRIP_COST_PCT
    )

    total_slippage_pct: float = (
        TOTAL_SLIPPAGE_PCT
    )

    @property
    def total_cost_pct(self) -> float:

        return (
            self.round_trip_cost_pct
            + self.total_slippage_pct
        )


COST_MODEL = CostModel()


# ====================================================================
# TERMINAL HELPERS
# ====================================================================

def clear_line() -> None:

    sys.stdout.write(
        "\r\033[2K"
    )

    sys.stdout.flush()


def progress_line(
    text: str
) -> None:

    sys.stdout.write(
        "\r\033[2K" + text
    )

    sys.stdout.flush()


# ====================================================================
# DATE
# ====================================================================

def parse_date(
    value: str
) -> pd.Timestamp:

    return pd.to_datetime(
        value,
        format="%Y-%m-%d"
    ).normalize()


def ask_date() -> pd.Timestamp:

    while True:

        value = input(
            "DATE (YYYY-MM-DD): "
        ).strip()

        try:

            return parse_date(
                value
            )

        except ValueError:

            print(
                "Invalid date. "
                "Use YYYY-MM-DD."
            )


# ====================================================================
# NSE UNIVERSE
# ====================================================================

def download_nse_csv(
    url: str
) -> pd.DataFrame:

    headers = {

        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Safari/537.36"
        ),

        "Accept": (
            "text/csv,application/csv,"
            "application/octet-stream,*/*"
        ),

        "Referer": (
            "https://www.nseindia.com/"
        ),
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return pd.read_csv(
        io.BytesIO(
            response.content
        )
    )


def find_symbol_column(
    df: pd.DataFrame
) -> str:

    candidates = [
        "Symbol",
        "SYMBOL",
        "symbol",
        "Ticker",
        "TICKER",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    raise ValueError(
        "Could not find NSE symbol column."
    )


def get_index_symbols(
    url: str
) -> set[str]:

    df = download_nse_csv(
        url
    )

    column = find_symbol_column(
        df
    )

    symbols = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return set(symbols)


def build_universe() -> List[str]:

    print(
        "Downloading NSE universe..."
    )

    nifty500 = get_index_symbols(
        NIFTY_500_URL
    )

    if EXCLUDE_NIFTY50:

        nifty50 = get_index_symbols(
            NIFTY_50_URL
        )

        symbols = (
            nifty500 - nifty50
        )

    else:

        symbols = nifty500

    tickers = [
        f"{symbol}.NS"
        for symbol in sorted(
            symbols
        )
        if symbol
    ]

    print(
        f"Universe: {len(tickers)} stocks"
    )

    return tickers


# ====================================================================
# YFINANCE
# ====================================================================

def yf_download_quiet(
    *args,
    **kwargs
) -> pd.DataFrame:

    stdout = io.StringIO()

    stderr = io.StringIO()

    try:

        with contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(
            stderr
        ):

            result = yf.download(
                *args,
                **kwargs
            )

        if result is None:

            return pd.DataFrame()

        return result

    except Exception:

        return pd.DataFrame()


# ====================================================================
# COLUMN HANDLING
# ====================================================================

def flatten_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    return df


# ====================================================================
# INDEX NORMALIZATION
# ====================================================================

def normalize_intraday_index(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df.empty:

        return df

    df = df.copy()

    df.index = pd.to_datetime(
        df.index
    )

    if getattr(
        df.index,
        "tz",
        None
    ) is not None:

        df.index = (
            df.index
            .tz_convert(
                MARKET_TZ
            )
        )

    else:

        df.index = (
            df.index
            .tz_localize(
                MARKET_TZ
            )
        )

    return df.sort_index()


# ====================================================================
# CLEAN OHLCV
# ====================================================================

def clean_ticker_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    df = df.copy()

    df = flatten_columns(
        df
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required:

        if column not in df.columns:

            return pd.DataFrame()

    df = df[
        required
    ].copy()

    df = normalize_intraday_index(
        df
    )

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    )

    # Remove impossible rows.
    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
    ]

    return df.sort_index()


# ====================================================================
# DOWNLOAD 15M DATA
# ====================================================================

def download_intraday_batch(
    tickers: List[str],
    target_date: pd.Timestamp
) -> Dict[str, pd.DataFrame]:

    start_date = (
        target_date
        - pd.Timedelta(
            days=WARMUP_DAYS
        )
    )

    end_date = (
        target_date
        + pd.Timedelta(days=1)
    )

    raw = yf_download_quiet(

        tickers,

        start=start_date.strftime(
            "%Y-%m-%d"
        ),

        end=end_date.strftime(
            "%Y-%m-%d"
        ),

        interval=SIGNAL_INTERVAL,

        group_by="ticker",

        auto_adjust=False,

        prepost=False,

        progress=False,

        threads=True,

        ignore_tz=False,

        timeout=DOWNLOAD_TIMEOUT,

        multi_level_index=True,
    )

    if raw.empty:

        return {}

    results = {}

    if isinstance(
        raw.columns,
        pd.MultiIndex
    ):

        level0 = set(
            raw.columns
            .get_level_values(0)
        )

        for ticker in tickers:

            if ticker not in level0:

                continue

            try:

                ticker_df = raw[
                    ticker
                ].copy()

                ticker_df = clean_ticker_data(
                    ticker_df
                )

                if not ticker_df.empty:

                    results[
                        ticker
                    ] = ticker_df

            except Exception:

                continue

    elif len(tickers) == 1:

        ticker_df = clean_ticker_data(
            raw
        )

        if not ticker_df.empty:

            results[
                tickers[0]
            ] = ticker_df

    return results


# ====================================================================
# SESSION FILTER
# ====================================================================

def filter_session(
    df: pd.DataFrame,
    target_date: pd.Timestamp
) -> pd.DataFrame:

    if df.empty:

        return df

    date_str = (
        target_date.strftime(
            "%Y-%m-%d"
        )
    )

    session_open = pd.Timestamp(
        f"{date_str} {MARKET_OPEN}",
        tz=MARKET_TZ
    )

    session_close = pd.Timestamp(
        f"{date_str} {MARKET_CLOSE}",
        tz=MARKET_TZ
    )

    return df[
        (df.index >= session_open)
        &
        (df.index < session_close)
    ].sort_index()


# ====================================================================
# SESSION VWAP
# ====================================================================

def calculate_session_vwap(
    df: pd.DataFrame
) -> pd.Series:

    typical_price = (
        df["High"]
        + df["Low"]
        + df["Close"]
    ) / 3.0

    cumulative_pv = (
        typical_price
        * df["Volume"]
    ).cumsum()

    cumulative_volume = (
        df["Volume"]
        .cumsum()
    )

    return (
        cumulative_pv
        /
        cumulative_volume.replace(
            0,
            np.nan
        )
    )


# ====================================================================
# RSI
# ====================================================================

def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        -
        100 / (1 + rs)
    )

    # Handle zero-loss case.
    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    return rsi


# ====================================================================
# MACD
# ====================================================================

def calculate_macd(
    close: pd.Series
) -> Tuple[
    pd.Series,
    pd.Series,
    pd.Series
]:

    ema_fast = (
        close
        .ewm(
            span=MACD_FAST,
            adjust=False,
            min_periods=MACD_FAST
        )
        .mean()
    )

    ema_slow = (
        close
        .ewm(
            span=MACD_SLOW,
            adjust=False,
            min_periods=MACD_SLOW
        )
        .mean()
    )

    macd = (
        ema_fast
        - ema_slow
    )

    signal = (
        macd
        .ewm(
            span=MACD_SIGNAL,
            adjust=False,
            min_periods=MACD_SIGNAL
        )
        .mean()
    )

    histogram = (
        macd
        - signal
    )

    return (
        macd,
        signal,
        histogram
    )


# ====================================================================
# ATR
# ====================================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD
) -> pd.Series:

    previous_close = (
        df["Close"]
        .shift(1)
    )

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = (
        df["High"]
        - previous_close
    ).abs()

    tr3 = (
        df["Low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1
    ).max(
        axis=1
    )

    return (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )


# ====================================================================
# ADD HISTORICAL INDICATORS
# ====================================================================

def add_historical_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["EMA9"] = (
        df["Close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
            min_periods=EMA_FAST
        )
        .mean()
    )

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
            min_periods=EMA_SLOW
        )
        .mean()
    )

    df["RSI"] = calculate_rsi(
        df["Close"]
    )

    (
        df["MACD"],
        df["MACD_Signal"],
        df["MACD_Hist"]
    ) = calculate_macd(
        df["Close"]
    )

    df["ATR"] = calculate_atr(
        df
    )

    df["ATR_Pct"] = (
        df["ATR"]
        / df["Close"]
    ) * 100

    return df


# ====================================================================
# CANDLE FEATURES
# ====================================================================

def add_candle_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    candle_range = (
        df["High"]
        - df["Low"]
    ).replace(
        0,
        np.nan
    )

    body = (
        df["Close"]
        - df["Open"]
    )

    df["Candle_Range"] = (
        candle_range
    )

    df["Body"] = (
        body.abs()
    )

    df["Body_Pct"] = (
        body
        / df["Open"]
    ) * 100

    df["Candle_Strength"] = (
        df["Body"]
        / candle_range
    )

    df["Upper_Wick"] = (
        df["High"]
        -
        df[
            ["Open", "Close"]
        ].max(axis=1)
    )

    df["Lower_Wick"] = (
        df[
            ["Open", "Close"]
        ].min(axis=1)
        -
        df["Low"]
    )

    df["Upper_Wick_Pct"] = (
        df["Upper_Wick"]
        /
        candle_range
    )

    df["Lower_Wick_Pct"] = (
        df["Lower_Wick"]
        /
        candle_range
    )

    return df


# ====================================================================
# ADD ALL HISTORICAL FEATURES
# ====================================================================

def add_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = add_historical_indicators(
        df
    )

    df = add_candle_features(
        df
    )

    df["Session_Date"] = (
        df.index.date
    )

    df["Bar_Time"] = (
        df.index.strftime(
            "%H:%M"
        )
    )

    return df


# ====================================================================
# RELATIVE VOLUME
# ====================================================================

def calculate_rvol_for_target(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    bar_time: str,
    current_volume: float
) -> float:

    if (
        df.empty
        or current_volume <= 0
    ):

        return np.nan

    target_day = (
        target_date.date()
    )

    historical = df[
        (
            df["Session_Date"]
            < target_day
        )
        &
        (
            df["Bar_Time"]
            == bar_time
        )
    ].copy()

    if historical.empty:

        return np.nan

    historical = (
        historical
        .sort_index()
        .tail(
            RVOL_LOOKBACK_DAYS
        )
    )

    volumes = (
        historical["Volume"]
        .replace(
            0,
            np.nan
        )
        .dropna()
    )

    if (
        len(volumes)
        < MIN_RVOL_OBSERVATIONS
    ):

        return np.nan

    baseline = float(
        volumes.median()
    )

    if baseline <= 0:

        return np.nan

    return (
        current_volume
        / baseline
    )


# ====================================================================
# NIFTY DATA
# ====================================================================

def download_nifty_data(
    target_date: pd.Timestamp
) -> pd.DataFrame:

    start_date = (
        target_date
        - pd.Timedelta(
            days=WARMUP_DAYS
        )
    )

    end_date = (
        target_date
        + pd.Timedelta(days=1)
    )

    raw = yf_download_quiet(

        NIFTY50_TICKER,

        start=start_date.strftime(
            "%Y-%m-%d"
        ),

        end=end_date.strftime(
            "%Y-%m-%d"
        ),

        interval=SIGNAL_INTERVAL,

        auto_adjust=False,

        prepost=False,

        progress=False,

        threads=False,

        ignore_tz=False,

        timeout=DOWNLOAD_TIMEOUT,
    )

    return clean_ticker_data(
        raw
    )


# ====================================================================
# BENCHMARK RETURN
# ====================================================================

def calculate_market_returns(
    nifty: pd.DataFrame,
    target_date: pd.Timestamp
) -> Tuple[float, float]:

    session = filter_session(
        nifty,
        target_date
    )

    if len(session) < 2:

        return (
            np.nan,
            np.nan
        )

    opening = session.iloc[0]

    confirmation = session.iloc[1]

    opening_return = (
        (
            float(
                opening["Close"]
            )
            -
            float(
                opening["Open"]
            )
        )
        /
        float(
            opening["Open"]
        )
    ) * 100

    total_return = (
        (
            float(
                confirmation["Close"]
            )
            -
            float(
                opening["Open"]
            )
        )
        /
        float(
            opening["Open"]
        )
    ) * 100

    return (
        opening_return,
        total_return
    )


# ====================================================================
# CLAMP
# ====================================================================

def clamp(
    value: float,
    low: float,
    high: float
) -> float:

    if not np.isfinite(value):

        return 0.0

    return float(
        np.clip(
            value,
            low,
            high
        )
    )


# ====================================================================
# CONTINUOUS SCORE HELPERS
# ====================================================================

def bearish_move_score(
    value_pct: float,
    scale: float
) -> float:

    if not np.isfinite(
        value_pct
    ):

        return 0.0

    return clamp(
        (-value_pct / scale),
        0,
        1
    )


def bullish_move_score(
    value_pct: float,
    scale: float
) -> float:

    if not np.isfinite(
        value_pct
    ):

        return 0.0

    return clamp(
        value_pct / scale,
        0,
        1
    )


# ====================================================================
# EXHAUSTION MODEL
# ====================================================================

def calculate_short_exhaustion(
    opening_pct: float,
    vwap_pct: float,
    ema20_pct: float,
    rsi: float,
    lower_wick_pct: float,
    momentum_deceleration: bool
) -> Tuple[
    float,
    List[str]
]:

    penalty = 0.0

    flags: List[str] = []

    # ------------------------------------------------------------
    # Opening collapse
    # ------------------------------------------------------------

    if opening_pct <= -MAX_OPENING_COLLAPSE:

        penalty += 12

        flags.append(
            "opening_collapse"
        )

    elif opening_pct <= -2.25:

        penalty += 6

        flags.append(
            "large_opening_move"
        )

    # ------------------------------------------------------------
    # VWAP extension
    # ------------------------------------------------------------

    if vwap_pct <= -MAX_SHORT_VWAP_DISTANCE:

        penalty += 12

        flags.append(
            "far_below_vwap"
        )

    elif vwap_pct <= -2.0:

        penalty += 6

        flags.append(
            "extended_vwap"
        )

    # ------------------------------------------------------------
    # EMA extension
    # ------------------------------------------------------------

    if ema20_pct <= -MAX_SHORT_EMA20_DISTANCE:

        penalty += 10

        flags.append(
            "far_below_ema20"
        )

    elif ema20_pct <= -2.5:

        penalty += 5

        flags.append(
            "extended_ema20"
        )

    # ------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------

    if np.isfinite(rsi):

        if rsi <= RSI_EXTREME_OVERSOLD:

            penalty += 15

            flags.append(
                "extreme_oversold"
            )

        elif rsi <= RSI_OVERSOLD:

            penalty += 8

            flags.append(
                "oversold"
            )

    # ------------------------------------------------------------
    # Lower wick
    # ------------------------------------------------------------

    if np.isfinite(
        lower_wick_pct
    ):

        if lower_wick_pct >= 0.45:

            penalty += 8

            flags.append(
                "large_lower_wick"
            )

        elif lower_wick_pct >= 0.30:

            penalty += 4

            flags.append(
                "lower_wick"
            )

    # ------------------------------------------------------------
    # Deceleration
    # ------------------------------------------------------------

    if momentum_deceleration:

        penalty += 8

        flags.append(
            "momentum_deceleration"
        )

    return (
        penalty,
        flags
    )


# ====================================================================
# ANALYSE 09:45
# ====================================================================

def analyse_at_0945(
    ticker: str,
    full_df: pd.DataFrame,
    target_date: pd.Timestamp,
    nifty_opening_return: float,
    nifty_total_return: float
) -> Optional[dict]:

    if full_df.empty:

        return None

    # ------------------------------------------------------------
    # Historical indicators already include warm-up data.
    # ------------------------------------------------------------

    df = add_features(
        full_df
    )

    session = filter_session(
        df,
        target_date
    )

    if len(session) < 2:

        return None

    opening = session.iloc[0]

    confirmation = session.iloc[1]

    # ============================================================
    # PRICE MOVEMENTS
    # ============================================================

    opening_price = float(
        opening["Open"]
    )

    opening_close = float(
        opening["Close"]
    )

    confirmation_open = float(
        confirmation["Open"]
    )

    decision_price = float(
        confirmation["Close"]
    )

    opening_pct = (
        (
            opening_close
            -
            opening_price
        )
        /
        opening_price
    ) * 100

    confirmation_pct = (
        (
            decision_price
            -
            confirmation_open
        )
        /
        confirmation_open
    ) * 100

    total_pct = (
        (
            decision_price
            -
            opening_price
        )
        /
        opening_price
    ) * 100

    # ============================================================
    # MARKET RELATIVE STRENGTH
    # ============================================================

    relative_opening_strength = (
        opening_pct
        -
        nifty_opening_return
        if np.isfinite(
            nifty_opening_return
        )
        else np.nan
    )

    relative_total_strength = (
        total_pct
        -
        nifty_total_return
        if np.isfinite(
            nifty_total_return
        )
        else np.nan
    )

    # ============================================================
    # INDICATORS
    # ============================================================

    vwap = calculate_session_vwap(
        session
    )

    vwap_value = float(
        vwap.iloc[1]
    )

    ema9 = float(
        confirmation["EMA9"]
    )

    ema20 = float(
        confirmation["EMA20"]
    )

    rsi = float(
        confirmation["RSI"]
    ) if pd.notna(
        confirmation["RSI"]
    ) else np.nan

    macd = float(
        confirmation["MACD"]
    ) if pd.notna(
        confirmation["MACD"]
    ) else np.nan

    macd_signal = float(
        confirmation["MACD_Signal"]
    ) if pd.notna(
        confirmation["MACD_Signal"]
    ) else np.nan

    macd_hist = float(
        confirmation["MACD_Hist"]
    ) if pd.notna(
        confirmation["MACD_Hist"]
    ) else np.nan

    atr = float(
        confirmation["ATR"]
    ) if pd.notna(
        confirmation["ATR"]
    ) else np.nan

    atr_pct = float(
        confirmation["ATR_Pct"]
    ) if pd.notna(
        confirmation["ATR_Pct"]
    ) else np.nan

    # ============================================================
    # PRICE DISTANCES
    # ============================================================

    vwap_pct = (
        (
            decision_price
            -
            vwap_value
        )
        /
        vwap_value
    ) * 100

    ema20_pct = (
        (
            decision_price
            -
            ema20
        )
        /
        ema20
    ) * 100

    ema9_pct = (
        (
            decision_price
            -
            ema9
        )
        /
        ema9
    ) * 100

    ema_spread_pct = (
        (
            ema9
            -
            ema20
        )
        /
        ema20
    ) * 100

    # ============================================================
    # ATR NORMALIZATION
    # ============================================================

    if (
        np.isfinite(atr)
        and atr > 0
    ):

        normalized_total_move = (
            (
                decision_price
                -
                opening_price
            )
            /
            atr
        )

    else:

        normalized_total_move = np.nan

    # ============================================================
    # STRUCTURE
    # ============================================================

    opening_high = float(
        opening["High"]
    )

    opening_low = float(
        opening["Low"]
    )

    confirmation_high = float(
        confirmation["High"]
    )

    confirmation_low = float(
        confirmation["Low"]
    )

    lower_low = (
        confirmation_low
        <
        opening_low
    )

    higher_high = (
        confirmation_high
        >
        opening_high
    )

    # ============================================================
    # CANDLE
    # ============================================================

    confirmation_range = float(
        confirmation["Candle_Range"]
    )

    candle_strength = float(
        confirmation["Candle_Strength"]
    ) if pd.notna(
        confirmation["Candle_Strength"]
    ) else np.nan

    lower_wick_pct = float(
        confirmation["Lower_Wick_Pct"]
    ) if pd.notna(
        confirmation["Lower_Wick_Pct"]
    ) else np.nan

    upper_wick_pct = float(
        confirmation["Upper_Wick_Pct"]
    ) if pd.notna(
        confirmation["Upper_Wick_Pct"]
    ) else np.nan

    bearish_confirmation = (
        decision_price
        <
        confirmation_open
    )

    bullish_confirmation = (
        decision_price
        >
        confirmation_open
    )

    # ============================================================
    # RANGE EXPANSION
    # ============================================================

    opening_range = (
        opening_high
        -
        opening_low
    )

    range_expansion = (
        confirmation_range
        /
        opening_range
        if opening_range > 0
        else np.nan
    )

    # ============================================================
    # RELATIVE VOLUME
    # ============================================================

    opening_rvol = (
        calculate_rvol_for_target(
            df,
            target_date,
            "09:15",
            float(
                opening["Volume"]
            )
        )
    )

    confirmation_rvol = (
        calculate_rvol_for_target(
            df,
            target_date,
            "09:30",
            float(
                confirmation["Volume"]
            )
        )
    )

    # ============================================================
    # MOMENTUM DECELERATION
    # ============================================================

    momentum_deceleration = (

        opening_pct < -0.75

        and confirmation_pct
        > opening_pct * 0.55

        and confirmation_pct > -0.25
    )

    momentum_acceleration = (

        opening_pct < 0

        and confirmation_pct
        < opening_pct * 0.75
    )

    # ============================================================
    # CONTINUATION
    # ============================================================

    short_continuation = (

        opening_pct < -0.25

        and confirmation_pct < -0.10

        and lower_low
    )

    long_continuation = (

        opening_pct > 0.25

        and confirmation_pct > 0.10

        and higher_high
    )

    # ============================================================
    # REVERSAL
    # ============================================================

    short_reversal = (

        opening_pct < -0.75

        and confirmation_pct > 0.30

        and not lower_low
    )

    long_reversal = (

        opening_pct > 0.75

        and confirmation_pct < -0.30

        and not higher_high
    )

    # ============================================================
    # EXHAUSTION
    # ============================================================

    exhaustion_penalty, exhaustion_flags = (
        calculate_short_exhaustion(

            opening_pct=opening_pct,

            vwap_pct=vwap_pct,

            ema20_pct=ema20_pct,

            rsi=rsi,

            lower_wick_pct=lower_wick_pct,

            momentum_deceleration=(
                momentum_deceleration
            ),
        )
    )

    # ============================================================
    # SHORT SCORE
    # ============================================================

    short_score = 0.0

    # ------------------------------------------------------------
    # 1. Opening momentum
    # ------------------------------------------------------------

    short_score += (
        12
        *
        bearish_move_score(
            opening_pct,
            1.50
        )
    )

    # ------------------------------------------------------------
    # 2. Confirmation momentum
    # ------------------------------------------------------------

    short_score += (
        12
        *
        bearish_move_score(
            confirmation_pct,
            1.00
        )
    )

    # ------------------------------------------------------------
    # 3. Market-relative weakness
    # ------------------------------------------------------------

    short_score += (
        12
        *
        bearish_move_score(
            relative_total_strength,
            1.50
        )
    )

    # ------------------------------------------------------------
    # 4. VWAP location
    # ------------------------------------------------------------

    short_score += (
        10
        *
        clamp(
            -vwap_pct / 2.0,
            0,
            1
        )
    )

    # ------------------------------------------------------------
    # 5. EMA structure
    # ------------------------------------------------------------

    ema_structure = 0.0

    if (
        decision_price
        < ema20
    ):

        ema_structure += 0.50

    if (
        ema9
        < ema20
    ):

        ema_structure += 0.50

    short_score += (
        10
        * ema_structure
    )

    # ------------------------------------------------------------
    # 6. MACD structure
    # ------------------------------------------------------------

    macd_structure = 0.0

    if (
        np.isfinite(macd)
        and np.isfinite(macd_signal)
        and macd < macd_signal
    ):

        macd_structure += 0.60

    if (
        np.isfinite(macd_hist)
        and macd_hist < 0
    ):

        macd_structure += 0.40

    short_score += (
        8
        *
        min(
            1.0,
            macd_structure
        )
    )

    # ------------------------------------------------------------
    # 7. Volume
    # ------------------------------------------------------------

    volume_strength = 0.0

    if np.isfinite(
        opening_rvol
    ):

        volume_strength += (
            clamp(
                (
                    opening_rvol
                    - 1.0
                )
                /
                1.5,
                0,
                1
            )
            * 0.40
        )

    if np.isfinite(
        confirmation_rvol
    ):

        volume_strength += (
            clamp(
                (
                    confirmation_rvol
                    - 1.0
                )
                /
                1.5,
                0,
                1
            )
            * 0.60
        )

    short_score += (
        8
        *
        min(
            1.0,
            volume_strength
        )
    )

    # ------------------------------------------------------------
    # 8. Structure
    # ------------------------------------------------------------

    structure_score = 0.0

    if lower_low:

        structure_score += 0.60

    if bearish_confirmation:

        structure_score += 0.20

    if (
        np.isfinite(
            lower_wick_pct
        )
        and lower_wick_pct < 0.20
    ):

        structure_score += 0.20

    short_score += (
        10
        *
        min(
            1.0,
            structure_score
        )
    )

    # ------------------------------------------------------------
    # 9. Range expansion
    # ------------------------------------------------------------

    if np.isfinite(
        range_expansion
    ):

        short_score += (
            5
            *
            clamp(
                (
                    range_expansion
                    - 0.75
                )
                /
                0.75,
                0,
                1
            )
        )

    # ------------------------------------------------------------
    # 10. ATR-normalized momentum
    # ------------------------------------------------------------

    if np.isfinite(
        normalized_total_move
    ):

        short_score += (
            5
            *
            clamp(
                -normalized_total_move
                / 2.0,
                0,
                1
            )
        )

    # ------------------------------------------------------------
    # Continuation bonus
    # ------------------------------------------------------------

    if short_continuation:

        short_score += 6

    # ------------------------------------------------------------
    # Acceleration bonus
    # ------------------------------------------------------------

    if momentum_acceleration:

        short_score += 4

    # ------------------------------------------------------------
    # Reversal penalty
    # ------------------------------------------------------------

    if short_reversal:

        short_score -= 25

    # ------------------------------------------------------------
    # Exhaustion penalty
    # ------------------------------------------------------------

    short_score -= exhaustion_penalty

    # ============================================================
    # RSI BALANCE
    # ============================================================

    if np.isfinite(rsi):

        # Prefer bearish momentum that is not completely exhausted.

        if 32 <= rsi <= 55:

            short_score += 5

        elif 28 <= rsi < 32:

            short_score += 2

        elif rsi < 22:

            short_score -= 5

    # ============================================================
    # LONG SCORE
    # ============================================================

    long_score = 0.0

    # Opening momentum
    long_score += (
        12
        *
        bullish_move_score(
            opening_pct,
            1.50
        )
    )

    # Confirmation momentum
    long_score += (
        12
        *
        bullish_move_score(
            confirmation_pct,
            1.00
        )
    )

    # Relative strength
    long_score += (
        12
        *
        bullish_move_score(
            relative_total_strength,
            1.50
        )
    )

    # VWAP
    long_score += (
        10
        *
        clamp(
            vwap_pct / 2.0,
            0,
            1
        )
    )

    # EMA structure
    long_ema_structure = 0.0

    if (
        decision_price
        > ema20
    ):

        long_ema_structure += 0.50

    if (
        ema9
        > ema20
    ):

        long_ema_structure += 0.50

    long_score += (
        10
        * long_ema_structure
    )

    # MACD
    long_macd_structure = 0.0

    if (
        np.isfinite(macd)
        and np.isfinite(macd_signal)
        and macd > macd_signal
    ):

        long_macd_structure += 0.60

    if (
        np.isfinite(macd_hist)
        and macd_hist > 0
    ):

        long_macd_structure += 0.40

    long_score += (
        8
        *
        min(
            1.0,
            long_macd_structure
        )
    )

    # Volume
    long_volume_strength = 0.0

    if np.isfinite(
        opening_rvol
    ):

        long_volume_strength += (
            clamp(
                (
                    opening_rvol
                    - 1.0
                )
                /
                1.5,
                0,
                1
            )
            * 0.40
        )

    if np.isfinite(
        confirmation_rvol
    ):

        long_volume_strength += (
            clamp(
                (
                    confirmation_rvol
                    - 1.0
                )
                /
                1.5,
                0,
                1
            )
            * 0.60
        )

    long_score += (
        8
        *
        min(
            1.0,
            long_volume_strength
        )
    )

    # Structure
    long_structure = 0.0

    if higher_high:

        long_structure += 0.60

    if bullish_confirmation:

        long_structure += 0.20

    if (
        np.isfinite(
            upper_wick_pct
        )
        and upper_wick_pct < 0.20
    ):

        long_structure += 0.20

    long_score += (
        10
        *
        min(
            1.0,
            long_structure
        )
    )

    # Range expansion
    if np.isfinite(
        range_expansion
    ):

        long_score += (
            5
            *
            clamp(
                (
                    range_expansion
                    - 0.75
                )
                /
                0.75,
                0,
                1
            )
        )

    # ATR normalized momentum
    if np.isfinite(
        normalized_total_move
    ):

        long_score += (
            5
            *
            clamp(
                normalized_total_move
                / 2.0,
                0,
                1
            )
        )

    if long_continuation:

        long_score += 6

    if long_reversal:

        long_score -= 25

    # RSI
    if np.isfinite(rsi):

        if 45 <= rsi <= 70:

            long_score += 5

        elif rsi > 78:

            long_score -= 6

    # ============================================================
    # FINAL SCORE
    # ============================================================

    short_score = clamp(
        short_score,
        0,
        100
    )

    long_score = clamp(
        long_score,
        0,
        100
    )

    # ============================================================
    # TURNOVER
    # ============================================================

    turnover_cr = (
        float(
            confirmation["Volume"]
        )
        *
        decision_price
        /
        10_000_000
    )

    # ============================================================
    # DECISION
    # ============================================================

    signal = "AVOID"

    conviction = "NEUTRAL"

    score = max(
        short_score,
        long_score
    )

    if (
        short_score
        >= SHORT_TRADEABLE_SCORE

        and short_score
        >= long_score

        and total_pct
        <= -MIN_SHORT_TOTAL_MOVE

        and opening_pct
        <= -MIN_OPENING_SHORT_MOVE

        and not short_reversal
    ):

        signal = "SHORT"

        score = short_score

        if (
            short_score
            >= SHORT_HIGH_CONVICTION_SCORE
        ):

            conviction = "HIGH"

        else:

            conviction = "TRADEABLE"

    elif (
        long_score
        >= LONG_TRADEABLE_SCORE

        and long_score
        > short_score

        and total_pct
        >= MIN_LONG_TOTAL_MOVE

        and opening_pct
        >= MIN_OPENING_LONG_MOVE

        and not long_reversal
    ):

        signal = "LONG"

        score = long_score

        if (
            long_score
            >= LONG_HIGH_CONVICTION_SCORE
        ):

            conviction = "HIGH"

        else:

            conviction = "TRADEABLE"

    # ============================================================
    # DATA QUALITY
    # ============================================================

    indicator_ready = all(
        np.isfinite(x)
        for x in [
            ema9,
            ema20,
            rsi,
            atr,
        ]
    )

    if not indicator_ready:

        return None

    return {

        "Ticker":
            ticker.replace(
                ".NS",
                ""
            ),

        "Signal":
            signal,

        "Conviction":
            conviction,

        "Score":
            round(
                score,
                2
            ),

        "Short Score":
            round(
                short_score,
                2
            ),

        "Long Score":
            round(
                long_score,
                2
            ),

        "Opening %":
            opening_pct,

        "09:30-09:45 %":
            confirmation_pct,

        "Total %":
            total_pct,

        "NIFTY Opening %":
            nifty_opening_return,

        "NIFTY Total %":
            nifty_total_return,

        "Relative Opening %":
            relative_opening_strength,

        "Relative Total %":
            relative_total_strength,

        "Price":
            decision_price,

        "VWAP %":
            vwap_pct,

        "EMA20 %":
            ema20_pct,

        "EMA9 %":
            ema9_pct,

        "EMA Spread %":
            ema_spread_pct,

        "ATR %":
            atr_pct,

        "ATR Move":
            normalized_total_move,

        "Opening RVOL":
            opening_rvol,

        "Confirmation RVOL":
            confirmation_rvol,

        "RSI":
            rsi,

        "MACD Hist":
            macd_hist,

        "MACD":
            (
                "Bullish"
                if macd > macd_signal
                else "Bearish"
            ),

        "Turnover Cr":
            turnover_cr,

        "Opening High":
            opening_high,

        "Opening Low":
            opening_low,

        "Lower Low":
            lower_low,

        "Higher High":
            higher_high,

        "Range Expansion":
            range_expansion,

        "Candle Strength":
            candle_strength,

        "Lower Wick":
            lower_wick_pct,

        "Upper Wick":
            upper_wick_pct,

        "Exhaustion Penalty":
            exhaustion_penalty,

        "Exhaustion Flags":
            ",".join(
                exhaustion_flags
            )
            if exhaustion_flags
            else "",

        "Decision Timestamp":
            confirmation.name,

        "_Data":
            df,
    }


# ====================================================================
# 1-MINUTE DATA
# ====================================================================

def download_1m_data(
    ticker: str,
    target_date: pd.Timestamp
) -> pd.DataFrame:

    start_date = target_date

    end_date = (
        target_date
        + pd.Timedelta(days=1)
    )

    raw = yf_download_quiet(

        ticker,

        start=start_date.strftime(
            "%Y-%m-%d"
        ),

        end=end_date.strftime(
            "%Y-%m-%d"
        ),

        interval="1m",

        auto_adjust=False,

        prepost=False,

        progress=False,

        threads=False,

        ignore_tz=False,

        timeout=DOWNLOAD_TIMEOUT,
    )

    return clean_ticker_data(
        raw
    )


# ====================================================================
# OUTCOME USING FINE DATA
# ====================================================================

def evaluate_from_bars(
    signal: str,
    decision_price: float,
    future: pd.DataFrame
) -> dict:

    if future.empty:

        return {

            "MFE":
                np.nan,

            "MAE":
                np.nan,

            "Outcome":
                "NO_DATA",

            "Target":
                np.nan,

            "Stop":
                np.nan,

            "Gross Return":
                np.nan,

            "Net Return":
                np.nan,

            "Bars Held":
                np.nan,
        }

    if signal == "LONG":

        target = (
            decision_price
            *
            (
                1
                +
                TARGET_PCT / 100
            )
        )

        stop = (
            decision_price
            *
            (
                1
                -
                STOP_PCT / 100
            )
        )

        mfe = (
            (
                future["High"].max()
                -
                decision_price
            )
            /
            decision_price
        ) * 100

        mae = (
            (
                future["Low"].min()
                -
                decision_price
            )
            /
            decision_price
        ) * 100

        outcome = "OPEN"

        exit_price = np.nan

        bars_held = 0

        for _, row in future.iterrows():

            bars_held += 1

            hit_stop = (
                row["Low"]
                <= stop
            )

            hit_target = (
                row["High"]
                >= target
            )

            if (
                hit_stop
                and hit_target
            ):

                # Still ambiguous if both occur in the
                # same 1-minute candle.
                #
                # Conservative assumption.

                outcome = "STOP"

                exit_price = stop

                break

            if hit_stop:

                outcome = "STOP"

                exit_price = stop

                break

            if hit_target:

                outcome = "TARGET"

                exit_price = target

                break

        if outcome == "OPEN":

            exit_price = float(
                future.iloc[-1]["Close"]
            )

            if (
                exit_price
                >
                decision_price
            ):

                outcome = "PROFIT"

            else:

                outcome = "LOSS"

    else:

        target = (
            decision_price
            *
            (
                1
                -
                TARGET_PCT / 100
            )
        )

        stop = (
            decision_price
            *
            (
                1
                +
                STOP_PCT / 100
            )
        )

        mfe = (
            (
                decision_price
                -
                future["Low"].min()
            )
            /
            decision_price
        ) * 100

        mae = (
            (
                future["High"].max()
                -
                decision_price
            )
            /
            decision_price
        ) * 100

        outcome = "OPEN"

        exit_price = np.nan

        bars_held = 0

        for _, row in future.iterrows():

            bars_held += 1

            hit_stop = (
                row["High"]
                >= stop
            )

            hit_target = (
                row["Low"]
                <= target
            )

            if (
                hit_stop
                and hit_target
            ):

                outcome = "STOP"

                exit_price = stop

                break

            if hit_stop:

                outcome = "STOP"

                exit_price = stop

                break

            if hit_target:

                outcome = "TARGET"

                exit_price = target

                break

        if outcome == "OPEN":

            exit_price = float(
                future.iloc[-1]["Close"]
            )

            if (
                exit_price
                <
                decision_price
            ):

                outcome = "PROFIT"

            else:

                outcome = "LOSS"

    # ============================================================
    # Gross return
    # ============================================================

    if signal == "LONG":

        gross_return = (
            (
                exit_price
                -
                decision_price
            )
            /
            decision_price
        ) * 100

    else:

        gross_return = (
            (
                decision_price
                -
                exit_price
            )
            /
            decision_price
        ) * 100

    net_return = (
        gross_return
        -
        COST_MODEL.total_cost_pct
    )

    return {

        "MFE":
            float(mfe),

        "MAE":
            float(mae),

        "Outcome":
            outcome,

        "Target":
            target,

        "Stop":
            stop,

        "Gross Return":
            float(gross_return),

        "Net Return":
            float(net_return),

        "Bars Held":
            bars_held,
    }


# ====================================================================
# HISTORICAL OUTCOME
# ====================================================================

def calculate_historical_outcome(
    analysis: dict,
    full_df: pd.DataFrame,
    target_date: pd.Timestamp
) -> dict:

    signal = analysis[
        "Signal"
    ]

    if signal not in (
        "LONG",
        "SHORT"
    ):

        return {

            "MFE":
                np.nan,

            "MAE":
                np.nan,

            "Outcome":
                "N/A",

            "Target":
                np.nan,

            "Stop":
                np.nan,

            "Gross Return":
                np.nan,

            "Net Return":
                np.nan,

            "Bars Held":
                np.nan,
        }

    decision_timestamp = (
        analysis[
            "Decision Timestamp"
        ]
    )

    decision_price = float(
        analysis["Price"]
    )

    # ------------------------------------------------------------
    # Prefer 1-minute data.
    # ------------------------------------------------------------

    if USE_1M_EVALUATION:

        ticker = (
            analysis["Ticker"]
            + ".NS"
        )

        one_minute = (
            download_1m_data(
                ticker,
                target_date
            )
        )

        if not one_minute.empty:

            future = one_minute[
                one_minute.index
                >
                decision_timestamp
            ].copy()

            if not future.empty:

                return evaluate_from_bars(
                    signal,
                    decision_price,
                    future
                )

    # ------------------------------------------------------------
    # 15-minute fallback.
    # ------------------------------------------------------------

    if not ALLOW_15M_FALLBACK:

        return {

            "MFE":
                np.nan,

            "MAE":
                np.nan,

            "Outcome":
                "NO_1M_DATA",

            "Target":
                np.nan,

            "Stop":
                np.nan,

            "Gross Return":
                np.nan,

            "Net Return":
                np.nan,

            "Bars Held":
                np.nan,
        }

    future = full_df[
        full_df.index
        >
        decision_timestamp
    ].copy()

    if future.empty:

        return {

            "MFE":
                np.nan,

            "MAE":
                np.nan,

            "Outcome":
                "NO_DATA",

            "Target":
                np.nan,

            "Stop":
                np.nan,

            "Gross Return":
                np.nan,

            "Net Return":
                np.nan,

            "Bars Held":
                np.nan,
        }

    return evaluate_from_bars(
        signal,
        decision_price,
        future
    )


# ====================================================================
# DISPLAY COLUMNS
# ====================================================================

DISPLAY_COLUMNS = [

    "Rank",

    "Ticker",

    "Signal",

    "Conviction",

    "Score",

    "Opening %",

    "09:30-09:45 %",

    "Total %",

    "Relative Total %",

    "Price",

    "VWAP %",

    "EMA20 %",

    "Opening RVOL",

    "Confirmation RVOL",

    "RSI",

    "MACD",

    "Turnover Cr",
]


# ====================================================================
# DISPLAY FORMAT
# ====================================================================

def format_display(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df.empty:

        return df

    out = df.copy()

    percentage_columns = [

        "Opening %",

        "09:30-09:45 %",

        "Total %",

        "Relative Total %",

        "VWAP %",

        "EMA20 %",
    ]

    for column in percentage_columns:

        out[column] = out[
            column
        ].map(

            lambda x:
            (
                f"{x:+.2f}%"
                if pd.notna(x)
                else "-"
            )
        )

    out["Price"] = out[
        "Price"
    ].map(

        lambda x:
        (
            f"₹{x:,.2f}"
            if pd.notna(x)
            else "-"
        )
    )

    for column in [
        "Opening RVOL",
        "Confirmation RVOL",
    ]:

        out[column] = out[
            column
        ].map(

            lambda x:
            (
                f"{x:.2f}x"
                if pd.notna(x)
                else "-"
            )
        )

    out["RSI"] = out[
        "RSI"
    ].map(

        lambda x:
        (
            f"{x:.1f}"
            if pd.notna(x)
            else "-"
        )
    )

    out["Turnover Cr"] = out[
        "Turnover Cr"
    ].map(

        lambda x:
        (
            f"{x:.1f}"
            if pd.notna(x)
            else "-"
        )
    )

    out["Score"] = out[
        "Score"
    ].map(

        lambda x:
        (
            f"{x:.0f}"
            if pd.notna(x)
            else "-"
        )
    )

    return out


# ====================================================================
# PRINT TABLE
# ====================================================================

def print_table(
    title: str,
    df: pd.DataFrame
) -> None:

    print()

    print(
        "=" * 150
    )

    print(title)

    print(
        "=" * 150
    )

    if df.empty:

        print(
            "No stocks matched."
        )

        return

    display = df[
        DISPLAY_COLUMNS
    ].copy()

    display = format_display(
        display
    )

    print(
        display.to_string(
            index=False
        )
    )


# ====================================================================
# SHORT DETAIL
# ====================================================================

def print_short_detail(
    df: pd.DataFrame
) -> None:

    if df.empty:

        return

    print()

    print(
        "=" * 110
    )

    print(
        "SHORT SETUP QUALITY"
    )

    print(
        "=" * 110
    )

    for _, row in df.head(
        TOP_SHORTS_TO_SHOW
    ).iterrows():

        rsi_text = (
            f"{row['RSI']:.1f}"
            if pd.notna(
                row["RSI"]
            )
            else "-"
        )

        print(

            f"{row['Ticker']:12s} | "
            f"Score {row['Score']:5.1f} | "
            f"Short {row['Short Score']:5.1f} | "
            f"RSI {rsi_text:>5s} | "
            f"RVOL "
            f"{row['Confirmation RVOL']:.2f}x"
            if pd.notna(
                row["Confirmation RVOL"]
            )
            else
            f"{row['Ticker']:12s} | "
            f"Score {row['Score']:5.1f}"
        )

        flags = row.get(
            "Exhaustion Flags",
            ""
        )

        if flags:

            print(
                f"    Exhaustion: "
                f"{flags}"
            )

        else:

            print(
                "    Exhaustion: none"
            )


# ====================================================================
# EQUITY CURVE
# ====================================================================

def calculate_drawdown(
    returns: pd.Series
) -> float:

    if returns.empty:

        return np.nan

    equity = (
        1
        +
        returns.fillna(
            0
        ) / 100
    ).cumprod()

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        /
        running_max
        - 1
    )

    return float(
        drawdown.min()
        * 100
    )


# ====================================================================
# HISTORICAL RESULTS
# ====================================================================

def print_historical_results(
    df: pd.DataFrame
) -> None:

    print()

    print(
        "=" * 140
    )

    print(
        "HISTORICAL POST-09:45 EVALUATION"
    )

    print(
        "=" * 140
    )

    if df.empty:

        print(
            "No LONG/SHORT signals to evaluate."
        )

        return

    columns = [

        "Ticker",

        "Signal",

        "Score",

        "Opening %",

        "09:30-09:45 %",

        "Total %",

        "MFE",

        "MAE",

        "Outcome",

        "Gross Return",

        "Net Return",
    ]

    display = df[
        columns
    ].copy()

    for column in [

        "Opening %",

        "09:30-09:45 %",

        "Total %",

        "MFE",

        "MAE",

        "Gross Return",

        "Net Return",
    ]:

        display[column] = display[
            column
        ].map(

            lambda x:
            (
                f"{x:+.2f}%"
                if pd.notna(x)
                else "-"
            )
        )

    display["Score"] = display[
        "Score"
    ].map(

        lambda x:
        (
            f"{x:.0f}"
            if pd.notna(x)
            else "-"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    # ============================================================
    # STATISTICS
    # ============================================================

    evaluated = df[
        df["Net Return"].notna()
    ].copy()

    if evaluated.empty:

        return

    total = len(
        evaluated
    )

    wins = (
        evaluated[
            "Net Return"
        ]
        > 0
    ).sum()

    losses = (
        evaluated[
            "Net Return"
        ]
        <= 0
    ).sum()

    win_rate = (
        wins / total * 100
        if total > 0
        else np.nan
    )

    avg_return = (
        evaluated[
            "Net Return"
        ].mean()
    )

    avg_win = (
        evaluated.loc[
            evaluated["Net Return"] > 0,
            "Net Return"
        ].mean()
    )

    avg_loss = (
        evaluated.loc[
            evaluated["Net Return"] <= 0,
            "Net Return"
        ].mean()
    )

    gross_profit = (
        evaluated.loc[
            evaluated["Net Return"] > 0,
            "Net Return"
        ].sum()
    )

    gross_loss = abs(
        evaluated.loc[
            evaluated["Net Return"] <= 0,
            "Net Return"
        ].sum()
    )

    profit_factor = (
        gross_profit
        /
        gross_loss
        if gross_loss > 0
        else np.inf
    )

    expectancy = avg_return

    max_drawdown = (
        calculate_drawdown(
            evaluated[
                "Net Return"
            ]
        )
    )

    print()

    print(
        "-" * 80
    )

    print(
        "PERFORMANCE STATISTICS"
    )

    print(
        "-" * 80
    )

    print(
        f"Signals evaluated : {total}"
    )

    print(
        f"Winners           : {wins}"
    )

    print(
        f"Losers            : {losses}"
    )

    print(
        f"Win rate          : {win_rate:.2f}%"
    )

    print(
        f"Average trade     : {avg_return:+.3f}%"
    )

    print(
        f"Average winner    : "
        f"{avg_win:+.3f}%"
    )

    print(
        f"Average loser     : "
        f"{avg_loss:+.3f}%"
    )

    print(
        f"Expectancy        : "
        f"{expectancy:+.3f}%"
    )

    print(
        f"Profit factor     : "
        f"{profit_factor:.2f}"
    )

    print(
        f"Max drawdown      : "
        f"{max_drawdown:.2f}%"
    )

    if evaluated["MFE"].notna().any():

        print(
            f"Average MFE       : "
            f"{evaluated['MFE'].mean():+.2f}%"
        )

    if evaluated["MAE"].notna().any():

        print(
            f"Average MAE       : "
            f"{evaluated['MAE'].mean():+.2f}%"
        )

    print()

    print(
        f"Assumed trading cost : "
        f"{COST_MODEL.round_trip_cost_pct:.3f}%"
    )

    print(
        f"Assumed slippage     : "
        f"{COST_MODEL.total_slippage_pct:.3f}%"
    )

    print(
        f"Total assumed cost   : "
        f"{COST_MODEL.total_cost_pct:.3f}%"
    )


# ====================================================================
# SCAN ONE DATE
# ====================================================================

def scan_date(
    target_date: pd.Timestamp,
    historical: bool = False
) -> pd.DataFrame:

    tickers = build_universe()

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    print(
        "Downloading NIFTY 50 benchmark..."
    )

    nifty = download_nifty_data(
        target_date
    )

    (
        nifty_opening_return,
        nifty_total_return
    ) = calculate_market_returns(
        nifty,
        target_date
    )

    if np.isfinite(
        nifty_total_return
    ):

        print(
            f"NIFTY 50 09:45 return: "
            f"{nifty_total_return:+.2f}%"
        )

    else:

        print(
            "WARNING: NIFTY benchmark unavailable."
        )

    # ------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------

    all_results = []

    total = len(
        tickers
    )

    processed = 0

    for start_idx in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = tickers[
            start_idx:
            start_idx + BATCH_SIZE
        ]

        batch_data = (
            download_intraday_batch(
                batch,
                target_date
            )
        )

        for ticker in batch:

            processed += 1

            progress_line(

                f"Scanning "
                f"{processed}/{total} | "
                f"Signals: "
                f"{len(all_results)}"
            )

            try:

                raw_df = batch_data.get(
                    ticker
                )

                if raw_df is None:

                    continue

                analysis = (
                    analyse_at_0945(

                        ticker,

                        raw_df,

                        target_date,

                        nifty_opening_return,

                        nifty_total_return,
                    )
                )

                if analysis is None:

                    continue

                # ------------------------------------------------
                # Liquidity
                # ------------------------------------------------

                if (
                    analysis[
                        "Turnover Cr"
                    ]
                    < MIN_TURNOVER_CR
                ):

                    continue

                # ------------------------------------------------
                # Only actionable signals
                # ------------------------------------------------

                if (
                    analysis[
                        "Signal"
                    ]
                    not in (
                        "LONG",
                        "SHORT"
                    )
                ):

                    continue

                # ------------------------------------------------
                # Historical outcome
                # ------------------------------------------------

                if historical:

                    evaluation = (
                        calculate_historical_outcome(

                            analysis,

                            raw_df,

                            target_date,
                        )
                    )

                else:

                    evaluation = {

                        "MFE":
                            np.nan,

                        "MAE":
                            np.nan,

                        "Outcome":
                            "LIVE",

                        "Target":
                            np.nan,

                        "Stop":
                            np.nan,

                        "Gross Return":
                            np.nan,

                        "Net Return":
                            np.nan,

                        "Bars Held":
                            np.nan,
                    }

                result = {

                    key: value

                    for key, value

                    in analysis.items()

                    if key != "_Data"
                }

                result.update(
                    evaluation
                )

                all_results.append(
                    result
                )

            except Exception:

                continue

        time.sleep(
            BATCH_DELAY
        )

    clear_line()

    if not all_results:

        return pd.DataFrame()

    results = pd.DataFrame(
        all_results
    )

    # ------------------------------------------------------------
    # Rank by signal-specific score.
    # ------------------------------------------------------------

    results["Rank Score"] = np.where(

        results["Signal"] == "SHORT",

        results["Short Score"],

        results["Long Score"]
    )

    results = results.sort_values(

        "Rank Score",

        ascending=False
    ).reset_index(
        drop=True
    )

    return results


# ====================================================================
# SHOW RESULTS
# ====================================================================

def show_results(
    results: pd.DataFrame,
    historical: bool
) -> None:

    if results.empty:

        print()

        print(
            "=" * 100
        )

        print(
            "NO ACTIONABLE SIGNALS"
        )

        print(
            "=" * 100
        )

        print(
            "No stock passed the V2 criteria."
        )

        return

    # ============================================================
    # SHORTS
    # ============================================================

    shorts = results[
        results["Signal"]
        == "SHORT"
    ].copy()

    shorts = shorts.sort_values(
        "Short Score",
        ascending=False
    ).head(
        TOP_SHORTS_TO_SHOW
    ).copy()

    shorts["Rank"] = range(
        1,
        len(shorts) + 1
    )

    # ============================================================
    # LONGS
    # ============================================================

    longs = results[
        results["Signal"]
        == "LONG"
    ].copy()

    longs = longs.sort_values(
        "Long Score",
        ascending=False
    ).head(
        TOP_LONGS_TO_SHOW
    ).copy()

    longs["Rank"] = range(
        1,
        len(longs) + 1
    )

    # ============================================================
    # PRINT
    # ============================================================

    print_table(
        "🔴 TOP SHORT CANDIDATES - V2",
        shorts
    )

    print_short_detail(
        shorts
    )

    print_table(
        "🟢 TOP LONG CANDIDATES - V2",
        longs
    )

    # ============================================================
    # HISTORICAL
    # ============================================================

    if historical:

        evaluation_df = results[
            results["Signal"].isin(
                [
                    "LONG",
                    "SHORT"
                ]
            )
        ].copy()

        print_historical_results(
            evaluation_df
        )

    # ============================================================
    # EXPORT
    # ============================================================

    if EXPORT_RESULTS:

        try:

            results.to_csv(
                EXPORT_FILENAME,
                index=False
            )

            print()

            print(
                f"Results exported to: "
                f"{EXPORT_FILENAME}"
            )

        except Exception as exc:

            print(
                f"CSV export failed: "
                f"{exc}"
            )


# ====================================================================
# MODE
# ====================================================================

def select_mode() -> Tuple[
    str,
    pd.Timestamp
]:

    print()

    print(
        "=" * 90
    )

    print(
        "NSE 09:45 SHORT-BIASED INTRADAY SCANNER V2"
    )

    print(
        "=" * 90
    )

    print()

    print(
        "1. Historical date"
    )

    print(
        "2. Live today"
    )

    print()

    choice = input(
        "Choice [1/2]: "
    ).strip()

    if choice == "1":

        target_date = ask_date()

        return (
            "historical",
            target_date
        )

    if choice == "2":

        now = pd.Timestamp.now(
            tz=MARKET_TZ
        )

        target_date = (
            now
            .tz_localize(None)
            .normalize()
        )

        return (
            "live",
            target_date
        )

    print(
        "Invalid choice."
    )

    sys.exit(1)


# ====================================================================
# LIVE TIME CHECK
# ====================================================================

def check_live_time(
    target_date: pd.Timestamp
) -> None:

    now = pd.Timestamp.now(
        tz=MARKET_TZ
    )

    if (
        now.date()
        != target_date.date()
    ):

        return

    market_time = now.time()

    decision_time = datetime.strptime(
        DECISION_TIME,
        "%H:%M"
    ).time()

    if market_time < decision_time:

        print()

        print(
            "WARNING:"
        )

        print(
            "The scanner is designed "
            "for use after 09:45 AM IST."
        )

        print(
            f"Current time: "
            f"{now.strftime('%H:%M:%S')} IST"
        )

        print()


# ====================================================================
# SUMMARY
# ====================================================================

def print_summary(
    results: pd.DataFrame
) -> None:

    print()

    print(
        "=" * 90
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 90
    )

    if results.empty:

        print(
            "Signals: 0"
        )

        return

    shorts = results[
        results["Signal"]
        == "SHORT"
    ]

    longs = results[
        results["Signal"]
        == "LONG"
    ]

    high_short = shorts[
        shorts["Short Score"]
        >= SHORT_HIGH_CONVICTION_SCORE
    ]

    high_long = longs[
        longs["Long Score"]
        >= LONG_HIGH_CONVICTION_SCORE
    ]

    print(
        f"Actionable signals : "
        f"{len(results)}"
    )

    print(
        f"Short signals      : "
        f"{len(shorts)}"
    )

    print(
        f"Long signals       : "
        f"{len(longs)}"
    )

    print(
        f"High-conviction S  : "
        f"{len(high_short)}"
    )

    print(
        f"High-conviction L  : "
        f"{len(high_long)}"
    )

    print(
        f"Displayed shorts   : "
        f"{min(len(shorts), TOP_SHORTS_TO_SHOW)}"
    )

    print(
        f"Displayed longs    : "
        f"{min(len(longs), TOP_LONGS_TO_SHOW)}"
    )

    print(
        "=" * 90
    )


# ====================================================================
# DATA QUALITY WARNING
# ====================================================================

def print_data_limitations() -> None:

    print()

    print(
        "=" * 90
    )

    print(
        "DATA / BACKTEST LIMITATIONS"
    )

    print(
        "=" * 90
    )

    print(
        "1. Intraday Yahoo data is limited."
    )

    print(
        "2. 1-minute historical evaluation may be unavailable"
        " for older dates."
    )

    print(
        "3. Current NIFTY constituents create survivorship bias"
        " for old historical dates."
    )

    print(
        "4. Yahoo data should not be treated as institutional-grade"
        " NSE execution data."
    )

    print(
        "5. Slippage/cost assumptions must be calibrated to"
        " your actual broker and execution."
    )


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:

    start_time = time.perf_counter()

    try:

        mode, target_date = (
            select_mode()
        )

        historical = (
            mode == "historical"
        )

        print()

        if historical:

            print(
                f"Historical scan: "
                f"{target_date:%Y-%m-%d}"
            )

        else:

            print(
                f"Live scan: "
                f"{target_date:%Y-%m-%d}"
            )

            check_live_time(
                target_date
            )

        print()

        print(
            "V2 FEATURES:"
        )

        print(
            "Historical EMA9 / EMA20 warm-up"
        )

        print(
            "Historical RSI / MACD / ATR warm-up"
        )

        print(
            "Session VWAP"
        )

        print(
            "Same-time-of-day relative volume"
        )

        print(
            "NIFTY-relative strength"
        )

        print(
            "ATR-normalized movement"
        )

        print(
            "Continuous scoring"
        )

        print(
            "Reduced correlated feature double-counting"
        )

        print(
            "Exhaustion protection"
        )

        print(
            "Transaction cost + slippage model"
        )

        if historical:

            if USE_1M_EVALUATION:

                print(
                    "1-minute post-entry evaluation"
                )

            else:

                print(
                    "15-minute post-entry evaluation"
                )

        print()

        results = scan_date(

            target_date,

            historical=historical
        )

        show_results(

            results,

            historical=historical
        )

        print_summary(
            results
        )

        print_data_limitations()

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        print()

        print(
            f"Runtime: "
            f"{elapsed:.1f}s"
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "V2 is a research/scanning system."
        )

        print(
            "A high score does not guarantee"
            " a profitable trade."
        )

    except KeyboardInterrupt:

        print()

        print(
            "Scanner stopped by user."
        )

        sys.exit(0)

    except Exception as exc:

        print()

        print(
            f"Fatal error: {exc}"
        )

        sys.exit(1)


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()

