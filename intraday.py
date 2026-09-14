"""
====================================================================
NSE INTRADAY 09:46 SHORT-BIASED MOMENTUM / REVERSAL SCANNER
====================================================================

PURPOSE
-------
Designed for the 09:46 AM IST decision point.

The model analyzes:

    09:15 - 09:30  Opening candle
    09:30 - 09:45  Confirmation candle

Primary objective:

    FIND HIGH-QUALITY SHORT CANDIDATES

The scanner is intentionally SHORT BIASED.

It does NOT simply select the biggest falling stocks.

It attempts to distinguish:

    Healthy bearish continuation
from
    Oversold / exhausted downside movement

The strategy prefers stocks showing:

    1. Opening weakness
    2. Confirmation weakness
    3. Lower-low structure
    4. Price below VWAP
    5. Price below EMA20
    6. EMA9 below EMA20
    7. Bearish MACD
    8. Strong volume
    9. Bearish candle structure
    10. Controlled RSI
    11. Continued downside momentum

It penalizes:

    1. Extremely oversold RSI
    2. Excessive distance below VWAP
    3. Excessive distance below EMA20
    4. Very large opening collapse
    5. Large lower wick
    6. Momentum deceleration

This is designed to reduce the probability of chasing a stock
that has already fallen too far before 09:46.

OUTPUT
------
The scanner ranks:

    TOP SHORT CANDIDATES

and optionally:

    TOP LONG CANDIDATES

The default output is limited to the best candidates rather than
showing every stock that passes a loose threshold.

HISTORICAL MODE
---------------
The signal is generated using ONLY candles through 09:45.

Candles after 09:45 are used ONLY to evaluate:

    MFE
    MAE
    TARGET
    STOP
    OUTCOME

IMPORTANT
---------
This is a research/scanning system.

It does not guarantee future returns.

====================================================================
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
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

INTERVAL = "15m"


# ====================================================================
# LIQUIDITY
# ====================================================================

MIN_TURNOVER_CR = 2.0


# ====================================================================
# VOLUME
# ====================================================================

VOLUME_LOOKBACK = 8


# ====================================================================
# INDICATORS
# ====================================================================

EMA_FAST = 9
EMA_SLOW = 20

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# ====================================================================
# SHORT-BIASED SIGNAL SETTINGS
# ====================================================================

# Lower than the old 65 threshold.
SHORT_TRADEABLE_SCORE = 60

# Stronger threshold for best shorts.
SHORT_HIGH_CONVICTION_SCORE = 75

# LONGS intentionally require more confirmation.
LONG_TRADEABLE_SCORE = 70
LONG_HIGH_CONVICTION_SCORE = 80


# ====================================================================
# MOVEMENT FILTERS
# ====================================================================

MIN_SHORT_TOTAL_MOVE = 0.30

MIN_LONG_TOTAL_MOVE = 0.50

MIN_OPENING_SHORT_MOVE = 0.35

MIN_OPENING_LONG_MOVE = 0.50


# ====================================================================
# EXHAUSTION SETTINGS
# ====================================================================

# Do not automatically reject a stock merely because it has fallen.
# Instead apply a progressively larger penalty.

RSI_OVERSOLD = 28

RSI_EXTREME_OVERSOLD = 22

MAX_SHORT_VWAP_DISTANCE = 3.0

MAX_SHORT_EMA20_DISTANCE = 3.5

MAX_OPENING_COLLAPSE = 3.0


# ====================================================================
# HISTORICAL EVALUATION
# ====================================================================

TARGET_PCT = 1.00

STOP_PCT = 0.75


# ====================================================================
# DISPLAY
# ====================================================================

TOP_SHORTS_TO_SHOW = 5

TOP_LONGS_TO_SHOW = 3


# ====================================================================
# DOWNLOAD
# ====================================================================

BATCH_SIZE = 50

DOWNLOAD_TIMEOUT = 20

BATCH_DELAY = 0.8


# ====================================================================
# TERMINAL HELPERS
# ====================================================================

def clear_line() -> None:

    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()


def progress_line(text: str) -> None:

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

            return parse_date(value)

        except ValueError:

            print(
                "Invalid date. Use YYYY-MM-DD."
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

    return set(
        symbols
    )


def build_universe() -> List[str]:

    print(
        "Adding NSE universe..."
    )

    nifty500 = get_index_symbols(
        NIFTY_500_URL
    )

    nifty50 = get_index_symbols(
        NIFTY_50_URL
    )

    symbols = sorted(
        nifty500 - nifty50
    )

    tickers = [
        f"{symbol}.NS"
        for symbol in symbols
        if symbol
    ]

    print(
        f"Universe: {len(tickers)} stocks"
    )

    return tickers


# ====================================================================
# YFINANCE QUIET DOWNLOAD
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
# MULTI INDEX
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
# CLEAN DATA
# ====================================================================

def clean_ticker_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df is None or df.empty:

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

    return df


# ====================================================================
# BATCH DOWNLOAD
# ====================================================================

def download_intraday_batch(
    tickers: List[str],
    target_date: pd.Timestamp
) -> Dict[str, pd.DataFrame]:

    start = target_date.strftime(
        "%Y-%m-%d"
    )

    end = (
        target_date
        + pd.Timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    raw = yf_download_quiet(
        tickers,
        start=start,
        end=end,
        interval=INTERVAL,
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
            raw.columns.get_level_values(0)
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

                    results[ticker] = ticker_df

            except Exception:

                continue

    else:

        if len(tickers) == 1:

            ticker_df = clean_ticker_data(
                raw
            )

            if not ticker_df.empty:

                results[tickers[0]] = ticker_df

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

    date_str = target_date.strftime(
        "%Y-%m-%d"
    )

    session_open = pd.Timestamp(
        f"{date_str} {MARKET_OPEN}",
        tz=MARKET_TZ
    )

    session_close = pd.Timestamp(
        f"{date_str} {MARKET_CLOSE}",
        tz=MARKET_TZ
    )

    df = df[
        (df.index >= session_open)
        &
        (df.index < session_close)
    ]

    return df.sort_index()


# ====================================================================
# VWAP
# ====================================================================

def calculate_vwap(
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
        / cumulative_volume.replace(
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
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        - 100 / (1 + rs)
    )

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
    pd.Series
]:

    ema_fast = (
        close
        .ewm(
            span=MACD_FAST,
            adjust=False
        )
        .mean()
    )

    ema_slow = (
        close
        .ewm(
            span=MACD_SLOW,
            adjust=False
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
            adjust=False
        )
        .mean()
    )

    return macd, signal


# ====================================================================
# ADD INDICATORS
# ====================================================================

def add_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["VWAP"] = calculate_vwap(
        df
    )

    df["EMA9"] = (
        df["Close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    df["RSI"] = calculate_rsi(
        df["Close"]
    )

    (
        df["MACD"],
        df["MACD_Signal"]
    ) = calculate_macd(
        df["Close"]
    )

    # ------------------------------------------------------------
    # Candle range
    # ------------------------------------------------------------

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
    ).abs()

    df["Candle_Strength"] = (
        body
        / candle_range
    )

    # ------------------------------------------------------------
    # Candle body direction
    # ------------------------------------------------------------

    df["Body_Pct"] = (
        (
            df["Close"]
            - df["Open"]
        )
        / df["Open"]
    ) * 100

    # ------------------------------------------------------------
    # Upper wick
    # ------------------------------------------------------------

    df["Upper_Wick"] = (
        df["High"]
        - df[
            ["Open", "Close"]
        ].max(axis=1)
    )

    # ------------------------------------------------------------
    # Lower wick
    # ------------------------------------------------------------

    df["Lower_Wick"] = (
        df[
            ["Open", "Close"]
        ].min(axis=1)
        - df["Low"]
    )

    # ------------------------------------------------------------
    # Wick percentages
    # ------------------------------------------------------------

    df["Upper_Wick_Pct"] = (
        df["Upper_Wick"]
        / candle_range
    )

    df["Lower_Wick_Pct"] = (
        df["Lower_Wick"]
        / candle_range
    )

    # ------------------------------------------------------------
    # Relative volume
    # ------------------------------------------------------------

    df["Volume_MA"] = (
        df["Volume"]
        .rolling(
            VOLUME_LOOKBACK,
            min_periods=3
        )
        .mean()
        .shift(1)
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        / df["Volume_MA"].replace(
            0,
            np.nan
        )
    )

    # ------------------------------------------------------------
    # Volume versus opening candle
    # ------------------------------------------------------------

    df["Opening_Volume"] = (
        df["Volume"].iloc[0]
        if len(df) > 0
        else np.nan
    )

    df["Volume_vs_Opening"] = (
        df["Volume"]
        / df["Opening_Volume"]
    )

    return df


# ====================================================================
# EXHAUSTION PENALTY
# ====================================================================

def calculate_short_exhaustion_penalty(
    opening_pct: float,
    total_pct: float,
    vwap_pct: float,
    ema20_pct: float,
    rsi: float,
    confirmation_lower_wick: float,
    momentum_deceleration: bool
) -> Tuple[float, List[str]]:

    penalty = 0.0

    flags = []

    # ------------------------------------------------------------
    # Excessive opening collapse
    # ------------------------------------------------------------

    if opening_pct <= -MAX_OPENING_COLLAPSE:

        penalty += 12

        flags.append(
            "opening_exhaustion"
        )

    elif opening_pct <= -2.25:

        penalty += 6

        flags.append(
            "large_opening_move"
        )

    # ------------------------------------------------------------
    # Very extended below VWAP
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
    # Very extended below EMA20
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
    # RSI exhaustion
    # ------------------------------------------------------------

    if pd.notna(rsi):

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
    # Large lower wick
    #
    # A large lower wick means buyers stepped in.
    # ------------------------------------------------------------

    if confirmation_lower_wick >= 0.40:

        penalty += 8

        flags.append(
            "large_lower_wick"
        )

    elif confirmation_lower_wick >= 0.30:

        penalty += 4

        flags.append(
            "lower_wick"
        )

    # ------------------------------------------------------------
    # Momentum deceleration
    # ------------------------------------------------------------

    if momentum_deceleration:

        penalty += 8

        flags.append(
            "momentum_deceleration"
        )

    return penalty, flags


# ====================================================================
# ANALYSE AT 09:45
# ====================================================================

def analyse_at_0945(
    ticker: str,
    df: pd.DataFrame
) -> Optional[dict]:

    if df.empty:

        return None

    df = add_indicators(
        df
    )

    if len(df) < 2:

        return None

    opening = df.iloc[0]

    confirmation = df.iloc[1]

    # ============================================================
    # OPENING MOVE
    # ============================================================

    opening_pct = (
        (
            float(opening["Close"])
            - float(opening["Open"])
        )
        / float(opening["Open"])
    ) * 100.0

    # ============================================================
    # CONFIRMATION MOVE
    # ============================================================

    confirmation_pct = (
        (
            float(confirmation["Close"])
            - float(confirmation["Open"])
        )
        / float(confirmation["Open"])
    ) * 100.0

    # ============================================================
    # TOTAL MOVE
    # ============================================================

    decision_price = float(
        confirmation["Close"]
    )

    total_pct = (
        (
            decision_price
            - float(opening["Open"])
        )
        / float(opening["Open"])
    ) * 100.0

    # ============================================================
    # INDICATORS
    # ============================================================

    vwap = float(
        confirmation["VWAP"]
    )

    ema9 = float(
        confirmation["EMA9"]
    )

    ema20 = float(
        confirmation["EMA20"]
    )

    rsi = (
        float(confirmation["RSI"])
        if pd.notna(
            confirmation["RSI"]
        )
        else np.nan
    )

    macd = float(
        confirmation["MACD"]
    )

    macd_signal = float(
        confirmation["MACD_Signal"]
    )

    volume_ratio = (
        float(
            confirmation["Volume_Ratio"]
        )
        if pd.notna(
            confirmation["Volume_Ratio"]
        )
        else np.nan
    )

    volume_vs_opening = (
        float(
            confirmation["Volume_vs_Opening"]
        )
        if pd.notna(
            confirmation["Volume_vs_Opening"]
        )
        else np.nan
    )

    candle_strength = (
        float(
            confirmation[
                "Candle_Strength"
            ]
        )
        if pd.notna(
            confirmation[
                "Candle_Strength"
            ]
        )
        else np.nan
    )

    upper_wick_pct = (
        float(
            confirmation[
                "Upper_Wick_Pct"
            ]
        )
        if pd.notna(
            confirmation[
                "Upper_Wick_Pct"
            ]
        )
        else np.nan
    )

    lower_wick_pct = (
        float(
            confirmation[
                "Lower_Wick_Pct"
            ]
        )
        if pd.notna(
            confirmation[
                "Lower_Wick_Pct"
            ]
        )
        else np.nan
    )

    # ============================================================
    # RELATIVE DISTANCES
    # ============================================================

    vwap_pct = (
        (
            decision_price
            - vwap
        )
        / vwap
    ) * 100.0

    ema20_pct = (
        (
            decision_price
            - ema20
        )
        / ema20
    ) * 100.0

    ema9_pct = (
        (
            decision_price
            - ema9
        )
        / ema9
    ) * 100.0

    # ============================================================
    # OPENING RANGE
    # ============================================================

    opening_high = float(
        opening["High"]
    )

    opening_low = float(
        opening["Low"]
    )

    # ============================================================
    # STRUCTURE
    # ============================================================

    higher_high = (
        float(confirmation["High"])
        > opening_high
    )

    lower_low = (
        float(confirmation["Low"])
        < opening_low
    )

    # ============================================================
    # CANDLE DIRECTION
    # ============================================================

    bullish_confirmation = (
        confirmation["Close"]
        > confirmation["Open"]
    )

    bearish_confirmation = (
        confirmation["Close"]
        < confirmation["Open"]
    )

    # ============================================================
    # RANGE EXPANSION
    # ============================================================

    opening_range = (
        float(opening["High"])
        - float(opening["Low"])
    )

    confirmation_range = (
        float(confirmation["High"])
        - float(confirmation["Low"])
    )

    range_expansion = (
        confirmation_range
        / opening_range
        if opening_range > 0
        else np.nan
    )

    # ============================================================
    # MOMENTUM DECELERATION
    # ============================================================

    momentum_deceleration = (
        opening_pct < -0.75
        and confirmation_pct > opening_pct * 0.55
        and confirmation_pct > -0.25
    )

    # ============================================================
    # CONTINUATION
    # ============================================================

    short_continuation = (
        opening_pct < -0.35
        and confirmation_pct < -0.15
        and lower_low
    )

    long_continuation = (
        opening_pct > 0.35
        and confirmation_pct > 0.15
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
    # SHORT EXHAUSTION
    # ============================================================

    exhaustion_penalty, exhaustion_flags = (
        calculate_short_exhaustion_penalty(
            opening_pct=opening_pct,
            total_pct=total_pct,
            vwap_pct=vwap_pct,
            ema20_pct=ema20_pct,
            rsi=rsi,
            confirmation_lower_wick=lower_wick_pct,
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
    # Opening weakness
    # ------------------------------------------------------------

    if opening_pct < -0.35:

        short_score += 8

    if opening_pct < -0.75:

        short_score += 5

    if opening_pct < -1.25:

        short_score += 3

    # ------------------------------------------------------------
    # Confirmation weakness
    # ------------------------------------------------------------

    if confirmation_pct < 0:

        short_score += 12

    if confirmation_pct < -0.30:

        short_score += 6

    if confirmation_pct < -0.60:

        short_score += 4

    # ------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------

    if decision_price < vwap:

        short_score += 12

    if vwap_pct < -0.50:

        short_score += 3

    # ------------------------------------------------------------
    # EMA20
    # ------------------------------------------------------------

    if decision_price < ema20:

        short_score += 8

    if ema9 < ema20:

        short_score += 10

    # ------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------

    if macd < macd_signal:

        short_score += 8

    # ------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------

    if pd.notna(rsi):

        if 35 <= rsi <= 55:

            short_score += 6

        elif 28 <= rsi < 35:

            short_score += 3

        elif rsi < 28:

            short_score -= 3

    # ------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------

    if pd.notna(volume_ratio):

        if volume_ratio >= 1.15:

            short_score += 6

        if volume_ratio >= 1.40:

            short_score += 4

        if volume_ratio >= 1.80:

            short_score += 3

    # ------------------------------------------------------------
    # Confirmation volume vs opening
    # ------------------------------------------------------------

    if pd.notna(volume_vs_opening):

        if volume_vs_opening >= 1.05:

            short_score += 5

        elif volume_vs_opening >= 0.80:

            short_score += 2

    # ------------------------------------------------------------
    # Lower low
    # ------------------------------------------------------------

    if lower_low:

        short_score += 10

    # ------------------------------------------------------------
    # Bearish candle
    # ------------------------------------------------------------

    if bearish_confirmation:

        short_score += 5

    # ------------------------------------------------------------
    # Candle strength
    # ------------------------------------------------------------

    if pd.notna(candle_strength):

        if candle_strength >= 0.55:

            short_score += 5

        elif candle_strength >= 0.40:

            short_score += 2

    # ------------------------------------------------------------
    # Bearish candle close quality
    #
    # Close near low is desirable.
    # ------------------------------------------------------------

    if pd.notna(lower_wick_pct):

        if lower_wick_pct <= 0.20:

            short_score += 4

        elif lower_wick_pct >= 0.40:

            short_score -= 5

    # ------------------------------------------------------------
    # Range expansion
    # ------------------------------------------------------------

    if pd.notna(range_expansion):

        if range_expansion >= 1.00:

            short_score += 4

        elif range_expansion >= 0.75:

            short_score += 2

    # ------------------------------------------------------------
    # Continuation
    # ------------------------------------------------------------

    if short_continuation:

        short_score += 8

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
    # LONG SCORE
    # ============================================================

    long_score = 0.0

    if opening_pct > 0.50:

        long_score += 8

    if opening_pct > 1.00:

        long_score += 5

    if confirmation_pct > 0:

        long_score += 12

    if confirmation_pct > 0.40:

        long_score += 5

    if decision_price > vwap:

        long_score += 12

    if decision_price > ema20:

        long_score += 8

    if ema9 > ema20:

        long_score += 10

    if macd > macd_signal:

        long_score += 8

    if pd.notna(rsi):

        if 45 <= rsi <= 70:

            long_score += 6

        elif rsi > 75:

            long_score -= 8

    if pd.notna(volume_ratio):

        if volume_ratio >= 1.15:

            long_score += 6

        if volume_ratio >= 1.50:

            long_score += 4

    if higher_high:

        long_score += 10

    if bullish_confirmation:

        long_score += 5

    if pd.notna(candle_strength):

        if candle_strength >= 0.55:

            long_score += 5

    if pd.notna(upper_wick_pct):

        if upper_wick_pct <= 0.20:

            long_score += 4

        elif upper_wick_pct >= 0.40:

            long_score -= 5

    if long_continuation:

        long_score += 8

    if long_reversal:

        long_score -= 25

    # ============================================================
    # NORMALIZE
    # ============================================================

    short_score = max(
        0,
        min(
            100,
            short_score
        )
    )

    long_score = max(
        0,
        min(
            100,
            long_score
        )
    )

    # ============================================================
    # DECISION
    # ============================================================

    signal = "AVOID"

    score = max(
        short_score,
        long_score
    )

    conviction = "NEUTRAL"

    # ------------------------------------------------------------
    # SHORT FIRST
    #
    # Because this is intentionally short biased.
    # ------------------------------------------------------------

    if (
        short_score >= SHORT_TRADEABLE_SCORE
        and short_score >= long_score
        and total_pct <= -MIN_SHORT_TOTAL_MOVE
        and opening_pct <= -MIN_OPENING_SHORT_MOVE
        and not short_reversal
    ):

        signal = "SHORT"

        score = short_score

        if short_score >= SHORT_HIGH_CONVICTION_SCORE:

            conviction = "HIGH"

        else:

            conviction = "TRADEABLE"

    elif (
        long_score >= LONG_TRADEABLE_SCORE
        and long_score > short_score
        and total_pct >= MIN_LONG_TOTAL_MOVE
        and opening_pct >= MIN_OPENING_LONG_MOVE
        and not long_reversal
    ):

        signal = "LONG"

        score = long_score

        if long_score >= LONG_HIGH_CONVICTION_SCORE:

            conviction = "HIGH"

        else:

            conviction = "TRADEABLE"

    # ============================================================
    # TURNOVER
    # ============================================================

    turnover_cr = (
        float(
            confirmation["Volume"]
        )
        * decision_price
        / 10_000_000
    )

    # ============================================================
    # RETURN
    # ============================================================

    return {
        "Ticker": ticker.replace(
            ".NS",
            ""
        ),

        "Signal": signal,

        "Conviction": conviction,

        "Score": round(
            score,
            1
        ),

        "Short Score": round(
            short_score,
            1
        ),

        "Long Score": round(
            long_score,
            1
        ),

        "Opening %": opening_pct,

        "09:30-09:45 %": confirmation_pct,

        "Total %": total_pct,

        "Price": decision_price,

        "VWAP %": vwap_pct,

        "EMA20 %": ema20_pct,

        "EMA9 %": ema9_pct,

        "Volume x": volume_ratio,

        "Volume vs Open": volume_vs_opening,

        "RSI": rsi,

        "MACD": (
            "Bullish"
            if macd > macd_signal
            else "Bearish"
        ),

        "Turnover Cr": turnover_cr,

        "Opening High": opening_high,

        "Opening Low": opening_low,

        "Lower Low": lower_low,

        "Higher High": higher_high,

        "Range Expansion": range_expansion,

        "Exhaustion Penalty": (
            exhaustion_penalty
        ),

        "Exhaustion Flags": (
            ",".join(
                exhaustion_flags
            )
            if exhaustion_flags
            else ""
        ),

        "Decision Timestamp": (
            confirmation.name
        ),

        "_Data": df,
    }


# ====================================================================
# HISTORICAL OUTCOME
# ====================================================================

def calculate_historical_outcome(
    analysis: dict,
    full_df: pd.DataFrame
) -> dict:

    signal = analysis[
        "Signal"
    ]

    if signal not in (
        "LONG",
        "SHORT"
    ):

        return {
            "MFE": np.nan,
            "MAE": np.nan,
            "Outcome": "N/A",
            "Target": np.nan,
            "Stop": np.nan,
        }

    decision_price = float(
        analysis["Price"]
    )

    decision_timestamp = (
        analysis[
            "Decision Timestamp"
        ]
    )

    future = full_df[
        full_df.index > decision_timestamp
    ].copy()

    if future.empty:

        return {
            "MFE": np.nan,
            "MAE": np.nan,
            "Outcome": "NO_DATA",
            "Target": np.nan,
            "Stop": np.nan,
        }

    # ============================================================
    # LONG
    # ============================================================

    if signal == "LONG":

        target = (
            decision_price
            * (
                1
                + TARGET_PCT / 100
            )
        )

        stop = (
            decision_price
            * (
                1
                - STOP_PCT / 100
            )
        )

        max_favorable = (
            (
                future["High"].max()
                - decision_price
            )
            / decision_price
        ) * 100

        max_adverse = (
            (
                future["Low"].min()
                - decision_price
            )
            / decision_price
        ) * 100

        outcome = "OPEN"

        for _, row in future.iterrows():

            hit_stop = (
                row["Low"]
                <= stop
            )

            hit_target = (
                row["High"]
                >= target
            )

            if hit_stop and hit_target:

                # Conservative assumption:
                # both levels occurred within the
                # same 15-minute candle, so assume
                # STOP happened first.

                outcome = "STOP"

                break

            if hit_stop:

                outcome = "STOP"

                break

            if hit_target:

                outcome = "TARGET"

                break

        if outcome == "OPEN":

            final_price = float(
                future.iloc[-1]["Close"]
            )

            if final_price > decision_price:

                outcome = "PROFIT"

            else:

                outcome = "LOSS"

    # ============================================================
    # SHORT
    # ============================================================

    else:

        target = (
            decision_price
            * (
                1
                - TARGET_PCT / 100
            )
        )

        stop = (
            decision_price
            * (
                1
                + STOP_PCT / 100
            )
        )

        max_favorable = (
            (
                decision_price
                - future["Low"].min()
            )
            / decision_price
        ) * 100

        max_adverse = (
            (
                future["High"].max()
                - decision_price
            )
            / decision_price
        ) * 100

        outcome = "OPEN"

        for _, row in future.iterrows():

            hit_stop = (
                row["High"]
                >= stop
            )

            hit_target = (
                row["Low"]
                <= target
            )

            if hit_stop and hit_target:

                # Conservative assumption.

                outcome = "STOP"

                break

            if hit_stop:

                outcome = "STOP"

                break

            if hit_target:

                outcome = "TARGET"

                break

        if outcome == "OPEN":

            final_price = float(
                future.iloc[-1]["Close"]
            )

            if final_price < decision_price:

                outcome = "PROFIT"

            else:

                outcome = "LOSS"

    return {
        "MFE": float(
            max_favorable
        ),

        "MAE": float(
            max_adverse
        ),

        "Outcome": outcome,

        "Target": target,

        "Stop": stop,
    }


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
    "Price",
    "VWAP %",
    "EMA20 %",
    "Volume x",
    "RSI",
    "MACD",
    "Turnover Cr",
]


# ====================================================================
# FORMAT DISPLAY
# ====================================================================

def format_display(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df.empty:

        return df

    out = df.copy()

    for column in [
        "Opening %",
        "09:30-09:45 %",
        "Total %",
        "VWAP %",
        "EMA20 %",
    ]:

        out[column] = out[
            column
        ].map(
            lambda x:
            f"{x:+.2f}%"
            if pd.notna(x)
            else "-"
        )

    out["Price"] = out[
        "Price"
    ].map(
        lambda x:
        f"₹{x:,.2f}"
        if pd.notna(x)
        else "-"
    )

    out["Volume x"] = out[
        "Volume x"
    ].map(
        lambda x:
        f"{x:.2f}x"
        if pd.notna(x)
        else "-"
    )

    out["RSI"] = out[
        "RSI"
    ].map(
        lambda x:
        f"{x:.1f}"
        if pd.notna(x)
        else "-"
    )

    out["Turnover Cr"] = out[
        "Turnover Cr"
    ].map(
        lambda x:
        f"{x:.1f}"
        if pd.notna(x)
        else "-"
    )

    out["Score"] = out[
        "Score"
    ].map(
        lambda x:
        f"{x:.0f}"
        if pd.notna(x)
        else "-"
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
        "=" * 135
    )

    print(title)

    print(
        "=" * 135
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
# PRINT TOP SHORT DETAIL
# ====================================================================

def print_short_detail(
    df: pd.DataFrame
) -> None:

    if df.empty:

        return

    print()

    print(
        "=" * 100
    )

    print(
        "SHORT SETUP QUALITY"
    )

    print(
        "=" * 100
    )

    for _, row in df.head(
        TOP_SHORTS_TO_SHOW
    ).iterrows():

        flags = row.get(
            "Exhaustion Flags",
            ""
        )

        print(
            f"{row['Ticker']:12s} | "
            f"Score {row['Score']:5.1f} | "
            f"Short {row['Short Score']:5.1f} | "
            f"RSI "
            f"{row['RSI']:.1f}"
            if pd.notna(row["RSI"])
            else
            f"{row['Ticker']:12s} | "
            f"Score {row['Score']:5.1f}"
        )

        if flags:

            print(
                f"    Exhaustion warnings: "
                f"{flags}"
            )

        else:

            print(
                "    Exhaustion warnings: none"
            )


# ====================================================================
# HISTORICAL RESULTS
# ====================================================================

def print_historical_results(
    df: pd.DataFrame
) -> None:

    print()

    print(
        "=" * 125
    )

    print(
        "HISTORICAL POST-09:45 EVALUATION"
    )

    print(
        "=" * 125
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
    ]:

        display[column] = display[
            column
        ].map(
            lambda x:
            f"{x:+.2f}%"
            if pd.notna(x)
            else "-"
        )

    display["Score"] = display[
        "Score"
    ].map(
        lambda x:
        f"{x:.0f}"
        if pd.notna(x)
        else "-"
    )

    print(
        display.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    total = len(
        display
    )

    targets = (
        df["Outcome"]
        == "TARGET"
    ).sum()

    stops = (
        df["Outcome"]
        == "STOP"
    ).sum()

    profits = (
        df["Outcome"]
        == "PROFIT"
    ).sum()

    losses = (
        df["Outcome"]
        == "LOSS"
    ).sum()

    resolved = (
        targets
        + stops
        + profits
        + losses
    )

    print()

    print(
        f"Signals evaluated : {total}"
    )

    print(
        f"Targets           : {targets}"
    )

    print(
        f"Stops             : {stops}"
    )

    print(
        f"Profit at close   : {profits}"
    )

    print(
        f"Loss at close     : {losses}"
    )

    if resolved > 0:

        target_rate = (
            targets / resolved
        ) * 100

        stop_rate = (
            stops / resolved
        ) * 100

        print(
            f"Target rate       : "
            f"{target_rate:.1f}%"
        )

        print(
            f"Stop rate         : "
            f"{stop_rate:.1f}%"
        )

    if df["MFE"].notna().any():

        print(
            f"Average MFE       : "
            f"{df['MFE'].mean():+.2f}%"
        )

    if df["MAE"].notna().any():

        print(
            f"Average MAE       : "
            f"{df['MAE'].mean():+.2f}%"
        )


# ====================================================================
# SCAN ONE DATE
# ====================================================================

def scan_date(
    target_date: pd.Timestamp,
    historical: bool = False
) -> pd.DataFrame:

    tickers = build_universe()

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
                f"Candidates: "
                f"{len(all_results)}"
            )

            try:

                raw_df = batch_data.get(
                    ticker
                )

                if raw_df is None:

                    continue

                session_df = (
                    filter_session(
                        raw_df,
                        target_date
                    )
                )

                if len(session_df) < 2:

                    continue

                analysis = (
                    analyse_at_0945(
                        ticker,
                        session_df
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
                # Keep actionable signals
                # ------------------------------------------------

                if analysis[
                    "Signal"
                ] not in (
                    "LONG",
                    "SHORT"
                ):

                    continue

                # ------------------------------------------------
                # Historical evaluation
                #
                # IMPORTANT:
                # Signal still uses only first two candles.
                # Full session is used only here.
                # ------------------------------------------------

                if historical:

                    evaluation = (
                        calculate_historical_outcome(
                            analysis,
                            session_df
                        )
                    )

                else:

                    evaluation = {
                        "MFE": np.nan,
                        "MAE": np.nan,
                        "Outcome": "LIVE",
                        "Target": np.nan,
                        "Stop": np.nan,
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

    # ============================================================
    # RANKING
    # ============================================================

    results = results.sort_values(
        [
            "Signal",
            "Score"
        ],
        ascending=[
            True,
            False
        ]
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
            "No stock passed the 09:45 criteria."
        )

        return

    # ============================================================
    # SHORTS
    # ============================================================

    shorts = results[
        results["Signal"] == "SHORT"
    ].copy()

    shorts = shorts.sort_values(
        "Short Score",
        ascending=False
    )

    shorts = shorts.head(
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
        results["Signal"] == "LONG"
    ].copy()

    longs = longs.sort_values(
        "Score",
        ascending=False
    )

    longs = longs.head(
        TOP_LONGS_TO_SHOW
    ).copy()

    longs["Rank"] = range(
        1,
        len(longs) + 1
    )

    # ============================================================
    # SHORT OUTPUT
    # ============================================================

    print_table(
        "🔴 TOP SHORT CANDIDATES",
        shorts
    )

    print_short_detail(
        shorts
    )

    # ============================================================
    # LONG OUTPUT
    # ============================================================

    print_table(
        "🟢 TOP LONG CANDIDATES",
        longs
    )

    # ============================================================
    # HISTORICAL
    # ============================================================

    if historical:

        evaluation_df = results[
            results[
                "Signal"
            ].isin(
                [
                    "LONG",
                    "SHORT"
                ]
            )
        ].copy()

        print_historical_results(
            evaluation_df
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
        "=" * 80
    )

    print(
        "NSE 09:46 SHORT-BIASED INTRADAY SCANNER"
    )

    print(
        "=" * 80
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
        "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    if results.empty:

        print(
            "Signals: 0"
        )

        return

    shorts = results[
        results["Signal"] == "SHORT"
    ]

    longs = results[
        results["Signal"] == "LONG"
    ]

    high_short = shorts[
        shorts["Score"]
        >= SHORT_HIGH_CONVICTION_SCORE
    ]

    high_long = longs[
        longs["Score"]
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
        "=" * 80
    )


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:

    start_time = time.perf_counter()

    try:

        mode, target_date = select_mode()

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
            "Strategy:"
        )

        print(
            "SHORT BIASED"
        )

        print(
            "09:15-09:30 opening weakness"
        )

        print(
            "09:30-09:45 confirmation"
        )

        print(
            "VWAP + EMA9/20 + volume + RSI + MACD"
        )

        print(
            "Lower-low continuation"
        )

        print(
            "Exhaustion protection"
        )

        print()

        results = scan_date(
            target_date,
            historical=historical
        )

        show_results(
            results,
            historical
        )

        print_summary(
            results
        )

        elapsed = (
            time.perf_counter()
            - start_time
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
            "This is a research/scanning tool."
        )

        print(
            "A high score does not guarantee "
            "a profitable short trade."
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

