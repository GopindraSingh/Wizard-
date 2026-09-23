"""
NSE 09:45 LIVE SHORT-BIASED SCANNER
===================================

Execution target:
    09:46 IST

Signal candles:
    09:15 - 09:30
    09:30 - 09:45

The scanner:
    - downloads historical 15m warm-up data
    - calculates EMA9 / EMA20 / RSI / MACD / ATR
    - calculates session VWAP
    - calculates same-time-of-day RVOL
    - compares stock performance with NIFTY 50
    - normalizes movement by ATR
    - applies exhaustion protection
    - applies liquidity filter
    - produces SHORT/LONG candidates
    - sends exactly one Telegram message per execution

Environment variables:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Optional:
    MIN_TURNOVER_CR
    TOP_SHORTS_TO_SHOW
    TOP_LONGS_TO_SHOW
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("nse_scanner")


# ================================================================
# CONFIGURATION
# ================================================================

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
EXECUTION_TIME = "09:46"
MARKET_CLOSE = "15:30"

SIGNAL_INTERVAL = "15m"

WARMUP_DAYS = 45
RVOL_LOOKBACK_DAYS = 20
MIN_RVOL_OBSERVATIONS = 8

EXCLUDE_NIFTY50 = True

MIN_TURNOVER_CR = float(
    os.getenv("MIN_TURNOVER_CR", "2.0")
)

EMA_FAST = 9
EMA_SLOW = 20

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_PERIOD = 14

SHORT_TRADEABLE_SCORE = 60.0
SHORT_HIGH_CONVICTION_SCORE = 75.0

LONG_TRADEABLE_SCORE = 70.0
LONG_HIGH_CONVICTION_SCORE = 80.0

MIN_SHORT_TOTAL_MOVE = 0.30
MIN_LONG_TOTAL_MOVE = 0.50

MIN_OPENING_SHORT_MOVE = 0.25
MIN_OPENING_LONG_MOVE = 0.35

RSI_OVERSOLD = 30.0
RSI_EXTREME_OVERSOLD = 22.0

MAX_SHORT_VWAP_DISTANCE = 3.0
MAX_SHORT_EMA20_DISTANCE = 3.5
MAX_OPENING_COLLAPSE = 3.0

NIFTY50_TICKER = "^NSEI"

BATCH_SIZE = 40
DOWNLOAD_TIMEOUT = 30
BATCH_DELAY = 0.8

TOP_SHORTS_TO_SHOW = int(
    os.getenv("TOP_SHORTS_TO_SHOW", "5")
)

TOP_LONGS_TO_SHOW = int(
    os.getenv("TOP_LONGS_TO_SHOW", "3")
)


# ================================================================
# TELEGRAM CONFIGURATION
# ================================================================

# IMPORTANT:
# These are ENVIRONMENT VARIABLE NAMES.
#
# Do NOT put the actual Telegram token here.
#
# GitHub Actions should provide:
#   TELEGRAM_BOT_TOKEN
#   TELEGRAM_CHAT_ID

TELEGRAM_BOT_TOKEN = os.getenv(
    "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1U0NRTEMiLCJqdGkiOiI2YWIxOTQyYzExMDA2ZDE4Nzk5NzVlNjYiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaXNFeHRlbmRlZCI6dHJ1ZSwiaWF0IjoxNzkwMDIyNzAwLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE4MjE1NjQwMDB9.hZWxTPTvOA5c_mpXk95aXNovdRq5gXQvDwEMzM3EUdk",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "1860594381",
    ""
).strip()


# ================================================================
# DATA CLASS
# ================================================================

@dataclass
class ScanStats:
    universe: int = 0
    downloaded: int = 0
    failed_download: int = 0
    analyzed: int = 0
    rejected_liquidity: int = 0
    actionable: int = 0
    shorts: int = 0
    longs: int = 0


# ================================================================
# TELEGRAM
# ================================================================

def telegram_enabled() -> bool:
    return bool(
        TELEGRAM_BOT_TOKEN
        and TELEGRAM_CHAT_ID
    )


def validate_configuration() -> None:
    """
    Validate critical configuration before starting the expensive scan.
    """

    missing = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


def send_telegram_message(
    message: str
) -> None:

    validate_configuration()

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    logger.info("Sending Telegram message...")

    response = requests.post(
        url,
        json=payload,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(
            f"Telegram API error: {data}"
        )

    logger.info("Telegram message sent successfully.")


# ================================================================
# YFINANCE
# ================================================================

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

    except Exception as exc:

        logger.warning(
            "Yahoo Finance download failed: %s",
            exc,
        )

        return pd.DataFrame()


# ================================================================
# NSE UNIVERSE
# ================================================================

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
        "Referer": "https://www.nseindia.com/",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return pd.read_csv(
        io.BytesIO(response.content)
    )


def find_symbol_column(
    df: pd.DataFrame
) -> str:

    for column in (
        "Symbol",
        "SYMBOL",
        "symbol",
        "Ticker",
        "TICKER",
    ):
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find NSE symbol column."
    )


def get_index_symbols(
    url: str
) -> set[str]:

    df = download_nse_csv(url)

    column = find_symbol_column(df)

    symbols = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return set(symbols)


def build_universe() -> List[str]:

    nifty500 = get_index_symbols(
        NIFTY_500_URL
    )

    logger.info(
        "NIFTY 500 symbols downloaded: %d",
        len(nifty500),
    )

    if EXCLUDE_NIFTY50:

        nifty50 = get_index_symbols(
            NIFTY_50_URL
        )

        logger.info(
            "NIFTY 50 symbols downloaded: %d",
            len(nifty50),
        )

        symbols = nifty500 - nifty50

    else:

        symbols = nifty500

    universe = [
        f"{symbol}.NS"
        for symbol in sorted(symbols)
        if symbol
    ]

    return universe


# ================================================================
# DATA CLEANING
# ================================================================

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


def normalize_intraday_index(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    df = df[
        ~df.index.isna()
    ]

    if getattr(
        df.index,
        "tz",
        None
    ) is not None:

        df.index = (
            df.index
            .tz_convert(MARKET_TZ)
        )

    else:

        df.index = (
            df.index
            .tz_localize(
                MARKET_TZ
            )
        )

    return df.sort_index()


def clean_ticker_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_columns(df)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.DataFrame()

    df = df[required].copy()

    df = normalize_intraday_index(df)

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["Volume"] >= 0)
        & (df["High"] >= df["Low"])
    ]

    return df.sort_index()


# ================================================================
# INTRADAY DOWNLOAD
# ================================================================

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

    logger.info(
        "Downloading batch of %d tickers...",
        len(tickers),
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

                    results[ticker] = ticker_df

            except Exception as exc:

                logger.warning(
                    "Failed cleaning %s: %s",
                    ticker,
                    exc,
                )

    elif len(tickers) == 1:

        ticker_df = clean_ticker_data(
            raw
        )

        if not ticker_df.empty:

            results[tickers[0]] = ticker_df

    return results


# ================================================================
# SESSION
# ================================================================

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
        tz=MARKET_TZ,
    )

    session_close = pd.Timestamp(
        f"{date_str} {MARKET_CLOSE}",
        tz=MARKET_TZ,
    )

    return df[
        (df.index >= session_open)
        & (df.index < session_close)
    ].sort_index()


# ================================================================
# SIGNAL BAR VALIDATION
# ================================================================

def get_signal_bars(
    df: pd.DataFrame,
    target_date: pd.Timestamp
) -> Optional[
    Tuple[pd.Series, pd.Series]
]:

    session = filter_session(
        df,
        target_date,
    )

    if session.empty:
        return None

    # We only want the first two regular-session
    # 15-minute candles:
    #
    # 09:15
    # 09:30
    #
    # Any later candles are ignored.

    expected_times = [
        "09:15",
        "09:30",
    ]

    bars = []

    for expected_time in expected_times:

        matches = session[
            session.index.strftime(
                "%H:%M"
            ) == expected_time
        ]

        if matches.empty:
            return None

        bars.append(
            matches.iloc[0]
        )

    return bars[0], bars[1]


# ================================================================
# INDICATORS
# ================================================================

def calculate_session_vwap(
    df: pd.DataFrame
) -> pd.Series:

    typical_price = (
        df["High"]
        + df["Low"]
        + df["Close"]
    ) / 3.0

    cumulative_pv = (
        typical_price * df["Volume"]
    ).cumsum()

    cumulative_volume = (
        df["Volume"].cumsum()
    )

    return (
        cumulative_pv
        /
        cumulative_volume.replace(
            0,
            np.nan,
        )
    )


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
        gain.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
    )

    avg_loss = (
        loss.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
    )

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan,
        )
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi.where(
        avg_loss != 0,
        100,
    )


def calculate_macd(
    close: pd.Series
) -> Tuple[
    pd.Series,
    pd.Series,
    pd.Series
]:

    ema_fast = (
        close.ewm(
            span=MACD_FAST,
            adjust=False,
            min_periods=MACD_FAST,
        ).mean()
    )

    ema_slow = (
        close.ewm(
            span=MACD_SLOW,
            adjust=False,
            min_periods=MACD_SLOW,
        ).mean()
    )

    macd = ema_fast - ema_slow

    signal = (
        macd.ewm(
            span=MACD_SIGNAL,
            adjust=False,
            min_periods=MACD_SIGNAL,
        ).mean()
    )

    histogram = macd - signal

    return (
        macd,
        signal,
        histogram,
    )


def calculate_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD
) -> pd.Series:

    previous_close = (
        df["Close"].shift(1)
    )

    tr1 = (
        df["High"] - df["Low"]
    )

    tr2 = (
        df["High"] - previous_close
    ).abs()

    tr3 = (
        df["Low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return (
        true_range.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
    )


def add_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["EMA9"] = (
        df["Close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
            min_periods=EMA_FAST,
        ).mean()
    )

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
            min_periods=EMA_SLOW,
        ).mean()
    )

    df["RSI"] = calculate_rsi(
        df["Close"]
    )

    (
        df["MACD"],
        df["MACD_Signal"],
        df["MACD_Hist"],
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

    candle_range = (
        df["High"] - df["Low"]
    ).replace(
        0,
        np.nan,
    )

    df["Candle_Range"] = (
        candle_range
    )

    df["Body"] = (
        df["Close"] - df["Open"]
    ).abs()

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
        / candle_range
    )

    df["Lower_Wick_Pct"] = (
        df["Lower_Wick"]
        / candle_range
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


# ================================================================
# RVOL
# ================================================================

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

    historical = df[
        (
            df["Session_Date"]
            < target_date.date()
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
            np.nan,
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


# ================================================================
# NIFTY
# ================================================================

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


def calculate_market_returns(
    nifty: pd.DataFrame,
    target_date: pd.Timestamp
) -> Tuple[float, float]:

    bars = get_signal_bars(
        nifty,
        target_date,
    )

    if bars is None:
        return np.nan, np.nan

    opening, confirmation = bars

    opening_return = (
        (
            opening["Close"]
            - opening["Open"]
        )
        / opening["Open"]
    ) * 100

    total_return = (
        (
            confirmation["Close"]
            - opening["Open"]
        )
        / opening["Open"]
    ) * 100

    return (
        float(opening_return),
        float(total_return),
    )


# ================================================================
# SCORE HELPERS
# ================================================================

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
            high,
        )
    )


def bearish_move_score(
    value_pct: float,
    scale: float
) -> float:

    if not np.isfinite(value_pct):
        return 0.0

    return clamp(
        -value_pct / scale,
        0,
        1,
    )


def bullish_move_score(
    value_pct: float,
    scale: float
) -> float:

    if not np.isfinite(value_pct):
        return 0.0

    return clamp(
        value_pct / scale,
        0,
        1,
    )


# ================================================================
# EXHAUSTION
# ================================================================

def calculate_short_exhaustion(
    opening_pct: float,
    vwap_pct: float,
    ema20_pct: float,
    rsi: float,
    lower_wick_pct: float,
    momentum_deceleration: bool
) -> Tuple[float, List[str]]:

    penalty = 0.0
    flags = []

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

    if np.isfinite(lower_wick_pct):

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

    if momentum_deceleration:

        penalty += 8
        flags.append(
            "momentum_deceleration"
        )

    return penalty, flags


# ================================================================
# ANALYSIS
# ================================================================

def analyse_at_0945(
    ticker: str,
    full_df: pd.DataFrame,
    target_date: pd.Timestamp,
    nifty_opening_return: float,
    nifty_total_return: float
) -> Optional[dict]:

    if full_df.empty:
        return None

    df = add_features(
        full_df
    )

    bars = get_signal_bars(
        df,
        target_date,
    )

    if bars is None:
        return None

    opening, confirmation = bars

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

    if opening_price <= 0:
        return None

    opening_pct = (
        (
            opening_close
            - opening_price
        )
        / opening_price
    ) * 100

    confirmation_pct = (
        (
            decision_price
            - confirmation_open
        )
        / confirmation_open
    ) * 100

    total_pct = (
        (
            decision_price
            - opening_price
        )
        / opening_price
    ) * 100

    relative_total_strength = (
        total_pct
        - nifty_total_return
        if np.isfinite(
            nifty_total_return
        )
        else np.nan
    )

    session = filter_session(
        df,
        target_date,
    )

    signal_session = session[
        session.index <= confirmation.name
    ]

    vwap = calculate_session_vwap(
        signal_session
    )

    if vwap.empty:
        return None

    vwap_value = float(
        vwap.iloc[-1]
    )

    if not np.isfinite(
        vwap_value
    ):
        return None

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

    macd = (
        float(confirmation["MACD"])
        if pd.notna(
            confirmation["MACD"]
        )
        else np.nan
    )

    macd_signal = (
        float(
            confirmation[
                "MACD_Signal"
            ]
        )
        if pd.notna(
            confirmation["MACD_Signal"]
        )
        else np.nan
    )

    macd_hist = (
        float(
            confirmation[
                "MACD_Hist"
            ]
        )
        if pd.notna(
            confirmation["MACD_Hist"]
        )
        else np.nan
    )

    atr = (
        float(confirmation["ATR"])
        if pd.notna(
            confirmation["ATR"]
        )
        else np.nan
    )

    atr_pct = (
        float(
            confirmation[
                "ATR_Pct"
            ]
        )
        if pd.notna(
            confirmation["ATR_Pct"]
        )
        else np.nan
    )

    if not all(
        np.isfinite(x)
        for x in [
            ema9,
            ema20,
            rsi,
            atr,
        ]
    ):

        return None

    if atr <= 0:
        return None

    vwap_pct = (
        (
            decision_price
            - vwap_value
        )
        / vwap_value
    ) * 100

    ema20_pct = (
        (
            decision_price
            - ema20
        )
        / ema20
    ) * 100

    ema9_pct = (
        (
            decision_price
            - ema9
        )
        / ema9
    ) * 100

    ema_spread_pct = (
        (
            ema9 - ema20
        )
        / ema20
    ) * 100

    normalized_total_move = (
        (
            decision_price
            - opening_price
        )
        / atr
    )

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
        < opening_low
    )

    higher_high = (
        confirmation_high
        > opening_high
    )

    confirmation_range = float(
        confirmation[
            "Candle_Range"
        ]
    )

    candle_strength = float(
        confirmation[
            "Candle_Strength"
        ]
    )

    lower_wick_pct = float(
        confirmation[
            "Lower_Wick_Pct"
        ]
    )

    upper_wick_pct = float(
        confirmation[
            "Upper_Wick_Pct"
        ]
    )

    bearish_confirmation = (
        decision_price
        < confirmation_open
    )

    bullish_confirmation = (
        decision_price
        > confirmation_open
    )

    opening_range = (
        opening_high
        - opening_low
    )

    range_expansion = (
        confirmation_range
        / opening_range
        if opening_range > 0
        else np.nan
    )

    opening_rvol = (
        calculate_rvol_for_target(
            df,
            target_date,
            "09:15",
            float(
                opening["Volume"]
            ),
        )
    )

    confirmation_rvol = (
        calculate_rvol_for_target(
            df,
            target_date,
            "09:30",
            float(
                confirmation["Volume"]
            ),
        )
    )

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

    exhaustion_penalty, exhaustion_flags = (
        calculate_short_exhaustion(
            opening_pct,
            vwap_pct,
            ema20_pct,
            rsi,
            lower_wick_pct,
            momentum_deceleration,
        )
    )

    # ============================================================
    # SHORT SCORE
    # ============================================================

    short_score = 0.0

    short_score += (
        12 * bearish_move_score(
            opening_pct,
            1.50,
        )
    )

    short_score += (
        12 * bearish_move_score(
            confirmation_pct,
            1.00,
        )
    )

    short_score += (
        12 * bearish_move_score(
            relative_total_strength,
            1.50,
        )
    )

    short_score += (
        10 * clamp(
            -vwap_pct / 2.0,
            0,
            1,
        )
    )

    ema_structure = 0.0

    if decision_price < ema20:
        ema_structure += 0.50

    if ema9 < ema20:
        ema_structure += 0.50

    short_score += (
        10 * ema_structure
    )

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
        8 * min(
            1.0,
            macd_structure,
        )
    )

    volume_strength = 0.0

    if np.isfinite(
        opening_rvol
    ):

        volume_strength += (
            clamp(
                (
                    opening_rvol
                    - 1.0
                ) / 1.5,
                0,
                1,
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
                ) / 1.5,
                0,
                1,
            )
            * 0.60
        )

    short_score += (
        8 * min(
            1.0,
            volume_strength,
        )
    )

    structure_score = 0.0

    if lower_low:
        structure_score += 0.60

    if bearish_confirmation:
        structure_score += 0.20

    if lower_wick_pct < 0.20:
        structure_score += 0.20

    short_score += (
        10 * min(
            1.0,
            structure_score,
        )
    )

    if np.isfinite(
        range_expansion
    ):

        short_score += (
            5 * clamp(
                (
                    range_expansion
                    - 0.75
                ) / 0.75,
                0,
                1,
            )
        )

    if np.isfinite(
        normalized_total_move
    ):

        short_score += (
            5 * clamp(
                -normalized_total_move
                / 2.0,
                0,
                1,
            )
        )

    if short_continuation:
        short_score += 6

    if momentum_acceleration:
        short_score += 4

    if short_reversal:
        short_score -= 25

    short_score -= (
        exhaustion_penalty
    )

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

    long_score += (
        12 * bullish_move_score(
            opening_pct,
            1.50,
        )
    )

    long_score += (
        12 * bullish_move_score(
            confirmation_pct,
            1.00,
        )
    )

    long_score += (
        12 * bullish_move_score(
            relative_total_strength,
            1.50,
        )
    )

    long_score += (
        10 * clamp(
            vwap_pct / 2.0,
            0,
            1,
        )
    )

    long_ema_structure = 0.0

    if decision_price > ema20:
        long_ema_structure += 0.50

    if ema9 > ema20:
        long_ema_structure += 0.50

    long_score += (
        10 * long_ema_structure
    )

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
        8 * min(
            1.0,
            long_macd_structure,
        )
    )

    long_volume_strength = 0.0

    if np.isfinite(
        opening_rvol
    ):

        long_volume_strength += (
            clamp(
                (
                    opening_rvol
                    - 1.0
                ) / 1.5,
                0,
                1,
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
                ) / 1.5,
                0,
                1,
            )
            * 0.60
        )

    long_score += (
        8 * min(
            1.0,
            long_volume_strength,
        )
    )

    long_structure = 0.0

    if higher_high:
        long_structure += 0.60

    if bullish_confirmation:
        long_structure += 0.20

    if upper_wick_pct < 0.20:
        long_structure += 0.20

    long_score += (
        10 * min(
            1.0,
            long_structure,
        )
    )

    if np.isfinite(
        range_expansion
    ):

        long_score += (
            5 * clamp(
                (
                    range_expansion
                    - 0.75
                ) / 0.75,
                0,
                1,
            )
        )

    if np.isfinite(
        normalized_total_move
    ):

        long_score += (
            5 * clamp(
                normalized_total_move / 2.0,
                0,
                1,
            )
        )

    if long_continuation:
        long_score += 6

    if long_reversal:
        long_score -= 25

    if 45 <= rsi <= 70:
        long_score += 5

    elif rsi > 78:
        long_score -= 6

    short_score = clamp(
        short_score,
        0,
        100,
    )

    long_score = clamp(
        long_score,
        0,
        100,
    )

    turnover_cr = (
        float(
            confirmation["Volume"]
        )
        * decision_price
        / 10_000_000
    )

    signal = "AVOID"
    conviction = "NEUTRAL"

    if (
        short_score
        >= SHORT_TRADEABLE_SCORE
        and short_score >= long_score
        and total_pct
        <= -MIN_SHORT_TOTAL_MOVE
        and opening_pct
        <= -MIN_OPENING_SHORT_MOVE
        and not short_reversal
    ):

        signal = "SHORT"

        score = short_score

        conviction = (
            "HIGH"
            if short_score
            >= SHORT_HIGH_CONVICTION_SCORE
            else "TRADEABLE"
        )

    elif (
        long_score
        >= LONG_TRADEABLE_SCORE
        and long_score > short_score
        and total_pct
        >= MIN_LONG_TOTAL_MOVE
        and opening_pct
        >= MIN_OPENING_LONG_MOVE
        and not long_reversal
    ):

        signal = "LONG"

        score = long_score

        conviction = (
            "HIGH"
            if long_score
            >= LONG_HIGH_CONVICTION_SCORE
            else "TRADEABLE"
        )

    else:

        score = max(
            short_score,
            long_score,
        )

    return {
        "Ticker": ticker.replace(
            ".NS",
            "",
        ),
        "Signal": signal,
        "Conviction": conviction,
        "Score": round(
            score,
            2,
        ),
        "Short Score": round(
            short_score,
            2,
        ),
        "Long Score": round(
            long_score,
            2,
        ),
        "Opening %": opening_pct,
        "09:30-09:45 %": confirmation_pct,
        "Total %": total_pct,
        "NIFTY Total %": nifty_total_return,
        "Relative Total %": relative_total_strength,
        "Price": decision_price,
        "VWAP %": vwap_pct,
        "EMA20 %": ema20_pct,
        "EMA9 %": ema9_pct,
        "EMA Spread %": ema_spread_pct,
        "ATR %": atr_pct,
        "ATR Move": normalized_total_move,
        "Opening RVOL": opening_rvol,
        "Confirmation RVOL": confirmation_rvol,
        "RSI": rsi,
        "MACD Hist": macd_hist,
        "MACD": (
            "Bullish"
            if macd > macd_signal
            else "Bearish"
        ),
        "Turnover Cr": turnover_cr,
        "Lower Low": lower_low,
        "Higher High": higher_high,
        "Range Expansion": range_expansion,
        "Candle Strength": candle_strength,
        "Lower Wick": lower_wick_pct,
        "Upper Wick": upper_wick_pct,
        "Exhaustion Penalty": exhaustion_penalty,
        "Exhaustion Flags": ",".join(
            exhaustion_flags
        ),
        "Decision Timestamp": confirmation.name,
    }


# ================================================================
# LIVE SCANNER
# ================================================================

def scan_live(
    target_date: pd.Timestamp
) -> Tuple[pd.DataFrame, ScanStats]:

    stats = ScanStats()

    universe = build_universe()

    stats.universe = len(universe)

    logger.info(
        "Universe: %d stocks",
        len(universe),
    )

    if not universe:
        raise RuntimeError(
            "NSE universe is empty."
        )

    # ------------------------------------------------------------
    # NIFTY
    # ------------------------------------------------------------

    logger.info(
        "Downloading NIFTY benchmark..."
    )

    nifty = download_nifty_data(
        target_date
    )

    if nifty.empty:
        raise RuntimeError(
            "NIFTY benchmark data unavailable."
        )

    (
        nifty_opening_return,
        nifty_total_return,
    ) = calculate_market_returns(
        nifty,
        target_date,
    )

    if np.isfinite(
        nifty_total_return
    ):

        logger.info(
            "NIFTY 09:45 return: %+0.2f%%",
            nifty_total_return,
        )

    else:

        raise RuntimeError(
            "NIFTY 09:15/09:30 signal bars unavailable."
        )

    # ------------------------------------------------------------
    # STOCKS
    # ------------------------------------------------------------

    all_results = []

    total = len(universe)

    processed = 0

    for start_idx in range(
        0,
        total,
        BATCH_SIZE,
    ):

        batch = universe[
            start_idx:
            start_idx + BATCH_SIZE
        ]

        batch_data = (
            download_intraday_batch(
                batch,
                target_date,
            )
        )

        stats.downloaded += len(
            batch_data
        )

        stats.failed_download += (
            len(batch)
            - len(batch_data)
        )

        for ticker in batch:

            processed += 1

            logger.info(
                "Scanning %d/%d: %s",
                processed,
                total,
                ticker,
            )

            raw_df = batch_data.get(
                ticker
            )

            if raw_df is None:

                logger.warning(
                    "No data for %s",
                    ticker,
                )

                continue

            try:

                analysis = analyse_at_0945(
                    ticker,
                    raw_df,
                    target_date,
                    nifty_opening_return,
                    nifty_total_return,
                )

                if analysis is None:
                    continue

                stats.analyzed += 1

                if (
                    analysis["Turnover Cr"]
                    < MIN_TURNOVER_CR
                ):

                    stats.rejected_liquidity += 1

                    continue

                if analysis["Signal"] not in (
                    "SHORT",
                    "LONG",
                ):

                    continue

                all_results.append(
                    analysis
                )

            except Exception as exc:

                logger.exception(
                    "Analysis failed for %s: %s",
                    ticker,
                    exc,
                )

        if (
            start_idx + BATCH_SIZE
            < total
        ):

            time.sleep(
                BATCH_DELAY
            )

    if not all_results:

        logger.info(
            "No actionable signals."
        )

        return (
            pd.DataFrame(),
            stats,
        )

    results = pd.DataFrame(
        all_results
    )

    results["Rank Score"] = np.where(
        results["Signal"] == "SHORT",
        results["Short Score"],
        results["Long Score"],
    )

    results = (
        results
        .sort_values(
            "Rank Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    stats.actionable = len(
        results
    )

    stats.shorts = int(
        (
            results["Signal"]
            == "SHORT"
        ).sum()
    )

    stats.longs = int(
        (
            results["Signal"]
            == "LONG"
        ).sum()
    )

    return (
        results,
        stats,
    )


# ================================================================
# TELEGRAM FORMATTER
# ================================================================

def safe_float(
    value,
    digits=2
) -> str:

    if pd.isna(value):
        return "-"

    return f"{float(value):.{digits}f}"


def build_telegram_message(
    results: pd.DataFrame,
    target_date: pd.Timestamp,
    stats: ScanStats,
) -> str:

    timestamp = pd.Timestamp.now(
        tz=MARKET_TZ
    )

    lines = [
        "📊 NSE 09:45 SCANNER",
        "",
        f"📅 {target_date:%d-%b-%Y}",
        f"⏱ Executed: {timestamp:%H:%M:%S} IST",
        "",
    ]

    if results.empty:

        lines.extend([
            "⚪ NO ACTIONABLE SIGNALS",
            "",
            "No stock passed the configured "
            "09:45 criteria.",
        ])

    else:

        shorts = (
            results[
                results["Signal"]
                == "SHORT"
            ]
            .sort_values(
                "Short Score",
                ascending=False,
            )
            .head(
                TOP_SHORTS_TO_SHOW
            )
        )

        longs = (
            results[
                results["Signal"]
                == "LONG"
            ]
            .sort_values(
                "Long Score",
                ascending=False,
            )
            .head(
                TOP_LONGS_TO_SHOW
            )
        )

        lines.append(
            "🔴 SHORT CANDIDATES"
        )

        if shorts.empty:

            lines.append("None")

        else:

            for rank, (
                _,
                row,
            ) in enumerate(
                shorts.iterrows(),
                start=1,
            ):

                lines.append(
                    f"{rank}. "
                    f"{row['Ticker']} | "
                    f"S={row['Short Score']:.0f} | "
                    f"Move={row['Total %']:+.2f}% | "
                    f"RVOL="
                    f"{safe_float(row['Confirmation RVOL'], 2)}x | "
                    f"RSI="
                    f"{safe_float(row['RSI'], 1)}"
                )

                if row[
                    "Exhaustion Flags"
                ]:

                    lines.append(
                        "   ⚠ "
                        + row[
                            "Exhaustion Flags"
                        ]
                    )

        lines.append("")

        lines.append(
            "🟢 LONG CANDIDATES"
        )

        if longs.empty:

            lines.append("None")

        else:

            for rank, (
                _,
                row,
            ) in enumerate(
                longs.iterrows(),
                start=1,
            ):

                lines.append(
                    f"{rank}. "
                    f"{row['Ticker']} | "
                    f"S={row['Long Score']:.0f} | "
                    f"Move={row['Total %']:+.2f}% | "
                    f"RVOL="
                    f"{safe_float(row['Confirmation RVOL'], 2)}x | "
                    f"RSI="
                    f"{safe_float(row['RSI'], 1)}"
                )

        lines.extend([
            "",
            "────────────────────",
        ])

    lines.extend([
        f"Universe: {stats.universe}",
        f"Downloaded: {stats.downloaded}",
        f"Download failures: "
        f"{stats.failed_download}",
        f"Analyzed: {stats.analyzed}",
        f"Liquidity rejected: "
        f"{stats.rejected_liquidity}",
        f"Actionable: {stats.actionable}",
        f"Shorts: {stats.shorts}",
        f"Longs: {stats.longs}",
        "",
        "Research scanner only. "
        "Signal ≠ guaranteed return.",
    ])

    return "\n".join(lines)


# ================================================================
# PUBLIC ENTRY POINT
# ================================================================

def run_live_scan() -> pd.DataFrame:

    # ------------------------------------------------------------
    # Validate configuration BEFORE expensive downloads.
    # ------------------------------------------------------------

    validate_configuration()

    now = pd.Timestamp.now(
        tz=MARKET_TZ
    )

    target_date = (
        now
        .tz_localize(None)
        .normalize()
    )

    logger.info(
        "Live scanner started: "
        "%s",
        now.strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
    )

    start_time = time.monotonic()

    results, stats = scan_live(
        target_date
    )

    elapsed = (
        time.monotonic()
        - start_time
    )

    logger.info(
        "Scan completed in %.2f seconds.",
        elapsed,
    )

    message = build_telegram_message(
        results,
        target_date,
        stats,
    )

    print()
    print(message)
    print()

    # Exactly one Telegram message.
    send_telegram_message(
        message
    )

    return results
