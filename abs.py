"""
====================================================================
NSE 09:45 AUTOMATIC MOMENTUM / REVERSAL TELEGRAM SCANNER
====================================================================

LIVE ONLY
---------

The script automatically runs every NSE trading day around 09:45 AM IST.

It:

    1. Downloads Nifty 500 constituents.
    2. Removes Nifty 50 constituents.
    3. Downloads 15-minute NSE data.
    4. Uses:
           09:15 - 09:30 opening candle
           09:30 - 09:45 confirmation candle
    5. Calculates:
           VWAP
           EMA 9
           EMA 20
           RSI 14
           MACD
           Relative Volume
           Candle Strength
           Opening-range structure
    6. Detects continuation/reversal.
    7. Gives extra weight to SHORT setups.
    8. Selects only the strongest candidates.
    9. Sends Telegram message.

TELEGRAM OUTPUT
---------------

🔴 TOP SHORT CANDIDATES

🟢 TOP LONG CANDIDATES

Nothing else is sent to Telegram.

IMPORTANT
---------

This is a research/scanning tool.
It is not financial advice and does not guarantee returns.

INSTALL
-------

pip install yfinance pandas numpy requests

TELEGRAM SETUP
--------------

Create a Telegram bot using BotFather.

Then set:

    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Either directly below or as environment variables.

Example Linux:

    export TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
    export TELEGRAM_CHAT_ID="YOUR_CHAT_ID"

Then:

    python3 intraday_0945_telegram.py

The program remains running and automatically scans every trading day.

====================================================================
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from datetime import datetime, time as dt_time
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
# TELEGRAM
# ====================================================================

# Recommended:
#
# export TELEGRAM_BOT_TOKEN="123456:ABC..."
# export TELEGRAM_CHAT_ID="-100123456789"
#
# Or put the values directly here.

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "PUT_YOUR_BOT_TOKEN_HERE"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    "PUT_YOUR_CHAT_ID_HERE"
)


# ====================================================================
# SCANNER SETTINGS
# ====================================================================

# Minimum turnover of confirmation candle.
MIN_TURNOVER_CR = 2.0

# Relative volume.
VOLUME_LOOKBACK = 8

# Indicators.
EMA_FAST = 9
EMA_SLOW = 20

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# ====================================================================
# SIGNAL SETTINGS
# ====================================================================

# These are intentionally not extremely restrictive.
#
# The goal is to produce a manageable shortlist rather than only
# one stock.

TRADEABLE_SCORE = 58

HIGH_CONVICTION_SCORE = 72

# Minimum total directional movement by 09:45.
MIN_DIRECTIONAL_MOVE = 0.30


# ====================================================================
# SHORT BIAS
# ====================================================================

# Because the scanner is primarily intended for short trades,
# short candidates receive a small ranking advantage.

SHORT_RANK_BONUS = 4.0

# Stronger preference for stocks that are already falling.

SHORT_MIN_OPENING_MOVE = -0.35
SHORT_STRONG_OPENING_MOVE = -0.75

SHORT_MIN_CONFIRMATION = -0.10
SHORT_STRONG_CONFIRMATION = -0.40


# ====================================================================
# NUMBER OF RESULTS
# ====================================================================

TOP_SHORTS = 7
TOP_LONGS = 5


# ====================================================================
# DOWNLOAD SETTINGS
# ====================================================================

BATCH_SIZE = 50

DOWNLOAD_TIMEOUT = 20

BATCH_DELAY = 0.8


# ====================================================================
# SCHEDULER SETTINGS
# ====================================================================

# Scan a few seconds after 09:45 so that the completed 09:30-09:45
# candle is more likely to be available.

SCAN_HOUR = 9
SCAN_MINUTE = 45
SCAN_SECOND = 10

# If Yahoo is slow or temporarily doesn't have the candle yet,
# retry several times.

MAX_SCAN_RETRIES = 4

RETRY_DELAY_SECONDS = 25


# ====================================================================
# NSE TRADING DAYS
# ====================================================================

# This is intentionally a small holiday list.
#
# Weekends are automatically excluded.
#
# Add NSE holidays here if desired.
#
# Format:
#     "YYYY-MM-DD"

NSE_HOLIDAYS = {
    # Example:
    # "2026-01-26",
}


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
# TELEGRAM
# ====================================================================

def telegram_configured() -> bool:

    if not TELEGRAM_BOT_TOKEN:
        return False

    if not TELEGRAM_CHAT_ID:
        return False

    if (
        TELEGRAM_BOT_TOKEN
        == "PUT_YOUR_BOT_TOKEN_HERE"
    ):
        return False

    if (
        TELEGRAM_CHAT_ID
        == "PUT_YOUR_CHAT_ID_HERE"
    ):
        return False

    return True


def send_telegram(
    message: str
) -> bool:

    if not telegram_configured():

        print()
        print(
            "Telegram is not configured."
        )

        print(
            "Set TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID."
        )

        return False

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ok", False):

            print(
                f"Telegram error: {data}"
            )

            return False

        print(
            "Telegram message sent."
        )

        return True

    except Exception as exc:

        print(
            f"Telegram send failed: {exc}"
        )

        return False


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
        "Referer": "https://www.nseindia.com/",
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
# COLUMN NORMALIZATION
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
            .tz_convert(MARKET_TZ)
        )

    else:

        df.index = (
            df.index
            .tz_localize(MARKET_TZ)
        )

    return df.sort_index()


# ====================================================================
# CLEAN TICKER DATA
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
# DOWNLOAD INTRADAY
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

                ticker_df = (
                    clean_ticker_data(
                        ticker_df
                    )
                )

                if not ticker_df.empty:

                    results[
                        ticker
                    ] = ticker_df

            except Exception:

                continue

    else:

        if len(tickers) == 1:

            ticker_df = (
                clean_ticker_data(
                    raw
                )
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

    return rsi.where(
        avg_loss != 0,
        100
    )


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

    candle_range = (
        df["High"]
        - df["Low"]
    )

    body = (
        df["Close"]
        - df["Open"]
    ).abs()

    df["Candle_Strength"] = (
        body
        / candle_range.replace(
            0,
            np.nan
        )
    )

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

    return df


# ====================================================================
# ANALYSE 09:45
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

    opening_price = float(
        opening["Open"]
    )

    confirmation_price = float(
        confirmation["Close"]
    )

    if opening_price <= 0:

        return None

    # ------------------------------------------------------------
    # Price movements
    # ------------------------------------------------------------

    opening_pct = (
        (
            float(opening["Close"])
            - opening_price
        )
        / opening_price
    ) * 100

    confirmation_pct = (
        (
            float(confirmation["Close"])
            - float(confirmation["Open"])
        )
        / float(confirmation["Open"])
    ) * 100

    total_pct = (
        (
            confirmation_price
            - opening_price
        )
        / opening_price
    ) * 100

    # ------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------

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
            confirmation[
                "Volume_Ratio"
            ]
        )
        if pd.notna(
            confirmation[
                "Volume_Ratio"
            ]
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

    # ------------------------------------------------------------
    # Relative location
    # ------------------------------------------------------------

    below_vwap = (
        confirmation_price
        < vwap
    )

    above_vwap = (
        confirmation_price
        > vwap
    )

    below_ema20 = (
        confirmation_price
        < ema20
    )

    above_ema20 = (
        confirmation_price
        > ema20
    )

    bearish_ema = (
        ema9 < ema20
    )

    bullish_ema = (
        ema9 > ema20
    )

    bearish_macd = (
        macd < macd_signal
    )

    bullish_macd = (
        macd > macd_signal
    )

    # ------------------------------------------------------------
    # Opening range
    # ------------------------------------------------------------

    opening_high = float(
        opening["High"]
    )

    opening_low = float(
        opening["Low"]
    )

    higher_high = (
        float(confirmation["High"])
        > opening_high
    )

    lower_low = (
        float(confirmation["Low"])
        < opening_low
    )

    bullish_confirmation = (
        confirmation["Close"]
        > confirmation["Open"]
    )

    bearish_confirmation = (
        confirmation["Close"]
        < confirmation["Open"]
    )

    # ------------------------------------------------------------
    # Reversal conditions
    # ------------------------------------------------------------

    short_reversal = (
        opening_pct < -0.70
        and confirmation_pct > 0.35
        and not lower_low
    )

    long_reversal = (
        opening_pct > 0.70
        and confirmation_pct < -0.35
        and not higher_high
    )

    # ------------------------------------------------------------
    # SHORT SCORE
    #
    # Deliberately gives more weight to falling stocks.
    # ------------------------------------------------------------

    short_score = 0.0

    if opening_pct <= SHORT_MIN_OPENING_MOVE:

        short_score += 8

    if opening_pct <= SHORT_STRONG_OPENING_MOVE:

        short_score += 6

    if confirmation_pct < SHORT_MIN_CONFIRMATION:

        short_score += 10

    if confirmation_pct <= SHORT_STRONG_CONFIRMATION:

        short_score += 6

    if total_pct <= -0.30:

        short_score += 7

    if total_pct <= -0.75:

        short_score += 5

    if below_vwap:

        short_score += 11

    if below_ema20:

        short_score += 8

    if bearish_ema:

        short_score += 8

    if bearish_macd:

        short_score += 7

    if not pd.isna(rsi):

        if 32 <= rsi <= 58:

            short_score += 7

        elif rsi < 32:

            # Still allow very weak stocks, but reduce bounce risk.
            short_score += 2

        elif rsi > 65:

            short_score -= 3

    if not pd.isna(volume_ratio):

        if volume_ratio >= 1.10:

            short_score += 7

        if volume_ratio >= 1.50:

            short_score += 5

        if volume_ratio >= 2.00:

            short_score += 3

    if lower_low:

        short_score += 9

    if bearish_confirmation:

        short_score += 5

    if not pd.isna(candle_strength):

        if candle_strength >= 0.50:

            short_score += 5

    # Strong continuation.
    if (
        opening_pct < 0
        and confirmation_pct < 0
        and lower_low
    ):

        short_score += 7

    # Penalize obvious bounce/reversal.
    if short_reversal:

        short_score -= 28

    # ------------------------------------------------------------
    # LONG SCORE
    # ------------------------------------------------------------

    long_score = 0.0

    if opening_pct >= 0.35:

        long_score += 8

    if opening_pct >= 0.75:

        long_score += 5

    if confirmation_pct > 0.10:

        long_score += 10

    if confirmation_pct >= 0.40:

        long_score += 6

    if total_pct >= 0.30:

        long_score += 7

    if total_pct >= 0.75:

        long_score += 5

    if above_vwap:

        long_score += 11

    if above_ema20:

        long_score += 8

    if bullish_ema:

        long_score += 8

    if bullish_macd:

        long_score += 7

    if not pd.isna(rsi):

        if 45 <= rsi <= 70:

            long_score += 7

        elif rsi > 75:

            long_score -= 5

    if not pd.isna(volume_ratio):

        if volume_ratio >= 1.10:

            long_score += 7

        if volume_ratio >= 1.50:

            long_score += 5

        if volume_ratio >= 2.00:

            long_score += 3

    if higher_high:

        long_score += 9

    if bullish_confirmation:

        long_score += 5

    if not pd.isna(candle_strength):

        if candle_strength >= 0.50:

            long_score += 5

    if (
        opening_pct > 0
        and confirmation_pct > 0
        and higher_high
    ):

        long_score += 7

    if long_reversal:

        long_score -= 28

    # ------------------------------------------------------------
    # Clamp
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Determine signal
    # ------------------------------------------------------------

    signal = "AVOID"

    score = 0.0

    conviction = "NEUTRAL"

    # Strong short gets priority.
    if (
        short_score >= TRADEABLE_SCORE
        and short_score > long_score
        and total_pct <= -MIN_DIRECTIONAL_MOVE
        and not short_reversal
    ):

        signal = "SHORT"

        score = short_score

    elif (
        long_score >= TRADEABLE_SCORE
        and long_score > short_score
        and total_pct >= MIN_DIRECTIONAL_MOVE
        and not long_reversal
    ):

        signal = "LONG"

        score = long_score

    # ------------------------------------------------------------
    # Conviction
    # ------------------------------------------------------------

    if signal != "AVOID":

        conviction = (
            "HIGH"
            if score >= HIGH_CONVICTION_SCORE
            else "TRADEABLE"
        )

    # ------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------

    turnover_cr = (
        float(
            confirmation["Volume"]
        )
        * confirmation_price
        / 10_000_000
    )

    if turnover_cr < MIN_TURNOVER_CR:

        return None

    # ------------------------------------------------------------
    # Ranking score
    #
    # Give shorts a small advantage.
    # ------------------------------------------------------------

    ranking_score = score

    if signal == "SHORT":

        ranking_score += SHORT_RANK_BONUS

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

        "Ranking": round(
            ranking_score,
            1
        ),

        "Opening %": opening_pct,

        "Confirmation %": confirmation_pct,

        "Total %": total_pct,

        "Price": confirmation_price,

        "VWAP %": (
            (
                confirmation_price
                - vwap
            )
            / vwap
        ) * 100,

        "EMA20 %": (
            (
                confirmation_price
                - ema20
            )
            / ema20
        ) * 100,

        "Volume x": volume_ratio,

        "RSI": rsi,

        "MACD": (
            "B"
            if bullish_macd
            else "S"
        ),

        "Turnover Cr": turnover_cr,

        "Timestamp": confirmation.name,
    }


# ====================================================================
# SCAN DATE
# ====================================================================

def scan_date(
    target_date: pd.Timestamp
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

                # Need the two completed candles.
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

                if analysis["Signal"] not in (
                    "SHORT",
                    "LONG"
                ):

                    continue

                all_results.append(
                    analysis
                )

            except Exception:

                continue

        time.sleep(
            BATCH_DELAY
        )

    clear_line()

    return pd.DataFrame(
        all_results
    )


# ====================================================================
# FORMAT TELEGRAM
# ====================================================================

def safe_number(
    value,
    decimals=2
) -> str:

    if pd.isna(value):

        return "-"

    return f"{float(value):.{decimals}f}"


def format_candidate(
    row: pd.Series,
    rank: int
) -> str:

    ticker = str(
        row["Ticker"]
    )

    score = safe_number(
        row["Score"],
        0
    )

    opening = safe_number(
        row["Opening %"],
        2
    )

    confirmation = safe_number(
        row["Confirmation %"],
        2
    )

    total = safe_number(
        row["Total %"],
        2
    )

    price = safe_number(
        row["Price"],
        2
    )

    volume = safe_number(
        row["Volume x"],
        2
    )

    rsi = safe_number(
        row["RSI"],
        1
    )

    return (
        f"<b>{rank}. {ticker}</b>  "
        f"<b>{score}%</b>\n"
        f"   Price: ₹{price}\n"
        f"   Open: {opening:+}%\n"
        f"   Confirm: {confirmation:+}%\n"
        f"   Total: {total:+}%\n"
        f"   Volume: {volume}x\n"
        f"   RSI: {rsi}"
    )


def build_telegram_message(
    results: pd.DataFrame,
    target_date: pd.Timestamp
) -> str:

    date_text = target_date.strftime(
        "%d-%m-%Y"
    )

    lines = []

    lines.append(
        f"<b>NSE 09:45 SCANNER</b>\n"
        f"{date_text}"
    )

    lines.append("")

    # ================================================================
    # SHORTS
    # ================================================================

    lines.append(
        "<b>🔴 TOP SHORT CANDIDATES</b>"
    )

    shorts = results[
        results["Signal"] == "SHORT"
    ].copy()

    if not shorts.empty:

        shorts = shorts.sort_values(
            "Ranking",
            ascending=False
        ).head(
            TOP_SHORTS
        )

        for rank, (_, row) in enumerate(
            shorts.iterrows(),
            start=1
        ):

            lines.append(
                format_candidate(
                    row,
                    rank
                )
            )

            lines.append("")

    else:

        lines.append(
            "No strong short setup."
        )

        lines.append("")

    # ================================================================
    # LONGS
    # ================================================================

    lines.append(
        "<b>🟢 TOP LONG CANDIDATES</b>"
    )

    longs = results[
        results["Signal"] == "LONG"
    ].copy()

    if not longs.empty:

        longs = longs.sort_values(
            "Ranking",
            ascending=False
        ).head(
            TOP_LONGS
        )

        for rank, (_, row) in enumerate(
            longs.iterrows(),
            start=1
        ):

            lines.append(
                format_candidate(
                    row,
                    rank
                )
            )

            lines.append("")

    else:

        lines.append(
            "No strong long setup."
        )

        lines.append("")

    lines.append(
        "<i>Signal generated using data "
        "through 09:45 IST.</i>"
    )

    return "\n".join(
        lines
    )


# ====================================================================
# TRADING DAY CHECK
# ====================================================================

def is_trading_day(
    date_value: pd.Timestamp
) -> bool:

    # Saturday.
    if date_value.weekday() == 5:

        return False

    # Sunday.
    if date_value.weekday() == 6:

        return False

    date_string = (
        date_value.strftime(
            "%Y-%m-%d"
        )
    )

    if date_string in NSE_HOLIDAYS:

        return False

    return True


# ====================================================================
# CURRENT IST
# ====================================================================

def now_ist() -> pd.Timestamp:

    return pd.Timestamp.now(
        tz=MARKET_TZ
    )


# ====================================================================
# WAIT UNTIL 09:45
# ====================================================================

def wait_until_scan_time() -> None:

    while True:

        now = now_ist()

        if not is_trading_day(
            now
        ):

            # Check again tomorrow.
            time.sleep(
                60
            )

            continue

        target = now.normalize() + pd.Timedelta(
            hours=SCAN_HOUR,
            minutes=SCAN_MINUTE,
            seconds=SCAN_SECOND
        )

        if now >= target:

            return

        remaining = (
            target - now
        ).total_seconds()

        minutes = int(
            remaining // 60
        )

        seconds = int(
            remaining % 60
        )

        progress_line(
            f"Waiting for 09:45 IST | "
            f"{minutes:02d}:{seconds:02d}"
        )

        time.sleep(
            min(
                5,
                max(
                    1,
                    remaining
                )
            )
        )


# ====================================================================
# WAIT FOR COMPLETED CANDLE
# ====================================================================

def wait_for_market_data() -> None:

    print()

    print(
        "Waiting for 09:30-09:45 candle..."
    )

    # Small safety delay.
    time.sleep(
        10
    )


# ====================================================================
# RUN ONE SCAN
# ====================================================================

def run_scan() -> bool:

    today = now_ist()

    target_date = (
        today
        .tz_localize(None)
        .normalize()
    )

    print()
    print("=" * 80)

    print(
        f"AUTOMATIC NSE SCAN "
        f"{today.strftime('%Y-%m-%d')}"
    )

    print(
        f"Time: {today.strftime('%H:%M:%S')} IST"
    )

    print("=" * 80)

    wait_for_market_data()

    for attempt in range(
        1,
        MAX_SCAN_RETRIES + 1
    ):

        print()

        print(
            f"Scan attempt "
            f"{attempt}/{MAX_SCAN_RETRIES}"
        )

        try:

            results = scan_date(
                target_date
            )

            # --------------------------------------------------------
            # If Yahoo has not returned enough data, retry.
            # --------------------------------------------------------

            if results.empty:

                if attempt < MAX_SCAN_RETRIES:

                    print(
                        "No candidates returned. "
                        "Retrying..."
                    )

                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

                    continue

                message = build_telegram_message(
                    results,
                    target_date
                )

                send_telegram(
                    message
                )

                return True

            # --------------------------------------------------------
            # Build Telegram message.
            # --------------------------------------------------------

            message = build_telegram_message(
                results,
                target_date
            )

            # --------------------------------------------------------
            # Send.
            # --------------------------------------------------------

            sent = send_telegram(
                message
            )

            if sent:

                print()
                print(
                    "Scan completed successfully."
                )

            return sent

        except Exception as exc:

            print()
            print(
                f"Scan error: {exc}"
            )

            if attempt < MAX_SCAN_RETRIES:

                print(
                    "Retrying..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

            else:

                error_message = (
                    "<b>⚠️ NSE SCANNER ERROR</b>\n\n"
                    f"{str(exc)[:700]}"
                )

                send_telegram(
                    error_message
                )

                return False

    return False


# ====================================================================
# DAILY LOOP
# ====================================================================

def run_forever() -> None:

    print()
    print("=" * 80)

    print(
        "NSE 09:45 TELEGRAM SCANNER"
    )

    print(
        "LIVE AUTOMATIC MODE"
    )

    print("=" * 80)

    print()

    print(
        f"Scheduled time: "
        f"{SCAN_HOUR:02d}:"
        f"{SCAN_MINUTE:02d}:"
        f"{SCAN_SECOND:02d} IST"
    )

    print(
        f"Top shorts: {TOP_SHORTS}"
    )

    print(
        f"Top longs : {TOP_LONGS}"
    )

    print()

    if telegram_configured():

        print(
            "Telegram: configured"
        )

    else:

        print(
            "Telegram: NOT configured"
        )

        print()
        print(
            "Set TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID."
        )

        print()

    last_scan_date = None

    while True:

        try:

            now = now_ist()

            current_date = (
                now.date()
            )

            # --------------------------------------------------------
            # Prevent duplicate scan.
            # --------------------------------------------------------

            if (
                last_scan_date
                == current_date
            ):

                # Already scanned today.
                time.sleep(
                    30
                )

                continue

            # --------------------------------------------------------
            # Wait for 09:45.
            # --------------------------------------------------------

            wait_until_scan_time()

            # --------------------------------------------------------
            # Run scanner.
            # --------------------------------------------------------

            today = now_ist()

            if (
                last_scan_date
                == today.date()
            ):

                continue

            success = run_scan()

            # Mark the day as processed even if there were
            # temporarily no candidates. This prevents repeated
            # Telegram messages.
            last_scan_date = today.date()

            print()

            if success:

                print(
                    "Today's scan finished."
                )

            else:

                print(
                    "Today's scan finished "
                    "with an error."
                )

            print()

            print(
                "Waiting for next trading day..."
            )

            # --------------------------------------------------------
            # Sleep until next loop.
            # --------------------------------------------------------

            time.sleep(
                60
            )

        except KeyboardInterrupt:

            print()
            print(
                "Scanner stopped by user."
            )

            return

        except Exception as exc:

            print()
            print(
                f"Main loop error: {exc}"
            )

            print(
                "Restarting loop..."
            )

            time.sleep(
                60
            )


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:

    run_forever()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()

