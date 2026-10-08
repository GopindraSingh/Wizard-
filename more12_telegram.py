from __future__ import annotations

from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import io
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests


# ================================================================
# MORE12 — UPSTOX API VERSION
# ================================================================
# Original scanner logic preserved.
# Yahoo Finance/yfinance has been removed.
# Market data is now supplied by Upstox V3.
#
# Telegram automation variant:
#   Always scans today's opening 15-minute candle (09:15-09:30 IST).
#   It does not expose historical-date or rolling-candle modes.
#
# Upstox market data is read-only; Telegram receives the scanner results.
# ================================================================

MARKET_TZ = "Asia/Kolkata"
MARKET_ZONE = ZoneInfo(MARKET_TZ)

UPSTOX_INTERVAL_MINUTES = 15
UPSTOX_INSTRUMENT_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)
UPSTOX_INSTRUMENT_CACHE = "upstox_nse_instruments.json"

UPSTOX_MAX_WORKERS = 20
UPSTOX_REQUESTS_PER_SECOND = 45
UPSTOX_REQUESTS_PER_MINUTE = 480
UPSTOX_REQUEST_TIMEOUT = 20
UPSTOX_MAX_RETRIES = 3
UPSTOX_DEBUG_ERRORS = False
UPSTOX_MAX_ERROR_LINES = 20

UPSTOX_ACCESS_TOKEN_ENV = "UPSTOX_ACCESS_TOKEN"

_UPSTOX_TOKEN: Optional[str] = None
_UPSTOX_SESSION_LOCAL = threading.local()
_UPSTOX_LIMITER = None
_UPSTOX_INSTRUMENT_MAP: Optional[Dict[str, str]] = None

# Internal run statistics consumed by the Telegram wrapper below.
LAST_SCAN_STATS = {
    "universe_size": 0,
    "symbols_with_target_day_candles": 0,
}


class UpstoxRateLimiter:
    """Thread-safe rate limiter used by the scanner's parallel requests."""

    def __init__(self, per_second: int, per_minute: int):
        self.per_second = per_second
        self.per_minute = per_minute
        self.lock = threading.Lock()
        self.starts = deque()

    def acquire(self) -> None:
        while True:
            sleep_for = 0.0

            with self.lock:
                now = time.monotonic()

                while self.starts and now - self.starts[0] >= 60.0:
                    self.starts.popleft()

                if len(self.starts) >= self.per_minute:
                    sleep_for = 60.0 - (now - self.starts[0]) + 0.001
                else:
                    recent_second = [
                        ts for ts in self.starts
                        if now - ts < 1.0
                    ]

                    if len(recent_second) >= self.per_second:
                        sleep_for = 1.0 - (now - recent_second[0]) + 0.001
                    else:
                        self.starts.append(now)
                        return

            time.sleep(max(sleep_for, 0.001))


def get_upstox_token() -> str:
    """Read the Upstox Analytics Token from environment; never prompt in CI."""
    global _UPSTOX_TOKEN, _UPSTOX_LIMITER

    if _UPSTOX_TOKEN:
        return _UPSTOX_TOKEN

    token = os.getenv(UPSTOX_ACCESS_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            "Missing UPSTOX_ACCESS_TOKEN. Add your Upstox Analytics Token "
            "as a GitHub Actions secret named UPSTOX_ACCESS_TOKEN."
        )

    _UPSTOX_TOKEN = token
    _UPSTOX_LIMITER = UpstoxRateLimiter(
        UPSTOX_REQUESTS_PER_SECOND,
        UPSTOX_REQUESTS_PER_MINUTE,
    )
    return _UPSTOX_TOKEN


def get_upstox_session() -> requests.Session:
    session = getattr(_UPSTOX_SESSION_LOCAL, "session", None)

    if session is None:
        session = requests.Session()
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        _UPSTOX_SESSION_LOCAL.session = session

    return session


def upstox_request(url: str) -> dict:
    """Authenticated GET with retry handling for transient API failures."""
    token = get_upstox_token()

    if _UPSTOX_LIMITER is None:
        raise RuntimeError("Upstox rate limiter is not initialized.")

    last_error = None

    for attempt in range(UPSTOX_MAX_RETRIES):
        _UPSTOX_LIMITER.acquire()

        try:
            response = get_upstox_session().get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                },
                timeout=UPSTOX_REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                payload = response.json()

                if payload.get("status") != "success":
                    raise RuntimeError(
                        f"Upstox returned non-success response: {payload}"
                    )

                return payload

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = RuntimeError(
                    f"Upstox HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
                if attempt < UPSTOX_MAX_RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue

            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]

            raise RuntimeError(
                f"Upstox HTTP {response.status_code}: {detail}"
            )

        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < UPSTOX_MAX_RETRIES - 1:
                time.sleep(1.0 * (attempt + 1))
                continue

    raise RuntimeError(f"Upstox request failed: {last_error}")


def download_upstox_instruments() -> Dict[str, str]:
    """Build trading-symbol -> NSE equity instrument-key mapping."""
    global _UPSTOX_INSTRUMENT_MAP

    if _UPSTOX_INSTRUMENT_MAP is not None:
        return _UPSTOX_INSTRUMENT_MAP

    cache_path = UPSTOX_INSTRUMENT_CACHE
    records = None

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
        except Exception:
            records = None

    if records is None:
        response = requests.get(
            UPSTOX_INSTRUMENT_URL,
            timeout=30,
        )
        response.raise_for_status()

        with gzip.GzipFile(
            fileobj=io.BytesIO(response.content)
        ) as gz:
            records = json.loads(
                gz.read().decode("utf-8")
            )

        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(records, handle)
        except Exception:
            pass

    mapping: Dict[str, str] = {}

    for item in records:
        if not isinstance(item, dict):
            continue

        if item.get("segment") != "NSE_EQ":
            continue

        symbol = str(
            item.get("trading_symbol", "")
        ).strip().upper()

        instrument_key = str(
            item.get("instrument_key", "")
        ).strip()

        if not symbol or not instrument_key:
            continue

        instrument_type = str(
            item.get("instrument_type", "")
        ).strip().upper()

        if symbol not in mapping:
            mapping[symbol] = instrument_key
        elif instrument_type == "EQ":
            mapping[symbol] = instrument_key

    if not mapping:
        raise RuntimeError(
            "Upstox NSE instrument master returned no NSE_EQ instruments."
        )

    _UPSTOX_INSTRUMENT_MAP = mapping

    print(
        f"✅ Upstox instrument map loaded: "
        f"{len(mapping)} NSE equities"
    )

    return mapping


def ticker_to_instrument_key(ticker: str) -> Optional[str]:
    symbol = (
        ticker
        .replace(".NS", "")
        .strip()
        .upper()
    )
    return download_upstox_instruments().get(symbol)


def parse_upstox_candles(payload: dict) -> pd.DataFrame:
    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    if not candles:
        return pd.DataFrame()

    rows = []

    for candle in candles:
        if len(candle) < 6:
            continue

        rows.append({
            "Timestamp": candle[0],
            "Open": candle[1],
            "High": candle[2],
            "Low": candle[3],
            "Close": candle[4],
            "Volume": candle[5],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Timestamp"]
    ).set_index("Timestamp")

    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
    ]

    return df.sort_index()


def download_upstox_current_day(
    instrument_key: str,
) -> pd.DataFrame:
    """Get current-session 15-minute candles from Upstox V3."""
    url = (
        "https://api.upstox.com/v3/"
        "historical-candle/intraday/"
        f"{requests.utils.quote(instrument_key, safe='')}/"
        f"minutes/{UPSTOX_INTERVAL_MINUTES}"
    )

    payload = upstox_request(url)
    return parse_upstox_candles(payload)


def download_upstox_historical_day(
    instrument_key: str,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Get 15-minute candles for one historical trading date."""
    to_date = target_date.normalize()
    from_date = to_date - pd.Timedelta(days=1)

    url = (
        "https://api.upstox.com/v3/"
        "historical-candle/"
        f"{requests.utils.quote(instrument_key, safe='')}/"
        f"minutes/{UPSTOX_INTERVAL_MINUTES}/"
        f"{to_date:%Y-%m-%d}/{from_date:%Y-%m-%d}"
    )

    payload = upstox_request(url)
    return parse_upstox_candles(payload)


def is_current_market_date(target_date: pd.Timestamp) -> bool:
    now = datetime.now(MARKET_ZONE)
    return target_date.date() == now.date()


def download_one_ticker(
    ticker: str,
    mode: str,
    target_date: pd.Timestamp,
) -> Tuple[str, pd.DataFrame, Optional[str]]:
    """Download exactly the dataset required for one scanner symbol."""
    try:
        instrument_key = ticker_to_instrument_key(ticker)

        if not instrument_key:
            return ticker, pd.DataFrame(), "instrument_key not found"

        if mode in {"1", "3"}:
            df = download_upstox_current_day(
                instrument_key
            )
        else:
            df = download_upstox_historical_day(
                instrument_key,
                target_date,
            )

        if df.empty:
            return ticker, df, "Upstox returned no candles"

        return ticker, df, None

    except Exception as exc:
        return ticker, pd.DataFrame(), str(exc)


def download_scanner_data(
    tickers: List[str],
    mode: str,
    target_date: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Parallel Upstox data download with shared rate limiting."""
    results: Dict[str, pd.DataFrame] = {}
    errors: List[Tuple[str, str]] = []

    print(
        f"⏳ Downloading 15m Upstox data for "
        f"{len(tickers)} equities..."
    )

    with ThreadPoolExecutor(
        max_workers=UPSTOX_MAX_WORKERS
    ) as executor:
        futures = [
            executor.submit(
                download_one_ticker,
                ticker,
                mode,
                target_date,
            )
            for ticker in tickers
        ]

        completed = 0

        for future in as_completed(futures):
            completed += 1

            try:
                ticker, df, error = future.result()
            except Exception as exc:
                ticker, df, error = "UNKNOWN", pd.DataFrame(), str(exc)

            if error:
                errors.append((ticker, error))

            if not df.empty:
                results[ticker] = df

            if completed % 25 == 0 or completed == len(futures):
                sys.stdout.write(
                    f"\r   Progress: "
                    f"{completed}/{len(futures)}"
                )
                sys.stdout.flush()

    if UPSTOX_DEBUG_ERRORS and errors:
        from collections import Counter
        summary = Counter(err for _, err in errors)
        print(f"⚠️ Symbols without usable data: {len(errors)}")
        print("   Common causes:")
        for reason, count in summary.most_common(5):
            print(f"   - {count}x {reason}")
        print("   Sample errors:")
        for ticker, error in errors[:UPSTOX_MAX_ERROR_LINES]:
            print(f"   - {ticker}: {error}")

    return results


def normalize_current_day_candles(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only target-day candles and normalize timestamps."""
    if df.empty:
        return df

    out = df.copy()

    out = out[
        out.index.date == target_date.date()
    ].sort_index()

    return out


def get_latest_yahoo_compatible_candle(
    df: pd.DataFrame,
) -> Optional[pd.Series]:
    """
    Preserve the original more12.py Mode 3 behavior exactly.

    The Yahoo implementation selected the final row returned for today
    with ``day_candles.iloc[[-1]]``. It did NOT remove a currently forming
    15-minute candle. We intentionally preserve that behavior here so that
    changing the data vendor does not silently change the strategy logic.
    """
    if df.empty:
        return None

    return df.iloc[-1]


def get_opening_15m_candle(
    df: pd.DataFrame,
) -> Optional[pd.Series]:
    """Return the completed 09:15-09:30 candle only."""
    if df.empty:
        return None

    opening = df[
        (df.index.hour == 9)
        & (df.index.minute == 15)
    ]

    if opening.empty:
        return None

    return opening.iloc[0]




def get_dynamic_nse_universe() -> List[str]:
    """Dynamically pulls a broad, liquid stock universe with your comprehensive pool."""
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/AnishDe1202/nifty-stocks-data/master/nifty500.json",
            timeout=5,
        )
        if response.status_code == 200:
            symbols = response.json()
            if isinstance(symbols, list) and len(symbols) > 0:
                return [f"{s}.NS" for s in symbols]
    except Exception:
        pass

    automated_pool = [
        "AARTIDRUGS", "AAVAS", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ABSLAMC", "ACC", "ACADEMY", 
        "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ATGL", "ADANIPOWER", "ABCAS", "AegisLOG", "AFFLE", 
        "AIAENG", "AJANTPHARM", "APLAPOLLO", "ALKEM", "ALKYLAMINE", "ALLCARGO", "AMARAJABAT", 
        "AMBUJACEM", "ANANDRATHI", "ANGELONE", "ANURAS", "APARINDS", "APOLLOHOSP", "APOLLOTYRE", 
        "APTUS", "ASAHIINDIA", "ASHOKLEY", "ASIANPAINT", "ASTERDM", "ASTRAZEN", "ASTRAL", "ATUL", 
        "AUBANK", "AUROPHARMA", "AVAS", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", 
        "BAJAJHLDNG", "BALAMINES", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", 
        "BANKINDIA", "BATAINDIA", "BAYERCROP", "BBL", "BDL", "BEL", "BEML", "BEPL", "BERGEPAINT", 
        "BFUTILITIE", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BIRLACORPN", "BSOFT", "BLS", 
        "BLUESTARCO", "BORORENEW", "BOSCHLTD", "BPCL", "BRIGADE", "BRITANNIA", "MAPMYINDIA", "BSE", 
        "BURGERKING", "CAMPUS", "CANBK", "CANFINHOME", "CAPLIPOINT", "CARBORUNIV", "CASTROLIND", 
        "CEATLTD", "CELEBRITY", "CENTRALBK", "CDSL", "CENTURYPLY", "CERA", "CESC", "CGCL", "CHALET", 
        "CHAMBLFERT", "CHOLAFIN", "CHOLAHLDNG", "CIPLA", "CUB", "CIEINDIA", "COALINDIA", "COCHINSHIP", 
        "COFORGE", "COLPAL", "CAMS", "CONCOR", "COROMANDEL", "CRAFTSMAN", "CREDITACC", "CROMPTON", 
        "CUMMINSIND", "CYIENT", "DABUR", "DalBHARAT", "DATAPATTNS", "DBL", "DCBBANK", "DCMSHRIRAM", 
        "DEEPAKFERT", "DEEPAKNTR", "DELHIVERY", "DEVYANI", "DIVISLAB", "DIXON", "LALPATHLAB", 
        "DRREDDY", "EIDPARRY", "EIHOTEL", "EICHERMOT", "ELGIEQUIP", "EMAMILTD", "ENDURANCE", 
        "ESCORTS", "EXIDEIND", "NYKAA", "FEDERALBNK", "FACT", "FINEORG", "FINCABLES", "FINPIPE", 
        "FSL", "FIVESTAR", "FORTIS", "GAIL", "GALAXYSURF", "GARFIBRES", "GESHIP", "GHCL", "GICRE", 
        "GILLETTE", "GLAND", "GLAXO", "GLENMARK", "MEDANTA", "GOCOLORS", "GODREJCP", "GODREJIND", 
        "GODREJPROP", "GRANULES", "GRASIM", "GRAVITA", "GRINDWELL", "GUJGASLTD", "GNFC", "GPPL", 
        "GSFC", "GSPL", "HEG", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HFCL", "HATSUN", 
        "HAVELLS", "HCG", "HIL", "HEMIPROPERTIES", "HINDALCO", "HINDCOPPER", "HINDPETRO", 
        "HINDUNILVR", "HINDZINC", "POWERMECH", "HSCL", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI", 
        "IDBI", "IDFC", "IDFCFIRSTB", "IEX", "IFBIND", "IIFL", "INDAMCO", "INDHOTEL", "INDIACEM", 
        "INDIAMART", "INDIANB", "INDOCO", "INDUSINDBK", "INDUSTOWER", "INFIBEAM", "INFY", "INGV", 
        "INSECTICID", "IOB", "IOC", "IPCALAB", "IRB", "IRCON", "IRCTC", "ITC", "ITI", "JANDJ", 
        "JCHAC", "JBCHEPHARM", "JKCEMENT", "JKIL", "JKLAKSHMI", "JKPAPER", "JMFINANCIL", "JSWENERGY", 
        "JSWSTEEL", "JTEKTINDIA", "JINDALSTEL", "JISLJALEQS", "JUBLFOOD", "JUBLINGRIA", "JUSTDIAL", 
        "JYOTHYLAB", "KAJARIACER", "KALPATPOWR", "KALYANKJIL", "KANSAINER", "KARURVYSYA", "KEC", 
        "KEI", "KNRCON", "KOTAKBANK", "KPRMILL", "KRBL", "KSCL", "KSB", "LODHA", "LTIM", "LTTS", 
        "LICHSGFIN", "LICI", "LINDEINDIA", "LUPIN", "LUXIND", "MMTC", "MOIL", "MRF", "MGL", "M&M", 
        "M&MFIN", "MAHABANK", "MAHICKM", "MAHLOG", "MANAPPURAM", "MRPL", "MARICO", "MARUTI", 
        "MASTEK", "MAXHEALTH", "MAZDOCK", "METROPOLIS", "MINDACORP", "MOTHERSON", "MPHASIS", "MCX", 
        "MUTHOOTFIN", "NESCO", "NESTLEIND", "NETWORK18", "NAM-INDIA", "NCC", "NLCINDIA", "NMDC", 
        "NTPC", "NH", "NUVAMA", "OBEROIRLTY", "ONGC", "OIL", "OLECTRA", "PAYTM", "OFSS", "PCJEWELLER", 
        "PEL", "PIIND", "PNBHOUSING", "PNCINFRA", "PVRINOX", "PageIND", "PERSISTENT", "PETRONET", 
        "PFIZER", "PHOENIXLTD", "PIDILITIND", "POLYCAB", "POONAWALLA", "PFC", "POWERGRID", "PRAJIND", 
        "PRESTIGE", "PRINCEPIPE", "PRSMJOHNSN", "PSS", "QUESS", "RBLBANK", "RECLTD", "RITES", "RADICO", 
        "RAIN", "RAJESHEXPO", "RALLIS", "RCF", "RELIANCE", "ROUTE", "SBICARD", "SBILIFE", "SBIN", 
        "SHREECEM", "SRF", "SANOFI", "SFL", "SHK", "SHOPERSTOP", "SHRIRAMFIN", "SIEMENS", "SOBHA", 
        "SOLARINDS", "SONACOMS", "SONATSOFTW", "SPARC", "STAR", "SBCL", "SUDARSCHEM", "SUMICHEM", 
        "SUNDARMFIN", "SUNDRMFAST", "SUNPHARMA", "SUNTV", "SUPRAJIT", "SUPREMEIND", "SUZLON", 
        "SWANENERGY", "SYMPHONY", "SYNGENE", "TVSMOTOR", "TATACHEM", "TATACOFFEE", "TATACOMM", "TCS", 
        "TATACONSUM", "TATAELXSI", "TATAINVEST", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TTML", 
        "TeamLease", "TECHM", "TECHNOE", "TEJASNET", "NIACL", "RAMCOCEM", "THERMAX", "THYROCARE", 
        "TIDEWATER", "TIMKEN", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT", "TRIVENI", 
        "TRITURBINE", "UCOBANK", "UFLEX", "UJJIVANSFB", "ULTRACEMCO", "UNICHEMLAB", "UPL", "UTIAMC", 
        "VGUARD", "VMART", "VODAFONE", "VOLTAS", "VRLLOG", "VSTIND", "WABAG", "WELCORP", "WELSPUNIND", 
        "WESTLIFE", "WHIRLPOOL", "WIPRO", "WOCKPHARMA", "YESBANK", "ZENSARTECH", "ZOMATO", "ZYDUSLIFE", 
        "ZYDUSWELL"
    ]
    return [f"{sym}.NS" for sym in automated_pool]



def scan_momentum_stocks(
    mode: str = "1",
    target_date_str: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scans the configured NSE universe using Upstox 15-minute candles.

    Mode 1:
        Current-day opening 15m candle.
    Mode 2:
        Historical opening 15m candle for supplied YYYY-MM-DD.
    Mode 3:
        Current-day latest completed 15m candle.
    """
    if mode in {"1", "3"}:
        current_date = datetime.now(
            MARKET_ZONE
        ).strftime("%Y-%m-%d")

    else:
        if not target_date_str:
            raise ValueError(
                "Historical mode requires a target date."
            )

        current_date = target_date_str

        datetime.strptime(
            current_date,
            "%Y-%m-%d",
        )

    global LAST_SCAN_STATS
    LAST_SCAN_STATS = {
        "universe_size": 0,
        "symbols_with_target_day_candles": 0,
    }

    target_date = pd.Timestamp(
        current_date
    ).normalize()

    now = datetime.now(MARKET_ZONE)

    # A live Upstox intraday request represents the current trading session.
    # Before 09:15 there is no current-session candle for today.  For Mode 1
    # there is additionally no completed opening candle before 09:30.
    if mode in {"1", "3"} and target_date.date() == now.date():
        if now < now.replace(hour=9, minute=15, second=0, microsecond=0):
            print(
                "⚠️ NSE market has not opened yet (09:15 IST). "
                "Live modes 1 and 3 have no current-day candle to scan."
            )
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

    print(
        f"📅 Scan Date: {current_date} | "
        f"Mode: "
        f"{'Latest Intraday Rolling' if mode == '3' else 'Opening Candle'}"
    )

    tickers = get_dynamic_nse_universe()
    LAST_SCAN_STATS["universe_size"] = len(tickers)

    print(
        f"📊 Universe: {len(tickers)} symbols"
    )

    # Load instrument master once before starting worker threads.
    download_upstox_instruments()

    data_by_ticker = download_scanner_data(
        tickers,
        mode,
        target_date,
    )

    all_losers = []
    shortlisted_losers = []
    all_gainers = []
    shortlisted_gainers = []
    symbols_with_target_day_candles = 0

    for ticker in tickers:
        try:
            ticker_df = data_by_ticker.get(
                ticker,
                pd.DataFrame(),
            )

            if ticker_df.empty:
                continue

            ticker_df = normalize_current_day_candles(
                ticker_df,
                target_date,
            )

            if ticker_df.empty:
                continue

            symbols_with_target_day_candles += 1

            # --------------------------------------------------------
            # Select the same logical candle as the original scanner.
            # --------------------------------------------------------
            if mode == "3":
                latest_candle = (
                    get_latest_yahoo_compatible_candle(
                        ticker_df,
                    )
                )

                candle_label = "Latest_15m_Close"

            else:
                latest_candle = (
                    get_opening_15m_candle(
                        ticker_df,
                    )
                )

                candle_label = "15m_Close_Entry"

                # Never use a still-forming opening candle in live mode.
                if mode == "1":
                    opening_end = now.replace(
                        hour=9,
                        minute=30,
                        second=0,
                        microsecond=0,
                    )

                    if (
                        target_date.date() == now.date()
                        and now < opening_end
                    ):
                        latest_candle = None

            if latest_candle is None:
                continue

            open_p = float(
                latest_candle["Open"]
            )
            close_p = float(
                latest_candle["Close"]
            )
            high_p = float(
                latest_candle["High"]
            )
            low_p = float(
                latest_candle["Low"]
            )
            vol = int(
                latest_candle["Volume"]
            )

            if open_p == 0:
                continue

            pct_change = (
                (close_p - open_p)
                / open_p
            ) * 100

            turnover_cr = (
                vol * close_p
            ) / 10_000_000

            candle_range = (
                high_p - low_p
            )

            close_location = (
                (close_p - low_p)
                / candle_range
                if candle_range > 0
                else 1.0
            )

            # --------------------------------------------------------
            # SHORT (SELLING) LOGIC — PRESERVED
            # --------------------------------------------------------
            if pct_change < 0:
                row_data_short = {
                    "Ticker": ticker.replace(
                        ".NS",
                        "",
                    ),
                    "Date": current_date,
                    "15m_Open": round(
                        open_p,
                        2,
                    ),
                    candle_label: round(
                        close_p,
                        2,
                    ),
                    "Candle_Drop_%": round(
                        pct_change,
                        2,
                    ),
                    "Turnover_Cr": round(
                        turnover_cr,
                        2,
                    ),
                    "1%_Profit_Target": round(
                        close_p * 0.99,
                        2,
                    ),
                    "Hard_Stop_Loss": round(
                        high_p,
                        2,
                    ),
                }

                # Tier 1: General Losers
                if (
                    pct_change < -0.5
                    and turnover_cr >= 4.0
                    and close_location <= 0.25
                ):
                    all_losers.append(
                        row_data_short
                    )

                # Tier 2: Shortlisted Most Falling
                if (
                    pct_change < -1.5
                    and turnover_cr >= 15.0
                    and close_location <= 0.15
                ):
                    shortlisted_losers.append(
                        row_data_short
                    )

            # --------------------------------------------------------
            # LONG (BUYING) LOGIC — PRESERVED
            # --------------------------------------------------------
            elif pct_change > 0:
                row_data_long = {
                    "Ticker": ticker.replace(
                        ".NS",
                        "",
                    ),
                    "Date": current_date,
                    "15m_Open": round(
                        open_p,
                        2,
                    ),
                    candle_label: round(
                        close_p,
                        2,
                    ),
                    "Candle_Gain_%": round(
                        pct_change,
                        2,
                    ),
                    "Turnover_Cr": round(
                        turnover_cr,
                        2,
                    ),
                    "1%_Profit_Target": round(
                        close_p * 1.01,
                        2,
                    ),
                    "Hard_Stop_Loss": round(
                        low_p,
                        2,
                    ),
                }

                # Tier 1: General Gainers
                if (
                    pct_change > 0.5
                    and turnover_cr >= 4.0
                    and close_location >= 0.75
                ):
                    all_gainers.append(
                        row_data_long
                    )

                # Tier 2: Shortlisted Most Rising
                if (
                    pct_change > 1.5
                    and turnover_cr >= 15.0
                    and close_location >= 0.85
                ):
                    shortlisted_gainers.append(
                        row_data_long
                    )

        except Exception:
            continue

    LAST_SCAN_STATS["symbols_with_target_day_candles"] = (
        symbols_with_target_day_candles
    )

    df_all_losers = pd.DataFrame(
        all_losers
    )

    if not df_all_losers.empty:
        df_all_losers = (
            df_all_losers
            .sort_values(
                by="Candle_Drop_%",
                ascending=True,
            )
            .reset_index(drop=True)
        )

    df_short_losers = pd.DataFrame(
        shortlisted_losers
    )

    if not df_short_losers.empty:
        df_short_losers = (
            df_short_losers
            .sort_values(
                by="Candle_Drop_%",
                ascending=True,
            )
            .reset_index(drop=True)
        )

    df_all_gainers = pd.DataFrame(
        all_gainers
    )

    if not df_all_gainers.empty:
        df_all_gainers = (
            df_all_gainers
            .sort_values(
                by="Candle_Gain_%",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    df_short_gainers = pd.DataFrame(
        shortlisted_gainers
    )

    if not df_short_gainers.empty:
        df_short_gainers = (
            df_short_gainers
            .sort_values(
                by="Candle_Gain_%",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return (
        df_all_losers,
        df_short_losers,
        df_all_gainers,
        df_short_gainers,
    )


# ====================================================================
# TELEGRAM DELIVERY — CURRENT-DAY OPENING CANDLE ONLY
# ====================================================================

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
TELEGRAM_API_TIMEOUT = 20
TELEGRAM_MAX_MESSAGE_CHARS = 3500


def _get_telegram_config() -> Tuple[str, str]:
    bot_token = os.getenv(TELEGRAM_BOT_TOKEN_ENV, "").strip()
    chat_id = os.getenv(TELEGRAM_CHAT_ID_ENV, "").strip()
    missing = []
    if not bot_token:
        missing.append(TELEGRAM_BOT_TOKEN_ENV)
    if not chat_id:
        missing.append(TELEGRAM_CHAT_ID_ENV)
    if missing:
        raise RuntimeError(
            "Missing Telegram GitHub Actions secret(s): " + ", ".join(missing)
        )
    return bot_token, chat_id


def send_telegram_message(text: str) -> None:
    """Send a plain-text Telegram message, splitting safely below API limits."""
    bot_token, chat_id = _get_telegram_config()
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = 0

    for line in (text or "").splitlines() or [""]:
        # Guard against one exceptionally long line.
        while len(line) > TELEGRAM_MAX_MESSAGE_CHARS - 100:
            part = line[: TELEGRAM_MAX_MESSAGE_CHARS - 100]
            line = line[TELEGRAM_MAX_MESSAGE_CHARS - 100 :]
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines, current_len = [], 0
            chunks.append(part)

        extra = len(line) + (1 if current_lines else 0)
        if current_lines and current_len + extra > TELEGRAM_MAX_MESSAGE_CHARS:
            chunks.append("\n".join(current_lines))
            current_lines, current_len = [], 0
        current_lines.append(line)
        current_len += len(line) + (1 if len(current_lines) > 1 else 0)

    if current_lines:
        chunks.append("\n".join(current_lines))

    for chunk in chunks:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            },
            timeout=TELEGRAM_API_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Telegram sendMessage failed (HTTP {response.status_code}): "
                f"{response.text[:300]}"
            )
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(
                f"Telegram rejected a message: {payload.get('description', 'unknown error')}"
            )


def _format_price(value) -> str:
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _format_signal_group(
    title: str,
    frame: pd.DataFrame,
    move_column: str,
    close_column: str,
) -> str:
    lines = [f"{title} ({len(frame)})"]
    if frame.empty:
        lines.append("No stocks matched.")
        return "\n".join(lines)

    lines.append("Symbol | Move | Turnover Cr | Open → Close | Target | Stop")
    for _, row in frame.iterrows():
        move = row.get(move_column, 0.0)
        try:
            move_text = f"{float(move):+.2f}%"
        except (TypeError, ValueError):
            move_text = "—"
        lines.append(
            f"{row.get('Ticker', '?')} | {move_text} | "
            f"{float(row.get('Turnover_Cr', 0.0)):.2f} | "
            f"{_format_price(row.get('15m_Open'))} → "
            f"{_format_price(row.get(close_column))} | "
            f"{_format_price(row.get('1%_Profit_Target'))} | "
            f"{_format_price(row.get('Hard_Stop_Loss'))}"
        )
    return "\n".join(lines)


def build_telegram_report(
    all_losers: pd.DataFrame,
    shortlisted_losers: pd.DataFrame,
    all_gainers: pd.DataFrame,
    shortlisted_gainers: pd.DataFrame,
    scan_date: str,
) -> str:
    stats = LAST_SCAN_STATS
    header = [
        "📊 MORE12 — OPENING 15-MIN SCANNER",
        f"Date: {scan_date} | Candle: 09:15–09:30 IST",
        f"Data: {stats.get('symbols_with_target_day_candles', 0)} / "
        f"{stats.get('universe_size', 0)} symbols with current-day candles",
        "",
    ]

    if not stats.get("symbols_with_target_day_candles", 0):
        header.extend([
            "⚠️ No current-day candles were available from Upstox.",
            "No signals were generated. This may be an NSE holiday or an API/token issue.",
            "On a normal trading day, check the GitHub Actions run logs and UPSTOX_ACCESS_TOKEN secret.",
        ])
        return "\n".join(header)

    sections = [
        _format_signal_group(
            "🟢 REAL-TIME MOMENTUM BUYERS", all_gainers,
            "Candle_Gain_%", "15m_Close_Entry",
        ),
        _format_signal_group(
            "🎯 HIGH-CONVICTION BUYERS", shortlisted_gainers,
            "Candle_Gain_%", "15m_Close_Entry",
        ),
        _format_signal_group(
            "🔴 REAL-TIME MOMENTUM SELLERS", all_losers,
            "Candle_Drop_%", "15m_Close_Entry",
        ),
        _format_signal_group(
            "🎯 HIGH-CONVICTION SELLERS", shortlisted_losers,
            "Candle_Drop_%", "15m_Close_Entry",
        ),
    ]
    return "\n".join(header + sections)


def run_current_day_telegram_scan() -> None:
    """Run only today's 09:15-09:30 opening-candle scan and deliver to Telegram."""
    now = datetime.now(MARKET_ZONE)

    # Scheduled run is weekdays at 09:35 IST, but guard against early manual runs.
    if now.weekday() >= 5:
        print("Skipping weekend; no NSE weekday scan scheduled.")
        return

    if (now.hour, now.minute) < (9, 31):
        message = (
            "⏰ MORE12 was triggered before 09:31 IST. "
            "The opening 15-minute candle has not yet been finalized; scan skipped."
        )
        print(message)
        # A manual early run need not send a Telegram notification.
        return

    scan_date = now.strftime("%Y-%m-%d")
    try:
        # Mode 1 = today's opening 15-minute candle only. No historical/rolling mode.
        results = scan_momentum_stocks("1")
        report = build_telegram_report(*results, scan_date=scan_date)
        send_telegram_message(report)
        print("Telegram scanner report delivered successfully.")
    except Exception as exc:
        print(f"Scanner failed: {type(exc).__name__}: {exc}")
        try:
            send_telegram_message(
                "❌ MORE12 scanner failed.\n"
                f"Date: {scan_date}\n"
                f"Error: {type(exc).__name__}: {str(exc)[:800]}\n"
                "Check the GitHub Actions run logs. Secrets are not printed by this message."
            )
        except Exception as telegram_exc:
            print(
                "Could not send Telegram failure notification: "
                f"{type(telegram_exc).__name__}: {telegram_exc}"
            )
        raise


if __name__ == "__main__":
    run_current_day_telegram_scan()
