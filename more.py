from datetime import datetime, timedelta
import os
import sys
import time
from typing import List, Tuple

import pandas as pd
import requests
import yfinance as yf


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
    mode: str = "1", target_date_str: str | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scans full NSE universe for either Opening 15m candle or Latest Intraday rolling 15m candle."""
    if mode == "1" or mode == "3":
        current_date = datetime.now().strftime("%Y-%m-%d")
        download_kwargs = {"period": "1d" if mode == "3" else "2d"}
    else:
        current_date = target_date_str
        target_dt = datetime.strptime(current_date, "%Y-%m-%d")
        end_dt = target_dt + timedelta(days=1)
        download_kwargs = {
            "start": target_dt.strftime("%Y-%m-%d"),
            "end": end_dt.strftime("%Y-%m-%d")
        }

    print(f"📅 Scan Date: {current_date} | Mode: {'Latest Intraday Rolling' if mode == '3' else 'Opening Candle'}")

    tickers = get_dynamic_nse_universe()
    print(f"⏳ Downloading intraday data for {len(tickers)} equities in batches...")

    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")

    dataframes = []
    batch_size = 50
    
    try:
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            df_batch = yf.download(
                batch, interval="15m", group_by="ticker", progress=False, threads=True, **download_kwargs
            )
            if not df_batch.empty:
                dataframes.append(df_batch)
            time.sleep(1)  # Rate-limit throttle safeguard
            
        if dataframes:
            data = pd.concat(dataframes, axis=1)
        else:
            data = pd.DataFrame()
            
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

    all_losers = []
    shortlisted_losers = []
    all_gainers = []
    shortlisted_gainers = []
    
    market_open_time = datetime.strptime("09:15:00", "%H:%M:%S").time()

    for ticker in tickers:
        try:
            if len(tickers) > 1:
                if ticker not in data.columns.levels[0]:
                    continue
                ticker_df = data[ticker].dropna(how="all")
            else:
                ticker_df = data.dropna(how="all")

            if ticker_df.empty:
                continue

            if isinstance(ticker_df.columns, pd.MultiIndex):
                ticker_df.columns = ticker_df.columns.get_level_values(0)

            ticker_df.index = pd.to_datetime(ticker_df.index)
            
            # Select target candle based on mode choice
            if mode == "3":
                # Latest completed intraday 15m candle for today
                today_mask = (ticker_df.index.date == pd.to_datetime(current_date).date())
                day_candles = ticker_df[today_mask]
                if day_candles.empty:
                    continue
                target_candle_df = day_candles.iloc[[-1]]  # Get the last completed row
                candle_label = "Latest_15m_Close"
            else:
                # Opening 15-minute candle (09:15:00)
                today_mask = (ticker_df.index.date == pd.to_datetime(current_date).date()) & (ticker_df.index.time == market_open_time)
                target_candle_df = ticker_df[today_mask]
                candle_label = "15m_Close_Entry"

            if not target_candle_df.empty:
                latest_candle = target_candle_df.iloc[0]

                open_p = float(latest_candle["Open"])
                close_p = float(latest_candle["Close"])
                high_p = float(latest_candle["High"])
                low_p = float(latest_candle["Low"])
                vol = int(latest_candle["Volume"])

                if open_p == 0:
                    continue

                pct_change = ((close_p - open_p) / open_p) * 100
                turnover_cr = (vol * close_p) / 10_000_000

                candle_range = high_p - low_p
                close_location = (
                    (close_p - low_p) / candle_range if candle_range > 0 else 1.0
                )

                # --- SHORT (SELLING) LOGIC ---
                if pct_change < 0:
                    row_data_short = {
                        "Ticker": ticker.replace(".NS", ""),
                        "Date": current_date,
                        "15m_Open": round(open_p, 2),
                        candle_label: round(close_p, 2),
                        "Candle_Drop_%": round(pct_change, 2),
                        "Turnover_Cr": round(turnover_cr, 2),
                        "1%_Profit_Target": round(close_p * 0.99, 2),
                        "Hard_Stop_Loss": round(high_p, 2),
                    }
                    # Tier 1: General Losers
                    if pct_change < -0.5 and turnover_cr >= 4.0 and close_location <= 0.25:
                        all_losers.append(row_data_short)

                    # Tier 2: Shortlisted Most Falling
                    if pct_change < -1.5 and turnover_cr >= 15.0 and close_location <= 0.15:
                        shortlisted_losers.append(row_data_short)

                # --- LONG (BUYING) LOGIC ---
                elif pct_change > 0:
                    row_data_long = {
                        "Ticker": ticker.replace(".NS", ""),
                        "Date": current_date,
                        "15m_Open": round(open_p, 2),
                        candle_label: round(close_p, 2),
                        "Candle_Gain_%": round(pct_change, 2),
                        "Turnover_Cr": round(turnover_cr, 2),
                        "1%_Profit_Target": round(close_p * 1.01, 2),
                        "Hard_Stop_Loss": round(low_p, 2),
                    }
                    # Tier 1: General Gainers
                    if pct_change > 0.5 and turnover_cr >= 4.0 and close_location >= 0.75:
                        all_gainers.append(row_data_long)

                    # Tier 2: Shortlisted Most Rising
                    if pct_change > 1.5 and turnover_cr >= 15.0 and close_location >= 0.85:
                        shortlisted_gainers.append(row_data_long)

        except Exception:
            continue

    df_all_losers = pd.DataFrame(all_losers)
    if not df_all_losers.empty:
        df_all_losers = df_all_losers.sort_values(by="Candle_Drop_%", ascending=True).reset_index(drop=True)

    df_short_losers = pd.DataFrame(shortlisted_losers)
    if not df_short_losers.empty:
        df_short_losers = df_short_losers.sort_values(by="Candle_Drop_%", ascending=True).reset_index(drop=True)

    df_all_gainers = pd.DataFrame(all_gainers)
    if not df_all_gainers.empty:
        df_all_gainers = df_all_gainers.sort_values(by="Candle_Gain_%", ascending=False).reset_index(drop=True)

    df_short_gainers = pd.DataFrame(shortlisted_gainers)
    if not df_short_gainers.empty:
        df_short_gainers = df_short_gainers.sort_values(by="Candle_Gain_%", ascending=False).reset_index(drop=True)

    return df_all_losers, df_short_losers, df_all_gainers, df_short_gainers


if __name__ == "__main__":
    print("Select Market Analysis Mode:")
    print("1. Opening 15m Candle (09:15 - 09:30) - Live (Today)")
    print("2. Opening 15m Candle (09:15 - 09:30) - Historical (Past Date)")
    print("3. Latest Completed 15m Candle - Live Intraday (Rolling Momentum)")
    choice = input("Enter your choice (1, 2 or 3): ").strip()

    target_date = None
    match choice:
        case "1" | "3":
            pass
        case "2":
            target_date = input("Enter the target date in format YYYY-MM-DD (e.g., 2026-08-15): ").strip()
            try:
                datetime.strptime(target_date, "%Y-%m-%d")
            except ValueError:
                print("❌ Invalid date format provided. Please use YYYY-MM-DD.")
                sys.exit(1)
        case _:
            print("⚠️ Invalid selection. Defaulting to Mode 1 (Live Opening Data).")
            choice = "1"

    all_losers, shortlisted_losers, all_gainers, shortlisted_gainers = scan_momentum_stocks(choice, target_date)

    # Output Gainers (Long Positions)
    print("\n" + "=" * 80)
    print("🟢 REAL-TIME MOMENTUM BUYERS (GAINERS):")
    if not all_gainers.empty:
        print(all_gainers.to_string(index=False))
    else:
        print("No stocks matched the general breakout criteria.")

    print("\n" + "-" * 80)
    print("🎯 HIGH-CONVICTION SHORTLISTED (MOST RISING) BUYERS:")
    if not shortlisted_gainers.empty:
        print(shortlisted_gainers.to_string(index=False))
    else:
        print("No stocks matched the high-conviction breakout criteria.")

    # Output Losers (Short Positions)
    print("\n" + "=" * 80)
    print("🔴 REAL-TIME MOMENTUM SELLERS (LOSERS):")
    if not all_losers.empty:
        print(all_losers.to_string(index=False))
    else:
        print("No stocks matched the general breakdown criteria.")

    print("\n" + "-" * 80)
    print("🎯 HIGH-CONVICTION SHORTLISTED (MOST FALLING) SELLERS:")
    if not shortlisted_losers.empty:
        print(shortlisted_losers.to_string(index=False))
    else:
        print("No stocks matched the high-conviction breakdown criteria.")
