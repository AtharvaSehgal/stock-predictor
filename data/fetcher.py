# data/fetcher.py
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

try:
    import streamlit as st
    STREAMLIT = True
except:
    STREAMLIT = False

def cache_it(func):
    if STREAMLIT:
        return st.cache_data(ttl=3600)(func)
    return func


# ---- NSE direct API headers ----
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/',
}


def get_nse_session():
    """Create a session with NSE cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get('https://www.nseindia.com', timeout=5)  # 5 sec max
    except:
        pass
    return session


@cache_it
def get_stock_data(symbol: str, period_days: int = 365) -> pd.DataFrame:
    """
    Fetch stock data — tries multiple free sources.
    symbol: e.g. 'RELIANCE.NS' or 'RELIANCE'
    """
    # Clean symbol — remove .NS or .BO suffix for NSE API
    clean = symbol.replace('.NS', '').replace('.BO', '').upper()

    # Method 1 — Stooq (free, no rate limit, works on cloud)
    df = _fetch_stooq(symbol, period_days)
    if df is not None and not df.empty:
        print(f"✅ Stooq: {len(df)} days for {symbol}")
        return df

    # Method 2 — NSE India direct API
    df = _fetch_nse(clean, period_days)
    if df is not None and not df.empty:
        print(f"✅ NSE API: {len(df)} days for {symbol}")
        return df

    # Method 3 — yfinance as last resort with long wait
    df = _fetch_yfinance_safe(symbol, period_days)
    if df is not None and not df.empty:
        print(f"✅ yfinance: {len(df)} days for {symbol}")
        return df

    print(f"❌ All sources failed for {symbol}")
    return None


def _fetch_stooq(symbol: str, period_days: int) -> pd.DataFrame:
    """Fetch from Stooq with hard timeout."""
    try:
        sym = symbol.replace('.NS', '.IN').replace('.BO', '.IN').lower()
        end = datetime.today()
        start = end - timedelta(days=period_days)

        url = (
            f"https://stooq.com/q/d/l/"
            f"?s={sym}"
            f"&d1={start.strftime('%Y%m%d')}"
            f"&d2={end.strftime('%Y%m%d')}"
            f"&i=d"
        )

        # Hard 8 second timeout
        resp = requests.get(url, timeout=8, headers=HEADERS)
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        if df.empty or 'Close' not in df.columns:
            return None

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        return df if len(df) > 10 else None

    except Exception as e:
        print(f"Stooq failed: {e}")
        return None


def _fetch_nse(symbol: str, period_days: int) -> pd.DataFrame:
    """Fetch from NSE India API directly."""
    try:
        session = get_nse_session()
        end = datetime.today()
        start = end - timedelta(days=period_days)

        url = (
            f"https://www.nseindia.com/api/historical/cm/equity"
            f"?symbol={symbol}"
            f"&series=[%22EQ%22]"
            f"&from={start.strftime('%d-%m-%Y')}"
            f"&to={end.strftime('%d-%m-%Y')}"
        )

        resp = session.get(url, timeout=15)
        data = resp.json().get('data', [])

        if not data:
            return None

        rows = []
        for item in data:
            rows.append({
                'Date':   pd.to_datetime(item['CH_TIMESTAMP']),
                'Open':   float(item['CH_OPENING_PRICE']),
                'High':   float(item['CH_TRADE_HIGH_PRICE']),
                'Low':    float(item['CH_TRADE_LOW_PRICE']),
                'Close':  float(item['CH_CLOSING_PRICE']),
                'Volume': float(item['CH_TOT_TRADED_QTY']),
            })

        df = pd.DataFrame(rows)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df

    except Exception as e:
        print(f"NSE API failed: {e}")
        return None


def _fetch_yfinance_safe(symbol: str, period_days: int) -> pd.DataFrame:
    """yfinance as last resort."""
    try:
        import yfinance as yf

        period_str = (
            "2y" if period_days >= 500 else
            "1y" if period_days >= 300 else
            "6mo"
        )

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period_str, timeout=8)

        if df.empty:
            return None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"yfinance fallback failed: {e}")
        return None


@cache_it
def get_live_price(symbol: str) -> dict:
    """Get live price — tries multiple sources."""
    clean = symbol.replace('.NS', '').replace('.BO', '').upper()

    # Try NSE quote API
    try:
        session = get_nse_session()
        url = f"https://www.nseindia.com/api/quote-equity?symbol={clean}"
        resp = session.get(url, timeout=10)
        data = resp.json()

        pd_data = data.get('priceInfo', {})
        info = data.get('metadata', {})

        return {
            'symbol':              symbol,
            'current_price':       pd_data.get('lastPrice', 'N/A'),
            'previous_close':      pd_data.get('previousClose', 'N/A'),
            'day_high':            pd_data.get('intraDayHighLow', {}).get('max', 'N/A'),
            'day_low':             pd_data.get('intraDayHighLow', {}).get('min', 'N/A'),
            'volume':              info.get('totalTradedVolume', 'N/A'),
            'market_cap':          'N/A',
            'pe_ratio':            data.get('metadata', {}).get('pdSymbolPe', 'N/A'),
            'fifty_two_week_high': pd_data.get('weekHighLow', {}).get('max', 'N/A'),
            'fifty_two_week_low':  pd_data.get('weekHighLow', {}).get('min', 'N/A'),
        }

    except Exception as e:
        print(f"NSE live price failed: {e}")

    # Fallback — return N/A values gracefully
    return {
        'symbol': symbol,
        'current_price': 'N/A', 'previous_close': 'N/A',
        'day_high': 'N/A', 'day_low': 'N/A',
        'volume': 'N/A', 'market_cap': 'N/A',
        'pe_ratio': 'N/A',
        'fifty_two_week_high': 'N/A',
        'fifty_two_week_low': 'N/A',
    }


def get_multiple_stocks(symbols: list, period_days: int = 365) -> dict:
    data = {}
    for symbol in symbols:
        df = get_stock_data(symbol, period_days)
        if df is not None:
            data[symbol] = df
    return data