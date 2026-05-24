import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Quick Scalp Scanner",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Quick Scalp Scanner")

DEFAULT_TICKERS = ["TQQQ", "SOXL", "QQQ", "NVDA", "AMD"]

with st.sidebar:
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS)
    timeframe = st.selectbox("Timeframe", ["1", "5"], index=1)

try:
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
except:
    POLYGON_API_KEY = ""

if not POLYGON_API_KEY:
    st.error("Missing POLYGON_API_KEY in Streamlit secrets.")
    st.stop()

def calc_rsi(series, period=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def fetch_data(symbol, minutes):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=3)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{minutes}/minute/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])

    df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "timestamp"
    }, inplace=True)

    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")

    df["typical_price"] = (
        df["high"] + df["low"] + df["close"]
    ) / 3

    df["vwap"] = (
        (df["typical_price"] * df["volume"]).cumsum()
        / df["volume"].cumsum()
    )

    df["ema9"] = df["close"].ewm(span=9).mean()

    df["ma50"] = df["close"].rolling(50).mean()

    df["rsi"] = calc_rsi(df["close"])

    return df

df = fetch_data(ticker, timeframe)

if df.empty:
    st.error("No data found.")
    st.stop()

latest = df.iloc[-1]

price = latest["close"]
vwap = latest["vwap"]
ema9 = latest["ema9"]
ma50 = latest["ma50"]
rsi = latest["rsi"]

st.subheader(f"{ticker} Signal")

col1, col2 = st.columns(2)

col1.metric("Current Price", f"${price:.2f}")
col2.metric("VWAP", f"${vwap:.2f}")

c1, c2, c3 = st.columns(3)

c1.metric("RSI", f"{rsi:.1f}")
c2.metric("9 EMA", f"${ema9:.2f}")
c3.metric("50 MA", f"${ma50:.2f}")

if (
    price > vwap
    and price > ema9
    and ema9 > ma50
    and 45 <= rsi <= 70
):
    st.success("BUY / SCALP SETUP")

elif price < vwap:
    st.warning("WAIT")

else:
    st.error("AVOID")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["close"],
    mode="lines",
    name="Price"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["vwap"],
    mode="lines",
    name="VWAP"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["ema9"],
    mode="lines",
    name="9 EMA"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["ma50"],
    mode="lines",
    name="50 MA"
))

st.plotly_chart(fig, use_container_width=True)

time.sleep(30)
st.rerun()