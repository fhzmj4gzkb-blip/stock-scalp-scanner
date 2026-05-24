import time
from datetime import datetime, timedelta, timezone
from feedparser import parse
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AI Quick Scalp Scanner", page_icon="📈", layout="centered")

st.title("📈 AI Quick Scalp Scanner")

DEFAULT_TICKERS = ["TQQQ", "SOXL", "QQQ", "NVDA", "AMD", "TSLA"]

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
    return 100 - (100 / (1 + rs))

def fetch_data(symbol, minutes):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=3)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{minutes}/minute/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )

    data = requests.get(url).json()

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

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

    df["vwap"] = (
        (df["typical_price"] * df["volume"]).cumsum()
        / df["volume"].cumsum()
    )

    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["rsi"] = calc_rsi(df["close"])
    df["avg_volume_20"] = df["volume"].rolling(20).mean()

    return df

def ai_score(price, vwap, ema9, ma50, rsi, volume, avg_volume):
    score = 0
    reasons = []

    if price > vwap:
        score += 25
        reasons.append("Price above VWAP")
    else:
        reasons.append("Price below VWAP")

    if price > ema9:
        score += 20
        reasons.append("Price above 9 EMA")
    else:
        reasons.append("Price below 9 EMA")

    if ema9 > ma50:
        score += 20
        reasons.append("9 EMA above 50 MA")
    else:
        reasons.append("9 EMA below 50 MA")

    if 45 <= rsi <= 70:
        score += 20
        reasons.append("RSI healthy")
    elif rsi > 70:
        reasons.append("RSI overbought")
    else:
        reasons.append("RSI weak")

    if volume > avg_volume:
        score += 15
        reasons.append("Volume above average")
    else:
        reasons.append("Volume below average")

    return score, reasons

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
volume = latest["volume"]
avg_volume = latest["avg_volume_20"]

score, reasons = ai_score(price, vwap, ema9, ma50, rsi, volume, avg_volume)

st.subheader(f"{ticker} Live Scalp Signal")

col1, col2 = st.columns(2)
col1.metric("Current Price", f"${price:.2f}")
col2.metric("AI Bull Score", f"{score}/100")

c1, c2, c3 = st.columns(3)
c1.metric("VWAP", f"${vwap:.2f}")
c2.metric("9 EMA", f"${ema9:.2f}")
c3.metric("50 MA", f"${ma50:.2f}")

v1, v2, v3 = st.columns(3)
v1.metric("RSI", f"{rsi:.1f}")
v2.metric("Volume", f"{volume:,.0f}")
v3.metric("Avg Vol 20", f"{avg_volume:,.0f}")

volume_strength = volume / avg_volume if avg_volume > 0 else 0
st.metric("Volume Strength", f"{volume_strength:.2f}x")

entry = price
stop_loss = min(vwap, ema9)
target1 = entry * 1.005
target2 = entry * 1.01

if score >= 80:
    st.success("BUY / STRONG SCALP SETUP")
elif score >= 60:
    st.warning("WAIT / SETUP FORMING")
else:
    st.error("AVOID / WEAK SETUP")

st.write("### AI Reasoning")
for r in reasons:
    st.write(f"- {r}")

st.write("### Suggested Trade Plan")
st.write(f"Possible Entry: **${entry:.2f}**")
st.write(f"Stop Loss Area: **${stop_loss:.2f}**")
st.write(f"Target 1: **${target1:.2f}**")
st.write(f"Target 2: **${target2:.2f}**")

fig = go.Figure()

fig.add_trace(go.Scatter(x=df["time"], y=df["close"], mode="lines", name="Price"))
fig.add_trace(go.Scatter(x=df["time"], y=df["vwap"], mode="lines", name="VWAP"))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema9"], mode="lines", name="9 EMA"))
fig.add_trace(go.Scatter(x=df["time"], y=df["ma50"], mode="lines", name="50 MA"))

fig.update_layout(height=450)

st.plotly_chart(fig, use_container_width=True)

st.write("### Volume Chart")
vol_fig = go.Figure()
vol_fig.add_trace(go.Bar(x=df["time"], y=df["volume"], name="Volume"))
vol_fig.add_trace(go.Scatter(x=df["time"], y=df["avg_volume_20"], mode="lines", name="Avg Volume 20"))

vol_fig.update_layout(height=300)

st.plotly_chart(vol_fig, use_container_width=True)

st.caption("This is a rule-based AI-style scanner, not financial advice. Always use stop loss.")

st.write("## 📰 Semiconductor Market News")

news_feed = parse(
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SOXX,NVDA,AMD,INTC,TSM&region=US&lang=en-US"
)

for entry in news_feed.entries[:5]:
    st.markdown(f"### [{entry.title}]({entry.link})")


time.sleep(30)
st.rerun()