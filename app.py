import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import feedparser

st.set_page_config(page_title="AI Quick Scalp Scanner", page_icon="📈", layout="centered")

st.title("📈 AI Quick Scalp Scanner")
st.caption("VWAP + RSI + EMA + Volume + News + Multi-Timeframe Scalp Dashboard")

DEFAULT_TICKERS = ["TQQQ", "SOXL", "SOXS", "QQQ", "NVDA", "AMD", "TSLA", "SMH", "SOXX"]

with st.sidebar:
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=1)
    custom_ticker = st.text_input("Or type ticker", "")
    if custom_ticker.strip():
        ticker = custom_ticker.strip().upper()

    timeframe = st.selectbox("Chart Timeframe", ["1", "5", "15"], index=1)
    auto_refresh = st.selectbox("Auto Refresh Seconds", [15, 30, 60], index=1)

try:
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
except Exception:
    POLYGON_API_KEY = ""

if not POLYGON_API_KEY:
    st.error("Missing POLYGON_API_KEY in Streamlit secrets.")
    st.stop()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_data(symbol, minutes, days=4):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{minutes}/minute/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )

    try:
        data = requests.get(url, timeout=20).json()
    except Exception:
        return pd.DataFrame()

    if "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])

    df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
        },
        inplace=True,
    )

    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["time_et"] = df["time"].dt.tz_convert("America/New_York")

    df = df[
        (df["time_et"].dt.time >= pd.to_datetime("09:30").time())
        & (df["time_et"].dt.time <= pd.to_datetime("16:00").time())
    ].copy()

    if df.empty:
        return df

    df["date_et"] = df["time_et"].dt.date

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_x_vol"] = df["typical_price"] * df["volume"]
    df["cum_tpv"] = df.groupby("date_et")["tp_x_vol"].cumsum()
    df["cum_vol"] = df.groupby("date_et")["volume"].cumsum()
    df["vwap"] = df["cum_tpv"] / df["cum_vol"]

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["rsi"] = calc_rsi(df["close"])
    df["avg_volume_20"] = df["volume"].rolling(20).mean()

    return df


def ai_score(price, vwap, ema9, ma50, rsi, volume, avg_volume):
    score = 0
    reasons = []

    if pd.isna(ma50) or pd.isna(rsi) or pd.isna(avg_volume):
        return 0, ["Not enough candles yet."]

    if price > vwap:
        score += 25
        reasons.append("✅ Price above VWAP")
    else:
        reasons.append("❌ Price below VWAP")

    if price > ema9:
        score += 20
        reasons.append("✅ Price above 9 EMA")
    else:
        reasons.append("❌ Price below 9 EMA")

    if ema9 > ma50:
        score += 20
        reasons.append("✅ 9 EMA above 50 MA")
    else:
        reasons.append("❌ 9 EMA below 50 MA")

    if 45 <= rsi <= 70:
        score += 20
        reasons.append("✅ RSI healthy for momentum")
    elif rsi > 70:
        reasons.append("⚠️ RSI overbought")
    else:
        reasons.append("❌ RSI weak")

    if volume > avg_volume:
        score += 15
        reasons.append("✅ Volume above 20-candle average")
    else:
        reasons.append("⚠️ Volume below average")

    return score, reasons


def signal_from_score(score):
    if score >= 80:
        return "BUY / STRONG SCALP SETUP", "success"
    if score >= 60:
        return "WAIT / SETUP FORMING", "warning"
    return "AVOID / WEAK SETUP", "error"


def simple_news_sentiment(title):
    bullish_words = [
        "surge", "rally", "beat", "upgrade", "strong", "growth", "demand",
        "record", "bullish", "soars", "jumps", "gain", "higher", "outperform",
        "ai boom", "partnership"
    ]
    bearish_words = [
        "falls", "drop", "downgrade", "weak", "cut", "risk", "ban",
        "restriction", "slump", "miss", "lower", "selloff", "concern",
        "investigation", "delay"
    ]

    text = title.lower()
    bull = sum(1 for w in bullish_words if w in text)
    bear = sum(1 for w in bearish_words if w in text)

    if bull > bear:
        return "Bullish"
    if bear > bull:
        return "Bearish"
    return "Neutral"


def fetch_semiconductor_news():
    feed_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SOXX,SMH,SOXL,NVDA,AMD,INTC,TSM,AVGO,MU&region=US&lang=en-US"
    feed = feedparser.parse(feed_url)
    return feed.entries[:8]


def multi_timeframe_check(symbol):
    checks = {}
    for tf in ["1", "5", "15"]:
        d = fetch_data(symbol, tf, days=4)
        if d.empty or len(d) < 60:
            checks[tf] = "N/A"
            continue

        row = d.iloc[-1]
        score, _ = ai_score(
            row["close"],
            row["vwap"],
            row["ema9"],
            row["ma50"],
            row["rsi"],
            row["volume"],
            row["avg_volume_20"],
        )

        if score >= 80:
            checks[tf] = "Bullish"
        elif score >= 60:
            checks[tf] = "Mixed"
        else:
            checks[tf] = "Weak"

    return checks


def relative_strength():
    symbols = ["SOXL", "SOXX", "SMH", "NVDA", "AMD", "QQQ"]
    rows = []

    for sym in symbols:
        d = fetch_data(sym, "5", days=3)
        if d.empty or len(d) < 2:
            continue

        first = d.iloc[0]["close"]
        last = d.iloc[-1]["close"]
        change = ((last - first) / first) * 100
        rows.append({"Ticker": sym, "Recent % Change": round(change, 2)})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Recent % Change", ascending=False)


df = fetch_data(ticker, timeframe)

if df.empty:
    st.error("No data found. Market may be closed, or your API plan may not support this request.")
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
signal, signal_type = signal_from_score(score)

st.subheader(f"{ticker} Live Scalp Signal")

col1, col2 = st.columns(2)
col1.metric("Current Price", f"${price:.2f}")
col2.metric("AI Bull Score", f"{score}/100")

c1, c2, c3 = st.columns(3)
c1.metric("VWAP", f"${vwap:.2f}")
c2.metric("9 EMA", f"${ema9:.2f}")
c3.metric("50 MA", f"${ma50:.2f}" if pd.notna(ma50) else "N/A")

v1, v2, v3 = st.columns(3)
v1.metric("RSI", f"{rsi:.1f}" if pd.notna(rsi) else "N/A")
v2.metric("Volume", f"{volume:,.0f}")
v3.metric("Avg Vol 20", f"{avg_volume:,.0f}" if pd.notna(avg_volume) else "N/A")

volume_strength = volume / avg_volume if pd.notna(avg_volume) and avg_volume > 0 else 0
st.metric("Volume Strength", f"{volume_strength:.2f}x")

if signal_type == "success":
    st.success(signal)
elif signal_type == "warning":
    st.warning(signal)
else:
    st.error(signal)

st.write("### AI Reasoning")
for r in reasons:
    st.write(r)

entry = price
stop_loss = min(vwap, ema9)
target1 = entry * 1.005
target2 = entry * 1.01

st.write("### Suggested Trade Plan")
st.write(f"Possible Entry: **${entry:.2f}**")
st.write(f"Stop Loss Area: **${stop_loss:.2f}**")
st.write(f"Target 1: **${target1:.2f}**")
st.write(f"Target 2: **${target2:.2f}**")

st.write("## ⏱️ Multi-Timeframe Confirmation")

mtf = multi_timeframe_check(ticker)

m1, m5, m15 = st.columns(3)
m1.metric("1 Min", mtf.get("1", "N/A"))
m5.metric("5 Min", mtf.get("5", "N/A"))
m15.metric("15 Min", mtf.get("15", "N/A"))

if mtf.get("1") == "Bullish" and mtf.get("5") == "Bullish":
    st.success("Short-term confirmation is bullish.")
elif mtf.get("5") == "Bullish" and mtf.get("15") == "Bullish":
    st.success("Stronger trend confirmation is bullish.")
elif "Weak" in mtf.values():
    st.warning("Not all timeframes agree. Be careful with size.")
else:
    st.info("Mixed timeframe signal. Wait for cleaner confirmation.")

chart_df = df.tail(150)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df["time_et"],
    open=chart_df["open"],
    high=chart_df["high"],
    low=chart_df["low"],
    close=chart_df["close"],
    name="Candles",
))
fig.add_trace(go.Scatter(x=chart_df["time_et"], y=chart_df["vwap"], mode="lines", name="VWAP"))
fig.add_trace(go.Scatter(x=chart_df["time_et"], y=chart_df["ema9"], mode="lines", name="9 EMA"))
fig.add_trace(go.Scatter(x=chart_df["time_et"], y=chart_df["ma50"], mode="lines", name="50 MA"))
fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

st.write("### Volume Chart")
vol_fig = go.Figure()
vol_fig.add_trace(go.Bar(x=chart_df["time_et"], y=chart_df["volume"], name="Volume"))
vol_fig.add_trace(go.Scatter(x=chart_df["time_et"], y=chart_df["avg_volume_20"], mode="lines", name="Avg Volume 20"))
vol_fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(vol_fig, use_container_width=True)

st.write("## 💪 Semiconductor Relative Strength")

rs_df = relative_strength()
if not rs_df.empty:
    st.dataframe(rs_df, use_container_width=True)
else:
    st.info("Relative strength data not available right now.")

st.write("## 📰 Recent Semiconductor Market News")

news = fetch_semiconductor_news()

if news:
    bull_count = 0
    bear_count = 0
    neutral_count = 0

    for item in news:
        sentiment = simple_news_sentiment(item.title)

        if sentiment == "Bullish":
            bull_count += 1
            emoji = "🟢"
        elif sentiment == "Bearish":
            bear_count += 1
            emoji = "🔴"
        else:
            neutral_count += 1
            emoji = "⚪"

        st.markdown(f"**{emoji} {sentiment}: [{item.title}]({item.link})**")

    total = bull_count + bear_count + neutral_count
    if total > 0:
        news_score = round((bull_count / total) * 100)
        st.metric("Simple News Bullish Score", f"{news_score}%")

        if news_score >= 60:
            st.success("News tone looks mostly supportive.")
        elif news_score <= 30:
            st.error("News tone looks cautious/bearish.")
        else:
            st.warning("News tone is mixed.")
else:
    st.info("No news found right now.")

st.caption(
    "This is a rule-based AI-style scanner, not financial advice. "
    "Use stop losses. News sentiment is simple keyword-based, not a professional prediction model."
)

st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

time.sleep(auto_refresh)
st.rerun()