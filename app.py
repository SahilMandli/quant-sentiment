import streamlit as st
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, base64, os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")

# Optional background style
if os.path.exists("tron.png"):
    with open("tron.png","rb") as f:
        bg = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    body,.stApp{{background:
        linear-gradient(rgba(0,0,0,.9),rgba(0,0,0,.9)),
        url("data:image/png;base64,{bg}") center/cover fixed;
        color:#fff;font-family:Arial}}
    h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
    .stSidebar{{background:rgba(0,0,30,.9);border-right:2px solid #0ff}}
    </style>""",unsafe_allow_html=True)

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    tf  = st.selectbox("Timeframe",["1W","1M","6M","YTD","1Y"],index=1)
    tkr = st.text_input("Ticker","NVDA").strip().upper()
    tech_w = st.slider("Technical Weight %",0,100,60)
    sent_w = 100-tech_w
    show_sma  = st.checkbox("SMA‑20",True)
    show_macd = st.checkbox("MACD",True)
    show_rsi  = st.checkbox("RSI",True)
    show_bb   = st.checkbox("Bollinger Bands",True)
    run = st.button("🚀 Analyze")

if not run or not tkr:
    st.stop()
today = dt.date.today()
delta = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf,365)
start = dt.date(today.year,1,1) if tf=="YTD" else today - dt.timedelta(days=delta)

@st.cache_data(ttl=900)
def load_price(tkr,start,end):
    df = yf.download(tkr,start=start,end=end+dt.timedelta(days=1),progress=False)
    if df.empty: return None
    df["Adj Close"] = df.get("Adj Close",df["Close"])
    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["MACD"] = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    std = df["Adj Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2*std
    df["BB_Lower"] = df["SMA_20"] - 2*std
    return df

price = load_price(tkr, start, today)
if price is None:
    st.error("No price data."); st.stop()
price = price.dropna(subset=["Adj Close", "SMA_20", "MACD", "RSI"])
last = price.iloc[-1]
@st.cache_data(ttl=300)
def reddit_sentiment(tkr):
    try:
        import praw
        r = praw.Reddit(
            client_id=st.secrets["reddit"]["client_id"],
            client_secret=st.secrets["reddit"]["client_secret"],
            user_agent="QuantDash")
        posts = list(r.subreddit("stocks+investing+wallstreetbets").search(tkr, limit=50, sort="new"))
        raw = [{"title": p.title, "text": p.selftext or "", "score": p.score} for p in posts]
    except Exception:
        url = f"https://api.pushshift.io/reddit/search/submission/?q={tkr}&subreddit=stocks,investing,wallstreetbets&sort=desc&size=50"
        data = requests.get(url, timeout=10).json().get("data", [])
        raw = [{"title": d.get("title", ""), "text": d.get("selftext", ""), "score": d.get("score", 0)} for d in data]

    if not raw:
        return 0.0, "B", pd.DataFrame()

    sia = SentimentIntensityAnalyzer()
    def score(p):
        base = (TextBlob(p["title"]+" "+p["text"]).sentiment.polarity + sia.polarity_scores(p["title"]+" "+p["text"])["compound"]) / 2
        return base * min(p["score"], 100) / 100

    avg = sum(score(p) for p in raw) / len(raw)
    rating = "A" if avg > 0.2 else "C" if avg < -0.2 else "B"
    return avg, rating, pd.DataFrame(raw)

sent_val, sent_rating, df_posts = reddit_sentiment(tkr)
tech = 0
tech += 1 if last["Adj Close"] > last["SMA_20"] else -1
tech += 1 if last["MACD"] > 0 else -1
tech += 1 if 40 < last["RSI"] < 70 else -1
if last["Adj Close"] > last["BB_Upper"]: tech += 0.5
elif last["Adj Close"] < last["BB_Lower"]: tech -= 0.5

blend = tech_w/100 * tech + sent_w/100 * sent_val
def verdict(b): return ("BUY","springgreen") if b>2 else ("SELL","salmon") if b<-2 else ("HOLD","khaki")
ver,color = verdict(blend)

tab_v, tab_ta, tab_r = st.tabs(["🏁 Verdict","📈 Technical","🗣️ Reddit Pulse"])
with tab_v:
    st.header("Overall Verdict")
    st.markdown(f"<h1 style='color:{color};text-align:center'>{ver}</h1>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score", f"{tech:.2f}")
    c2.metric("Sentiment Rating", sent_rating)
    c3.metric("Sentiment Score", f"{sent_val:.2f}")
    c4.metric("Blended Score", f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech + {sent_w}% Sentiment")
with tab_ta:
    df = price.loc[start:today]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Price", line=dict(color="#0ff")))
    if show_sma: fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA‑20", line=dict(dash="dash", color="#ff0")))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper BB", line=dict(color="#0f0", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower BB", line=dict(color="#0f0", dash="dot")))
    fig.update_layout(template="plotly_dark", height=350, title="Price / SMA / Bollinger")
    st.plotly_chart(fig, use_container_width=True)
    if show_macd: st.line_chart(df["MACD"], height=200)
    if show_rsi: st.line_chart(df["RSI"], height=200)
    st.subheader("Candlestick")
    candle = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#0ff", decreasing_line_color="#f44")])
    candle.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False)
    st.plotly_chart(candle, use_container_width=True)

with tab_r:
    st.header("Latest Reddit Mentions")
    if df_posts.empty:
        st.info("No recent posts.")
    else:
        st.dataframe(df_posts[["title", "score"]].head(20), use_container_width=True)
