import streamlit as st
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, base64, os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────── PAGE CONFIG & OPTIONAL BACKGROUND ───────────────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
if os.path.exists("tron.png"):
    bg64 = base64.b64encode(open("tron.png","rb").read()).decode()
    st.markdown(f"""
    <style>
      body,.stApp{{background:
        linear-gradient(rgba(0,0,0,.9),rgba(0,0,0,.9)),
        url("data:image/png;base64,{bg64}") center/cover fixed;
        color:#fff;font-family:Arial}}
      h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
      .stSidebar{{background:rgba(0,0,30,.93);border-right:2px solid #0ff}}
    </style>""", unsafe_allow_html=True)

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)

# ─────────── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    tf  = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], index=1)
    tkr = st.text_input("Ticker", "NVDA").strip().upper()
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    # technical factors
    show_sma  = st.checkbox("SMA‑20", True)
    show_macd = st.checkbox("MACD", True)
    show_rsi  = st.checkbox("RSI", True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("---")
    # fundamental factors
    show_pe = st.checkbox("P/E ratio", True)
    show_de = st.checkbox("Debt / Equity", True)
    show_ev = st.checkbox("EV / EBITDA", True)

    run = st.button("🚀 Analyze")

if not run or not tkr:
    st.stop()

# ─────────── DATE RANGE ──────────────────────────────────────────────
today = dt.date.today()
if tf == "YTD":
    start = dt.date(today.year, 1, 1)
else:
    days = {"1W":7, "1M":30, "6M":180, "1Y":365}[tf]
    start = today - dt.timedelta(days=days)

# ─────────── PRICE & INDICATORS ──────────────────────────────────────
@st.cache_data(ttl=900)
def load_price(tkr, start, end):
    df = yf.download(tkr, start=start, end=end+dt.timedelta(days=1), progress=False)
    if df.empty:
        return None
    # ensure Adj Close
    df["Adj Close"] = df.get("Adj Close", df["Close"])
    # indicators
    df["SMA_20"]    = df["Adj Close"].rolling(20).mean()
    df["MACD"]      = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs    = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"]       = 100 - 100/(1+rs)
    std             = df["Adj Close"].rolling(20).std()
    df["BB_Upper"]  = df["SMA_20"] + 2*std
    df["BB_Lower"]  = df["SMA_20"] - 2*std
    return df

price = load_price(tkr, start, today)
if price is None:
    st.error("No price data."); st.stop()

# ─── drop rows that miss only Adj Close (always exists after fill) ───
price = price.dropna(subset=["Adj Close"])
if price.empty:
    st.error("No usable price rows for this period."); st.stop()
last = price.iloc[-1]

# ─────────── FUNDAMENTALS ────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fundamentals(tkr):
    info = yf.Ticker(tkr).fast_info or {}
    return {
        "pe":      info.get("trailingPe",   np.nan),
        "de":      info.get("debtToEquity", np.nan),
        "ev":      info.get("evToEbitda",   np.nan)
    }
fund = fundamentals(tkr)

# ─────────── REDDIT SENTIMENT ────────────────────────────────────────
@st.cache_data(ttl=300)
def reddit_sentiment(tkr):
    # try PRAW first
    try:
        import praw
        rd = praw.Reddit(
            client_id=st.secrets["reddit"]["client_id"],
            client_secret=st.secrets["reddit"]["client_secret"],
            user_agent="QuantDash"
        )
        posts = rd.subreddit("stocks+investing+wallstreetbets").search(tkr, limit=50, sort="new")
        raw   = [{"title":p.title,"text":p.selftext or "","score":p.score} for p in posts]
    except Exception:
        url = (f"https://api.pushshift.io/reddit/search/submission/?q={tkr}"
               "&subreddit=stocks,investing,wallstreetbets&sort=desc&size=50")
        raw = [{"title":d.get("title",""), "text":d.get("selftext",""), "score":d.get("score",0)}
               for d in requests.get(url, timeout=10).json().get("data", [])]

    if not raw:
        return 0.0, "B", pd.DataFrame()

    sia = SentimentIntensityAnalyzer()
    def hybrid(p):
        txt = p["title"] + " " + p["text"]
        base = (TextBlob(txt).sentiment.polarity + sia.polarity_scores(txt)["compound"]) / 2
        return base * min(p["score"],100)/100

    avg = sum(hybrid(p) for p in raw) / len(raw)
    rating = "A" if avg>0.2 else "C" if avg<-0.2 else "B"
    return avg, rating, pd.DataFrame(raw)

sent_val, sent_rating, df_posts = reddit_sentiment(tkr)

# ─────────── TECH + FUND SCORE ───────────────────────────────────────
tech = 0.0
if show_sma and "SMA_20" in last:
    tech += 1 if last["Adj Close"] > last["SMA_20"] else -1
if show_macd and "MACD" in last:
    tech += 1 if last["MACD"] > 0 else -1
if show_rsi and "RSI" in last:
    tech += 1 if 40 < last["RSI"] < 70 else -1
if show_bb and {"BB_Upper","BB_Lower"}.issubset(last.index):
    if last["Adj Close"] > last["BB_Upper"]:
        tech += 0.5
    elif last["Adj Close"] < last["BB_Lower"]:
        tech -= 0.5

if show_pe and not np.isnan(fund["pe"]):
    tech += 1 if fund["pe"] < 18 else -1
if show_de and not np.isnan(fund["de"]):
    tech += 0.5 if fund["de"] < 1 else -0.5
if show_ev and not np.isnan(fund["ev"]):
    tech += 1 if fund["ev"] < 12 else -1

# ─────────── BLEND & VERDICT ─────────────────────────────────────────
blend = tech_w/100 * tech + sent_w/100 * sent_val
ver, color = ("BUY","springgreen") if blend > 2 else ("SELL","salmon") if blend < -2 else ("HOLD","khaki")

# ─────────── DISPLAY TABS ────────────────────────────────────────────
tab_v, tab_ta, tab_f, tab_r = st.tabs(["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])

with tab_v:
    st.header("Overall Verdict")
    st.markdown(f"<h1 style='color:{color};text-align:center'>{ver}</h1>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score", f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score",  f"{sent_val:.2f}")
    c4.metric("Blended",     f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech + {sent_w}% Sentiment")

with tab_ta:
    df = price.loc[start:today]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Price", line=dict(color="#0ff")))
    if show_sma and "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA‑20", line=dict(dash="dash", color="#ff0")))
    if show_bb and {"BB_Upper","BB_Lower"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper BB", line=dict(dash="dot", color="#0f0")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower BB", line=dict(dash="dot", color="#0f0")))
    fig.update_layout(template="plotly_dark", height=350, title="Price / SMA / Bollinger")
    st.plotly_chart(fig, use_container_width=True)

    if show_macd and "MACD" in df.columns:
        st.line_chart(df["MACD"], height=200)
    if show_rsi and "RSI" in df.columns:
        st.line_chart(df["RSI"], height=200)

    st.subheader("Candlestick")
    candle = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#0ff", decreasing_line_color="#f44"
    )])
    candle.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False)
    st.plotly_chart(candle, use_container_width=True)

with tab_f:
    st.header("Key Ratios")
    st.table(pd.DataFrame({
        "Metric":["P/E","Debt / Equity","EV / EBITDA"],
        "Value":[fund["pe"], fund["de"], fund["ev"]]
    }).set_index("Metric"))

with tab_r:
    st.header("Latest Reddit Mentions")
    if df_posts.empty:
        st.info("No recent posts.")
    else:
        st.dataframe(df_posts[["title","score"]].head(20), use_container_width=True)
