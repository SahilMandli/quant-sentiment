# app.py
import os, time, shutil
import datetime as dt
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh

# ─── CONFIG ────────────────────────────────────────────────────────
TICKERS       = ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR"]
SUBS          = ["stocks","investing","wallstreetbets"]
UA            = {"User-Agent":"Mozilla/5.0 (ValueTron/1.3)"}
REFRESH_SEC   = 3 * 3600    # refresh Reddit cache every 3h
POST_LIMIT    = 40
POSTS_CSV     = "reddit_posts.csv"
SENTS_CSV     = "reddit_sentiments.csv"
PRICE_TTL     = 900         # 15 min price cache

# ─── PAGE SETUP ────────────────────────────────────────────────────
st.set_page_config("📈 ValueTron", "⚡️", layout="wide")
st.markdown("<h1 style='text-align:center'>⚡️ ValueTron</h1>", unsafe_allow_html=True)
st_autorefresh(interval=30 * 60 * 1000, key="reload")  # full page reload every 30m

# ─── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    tf     = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], index=1)
    tkr    = st.selectbox("Ticker", TICKERS, index=0)
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    st.markdown("### Technical Indicators")
    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD",   True)
    show_rsi  = st.checkbox("RSI",    True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("### Fundamental Ratios")
    show_pe = st.checkbox("P/E",           True)
    show_de = st.checkbox("Debt / Equity", True)
    show_ev = st.checkbox("EV / EBITDA",   True)

# ─── 0 | FETCH & CACHE REDDIT POSTS + SENTIMENT ───────────────────
def fetch_reddit(ticker):
    rows = []
    # 1) Reddit JSON
    for sub in SUBS:
        url = (f"https://www.reddit.com/r/{sub}/search.json"
               f"?q={ticker}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1")
        try:
            r = requests.get(url, headers=UA, timeout=8)
            if r.ok:
                for c in r.json().get("data",{}).get("children",[]):
                    d = c["data"]
                    rows.append({
                        "ticker": ticker,
                        "title":  d.get("title",""),
                        "text":   d.get("selftext",""),
                        "score":  d.get("score",0)
                    })
        except:
            pass
        time.sleep(0.3)
    if rows:
        return rows
    # 2) Pushshift fallback
    base = "https://api.pushshift.io/reddit/search/submission/"
    for sub in SUBS:
        url = f"{base}?q={ticker}&subreddit={sub}&after=7d&size={POST_LIMIT}&sort=desc"
        try:
            for d in requests.get(url,timeout=8).json().get("data",[]):
                rows.append({
                    "ticker": ticker,
                    "title":  d.get("title",""),
                    "text":   d.get("selftext",""),
                    "score":  d.get("score",0)
                })
        except:
            pass
        time.sleep(0.2)
    return rows

def refresh_reddit():
    # only if SENTS_CSV missing or stale
    if os.path.exists(SENTS_CSV):
        age = time.time() - os.path.getmtime(SENTS_CSV)
        if age < REFRESH_SEC:
            return
    all_posts = []
    for tk in TICKERS:
        all_posts += fetch_reddit(tk)
    if not all_posts:
        return
    df = pd.DataFrame(all_posts)
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df.apply(
        lambda r: ((TextBlob(r.title+" "+r.text).sentiment.polarity
                    + sia.polarity_scores(r.title+" "+r.text)["compound"])/2)
                   * min(r.score,100)/100,
        axis=1
    )
    df.to_csv(POSTS_CSV, index=False)
    df.groupby("ticker")["sentiment"].mean().round(4)\
      .reset_index().to_csv(SENTS_CSV, index=False)

refresh_reddit()

# ─── 1 | LOAD & CLASSIFY SENTIMENT ────────────────────────────────
# 1a) from fresh cache
sent_df = pd.DataFrame()
if os.path.exists(SENTS_CSV):
    sent_df = pd.read_csv(SENTS_CSV)
try:
    sent_val = float(sent_df.loc[sent_df.ticker==tkr, "sentiment"].mean())
except:
    sent_val = 0.0

# 1b) classify (±5%)
if   sent_val >  0.05: sent_rating = "A"
elif sent_val < -0.05: sent_rating = "C"
else:                  sent_rating = "B"

# 1c) load raw posts for display
posts_df = pd.DataFrame()
if os.path.exists(POSTS_CSV):
    try:
        posts_df = pd.read_csv(POSTS_CSV)
        posts_df = posts_df.loc[posts_df.ticker==tkr, ["title","score"]].head(20)
    except:
        posts_df = pd.DataFrame(columns=["title","score"])

# ─── 2 | LOAD PRICE + COMPUTE INDICATORS ──────────────────────────
@st.cache_data(ttl=PRICE_TTL)
def load_price(sym, start, end):
    df = yf.download(sym, start=start, end=end+dt.timedelta(days=1),
                     progress=False, auto_adjust=False)
    if df.empty:
        return None
    # flatten multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    # normalize to lower, no spaces
    df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
    # pick one close column
    if "adj_close" in df.columns:
        df["Adj Close"] = df["adj_close"]
    elif "close" in df.columns:
        df["Adj Close"] = df["close"]
    else:
        # fallback numeric
        num = df.select_dtypes("number")
        df["Adj Close"] = num.iloc[:,0]
    # indicators
    df["SMA_20"]   = df["Adj Close"].rolling(20).mean()
    df["MACD"]     = df["Adj Close"].ewm(span=12).mean() - df["Adj Close"].ewm(span=26).mean()
    delta         = df["Adj Close"].diff()
    rs            = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"]     = 100 - 100/(1+rs)
    std           = df["Adj Close"].rolling(20).std()
    df["BB_Upper"]= df["SMA_20"] + 2*std
    df["BB_Lower"]= df["SMA_20"] - 2*std
    return df

today = dt.date.today()
delta = {"1W":7,"1M":30,"6M":180,"1Y":365}[tf]
start = dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=delta)
price = load_price(tkr, start, today)
if price is None:
    st.error(f"❌ No price data for {tkr}")
    st.stop()
last = price.iloc[-1]

# ─── 3 | LOAD FUNDAMENTALS ─────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_fund(sym):
    info = yf.Ticker(sym).info
    return {
        "pe": info.get("trailingPE",np.nan),
        "de": info.get("debtToEquity",np.nan),
        "ev": info.get("enterpriseToEbitda",np.nan)
    }

fund = load_fund(tkr)

# ─── 4 | COMPUTE TECHNICAL + FUND SCORE ───────────────────────────
tech = 0.0
if show_sma  and not np.isnan(last["SMA_20"]):
    tech += 1 if last["Adj Close"] > last["SMA_20"] else -1
if show_macd and not np.isnan(last["MACD"]):
    tech += 1 if last["MACD"] > 0 else -1
if show_rsi  and not np.isnan(last["RSI"]):
    tech += 1 if 40 < last["RSI"] < 70 else -1
if show_bb  and not (np.isnan(last["BB_Upper"]) or np.isnan(last["BB_Lower"])):
    tech +=  0.5 if last["Adj Close"] >  last["BB_Upper"] else 0
    tech += -0.5 if last["Adj Close"] <  last["BB_Lower"] else 0

if show_pe and not np.isnan(fund["pe"]):
    tech += 1.0 if fund["pe"] < 18 else -1.0
if show_de and not np.isnan(fund["de"]):
    tech += 0.5 if fund["de"] < 1 else -0.5
if show_ev and not np.isnan(fund["ev"]):
    tech += 1.0 if fund["ev"] < 12 else -1.0

# ─── 5 | BLEND + VERDICT ───────────────────────────────────────────
blend = tech_w/100 * tech + sent_w/100 * sent_val
if blend >  2: ver, clr = "BUY",  "springgreen"
elif blend < -2: ver, clr = "SELL", "salmon"
else:            ver, clr = "HOLD","khaki"

# ─── 6 | RENDER TABS ───────────────────────────────────────────────
tab_v,tab_ta,tab_f,tab_r = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"]
)

with tab_v:
    st.markdown(f"<h2 style='color:{clr};text-align:center'>{ver}</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score",  f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score",  f"{sent_val:.2f}")
    c4.metric("Blended",     f"{blend:.2f}")
    st.caption(f"{tech_w}% Technical + {sent_w}% Sentiment")

with tab_ta:
    dfp = price.loc[start:today]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp.index, y=dfp["Adj Close"], name="Price"))
    if show_sma:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["SMA_20"], name="SMA-20", line=dict(dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["BB_Upper"], name="Upper BB", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["BB_Lower"], name="Lower BB", line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)
    if show_macd: st.line_chart(dfp["MACD"], height=180)
    if show_rsi:  st.line_chart(dfp["RSI"],  height=180)

with tab_f:
    ratios = pd.DataFrame({
        "Metric": ["P/E","Debt / Equity","EV / EBITDA"],
        "Value":  [fund["pe"],fund["de"],fund["ev"]]
    }).set_index("Metric")
    st.table(ratios)

with tab_r:
    if posts_df.empty:
        st.info("No Reddit posts found.")
    else:
        st.dataframe(posts_df, hide_index=True, use_container_width=True)
