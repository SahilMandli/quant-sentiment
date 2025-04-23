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

# ─── Constants & file paths ──────────────────────────────────────────
TICKERS        = ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR"]
SUBS           = ["stocks","investing","wallstreetbets"]
UA             = {"User-Agent":"Mozilla/5.0 (QuantDash/0.3)"}
POST_LIMIT     = 40
CACHE_INTERVAL = 3 * 3600   # refresh Reddit every 3h
POSTS_CSV_OLD  = "/mnt/data/reddit_posts.csv"
SENTS_CSV_OLD  = "/mnt/data/reddit_sentiments.csv"
POSTS_CSV      = "reddit_posts.csv"
SENTS_CSV      = "reddit_sentiments.csv"
PRICE_TTL      = 900        # 15min

# ─── Copy historic CSVs if they exist under /mnt/data ────────────────
if not os.path.exists(POSTS_CSV) and os.path.exists(POSTS_CSV_OLD):
    shutil.copy2(POSTS_CSV_OLD, POSTS_CSV)
if not os.path.exists(SENTS_CSV) and os.path.exists(SENTS_CSV_OLD):
    shutil.copy2(SENTS_CSV_OLD, SENTS_CSV)

# ─── Page config ─────────────────────────────────────────────────────
st.set_page_config("⚡️ Quant Sentiment", "📈", layout="wide")
st.markdown("<h1 style='text-align:center'>⚡️ Quant Sentiment Dashboard</h1>",
            unsafe_allow_html=True)
st_autorefresh(interval=30*60*1000, key="full_refresh")  # full reload every 30m

# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    tf       = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], index=1)
    tkr      = st.selectbox("Ticker", TICKERS, index=0)
    tech_w   = st.slider("Technical Weight %", 0, 100, 60)
    sent_w   = 100 - tech_w

    st.markdown("#### Technical indicators")
    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD",   True)
    show_rsi  = st.checkbox("RSI",    True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("---")
    st.markdown("#### Fundamental ratios")
    show_pe = st.checkbox("P/E ratio",       True)
    show_de = st.checkbox("Debt / Equity",   True)
    show_ev = st.checkbox("EV / EBITDA",     True)

# ─── 0 | Fetch & cache Reddit posts + sentiment ─────────────────────
def fetch_reddit(symbol):
    rows=[]
    # Official Reddit JSON
    for sub in SUBS:
        url = (f"https://www.reddit.com/r/{sub}/search.json"
               f"?q={symbol}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1")
        try:
            r = requests.get(url, headers=UA, timeout=8)
            if r.ok:
                for c in r.json().get("data",{}).get("children",[]):
                    d=c["data"]
                    rows.append({
                        "ticker":symbol,
                        "title":d.get("title",""),
                        "text":d.get("selftext",""),
                        "score":d.get("score",0)
                    })
        except: pass
        time.sleep(0.3)
    if rows: return rows
    # Pushshift fallback
    for sub in SUBS:
        url = (f"https://api.pushshift.io/reddit/search/submission/"
               f"?q={symbol}&subreddit={sub}&after=7d&size={POST_LIMIT}&sort=desc")
        try:
            for d in requests.get(url,timeout=8).json().get("data",[]):
                rows.append({
                    "ticker":symbol,
                    "title":d.get("title",""),
                    "text":d.get("selftext",""),
                    "score":d.get("score",0)
                })
        except: pass
        time.sleep(0.2)
    return rows

def refresh_cache():
    if os.path.exists(SENTS_CSV):
        age = time.time() - os.path.getmtime(SENTS_CSV)
        if age < CACHE_INTERVAL:
            return
    allr=[]
    for sym in TICKERS:
        allr += fetch_reddit(sym)
    if not allr:
        return
    df = pd.DataFrame(allr)
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df.apply(
        lambda r: ((TextBlob((r.title+" "+r.text)).sentiment.polarity
                    + sia.polarity_scores(r.title+" "+r.text)["compound"]) / 2)
                  * min(r.score,100)/100,
        axis=1)
    df.to_csv(POSTS_CSV,index=False)
    df.groupby("ticker")["sentiment"].mean().round(4)\
      .reset_index().to_csv(SENTS_CSV,index=False)

refresh_cache()

# ─── 1 | Load sentiment & raw posts from CSV ────────────────────────
try:
    sent_df  = pd.read_csv(SENTS_CSV).set_index("ticker")
    sent_val = float(sent_df.at[tkr,"sentiment"])
except:
    sent_val = 0.0
sent_rating = "A" if sent_val>0.2 else "C" if sent_val< -0.2 else "B"

try:
    df_posts = (pd.read_csv(POSTS_CSV)
                  .query("ticker==@tkr")[["title","score"]]
                  .head(20))
except:
    df_posts = pd.DataFrame(columns=["title","score"])

# ─── 2 | Load price & indicators ────────────────────────────────────
@st.cache_data(ttl=PRICE_TTL)
def load_price(sym,start,end):
    raw = yf.download(sym,start=start,end=end+dt.timedelta(days=1),
                      progress=False,auto_adjust=False)
    if raw.empty: return None

    # **** FLATTEN MULTIINDEX COLUMNS RIGHT AWAY ****
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(1)

    df = raw.copy()
    # ensure a single "Adj Close" series
    if "Adj Close" not in df.columns:
        for c in df.columns:
            if "adj close" in c.lower():
                df["Adj Close"] = df[c]
                break

    # technical indicators
    df["SMA_20"]     = df["Adj Close"].rolling(20).mean()
    df["MACD"]       = df["Adj Close"].ewm(span=12).mean() \
                       - df["Adj Close"].ewm(span=26).mean()
    delta = df["Adj Close"].diff()
    rs    = (delta.clip(lower=0).rolling(14).mean()
             / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan))
    df["RSI"]       = 100 - 100/(1+rs)
    std             = df["Adj Close"].rolling(20).std()
    df["BB_Upper"]  = df["SMA_20"] + 2*std
    df["BB_Lower"]  = df["SMA_20"] - 2*std
    return df

today = dt.date.today()
days  = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf,365)
start = dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=days)
price = load_price(tkr,start,today)
if price is None:
    st.error(f"❌ No price data for {tkr}")
    st.stop()
last = price.iloc[-1]

# ─── 3 | Load fundamentals ──────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_fund(sym):
    info = yf.Ticker(sym).info
    return {
        "pe": info.get("trailingPE",      np.nan),
        "de": info.get("debtToEquity",    np.nan),
        "ev": info.get("enterpriseToEbitda", np.nan)
    }

fund = load_fund(tkr)

# ─── 4 | Compute tech + fund score ──────────────────────────────────
tech = 0.0
if show_sma  and not pd.isna(last.SMA_20):
    tech += 1  if last["Adj Close"]> last["SMA_20"] else -1
if show_macd and not pd.isna(last.MACD):
    tech += 1  if last["MACD"]>  0 else -1
if show_rsi  and not pd.isna(last.RSI):
    tech += 1  if 40< last["RSI"]<70 else -1
if show_bb and not (pd.isna(last.BB_Upper) or pd.isna(last.BB_Lower)):
    tech += 0.5 if last["Adj Close"]> last["BB_Upper"] else 0
    tech -= 0.5 if last["Adj Close"]< last["BB_Lower"] else 0

if show_pe and not pd.isna(fund["pe"]):
    tech += 1.0  if fund["pe"]<18 else -1.0
if show_de and not pd.isna(fund["de"]):
    tech += 0.5  if fund["de"]<1 else -0.5
if show_ev and not pd.isna(fund["ev"]):
    tech += 1.0  if fund["ev"]<12 else -1.0

# ─── 5 | Blend + verdict ────────────────────────────────────────────
blend = tech_w/100*tech + sent_w/100*sent_val
if blend>  2: ver,color = "BUY",  "springgreen"
elif blend< -2: ver,color = "SELL", "salmon"
else:           ver,color = "HOLD", "khaki"

# ─── 6 | Render Tabs ────────────────────────────────────────────────
tab_v,tab_ta,tab_f,tab_r = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"]
)

with tab_v:
    st.markdown(f"<h2 style='color:{color};text-align:center'>{ver}</h2>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score" , f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score" , f"{sent_val:.2f}")
    c4.metric("Blended"    , f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech  +  {sent_w}% Sent")

with tab_ta:
    df = price.loc[start:today]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Adj Close"],name="Price"))
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index,y=df["SMA_20"],name="SMA-20",
                                 line=dict(dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],name="Upper BB",
                                 line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],name="Lower BB",
                                 line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark",height=350)
    st.plotly_chart(fig,use_container_width=True)
    if show_macd: st.line_chart(df["MACD"],height=180)
    if show_rsi:  st.line_chart(df["RSI"], height=180)

with tab_f:
    st.table(pd.DataFrame({
        "Metric":["P/E","Debt/Equity","EV/EBITDA"],
        "Value":[fund["pe"],fund["de"],fund["ev"]]
    }).set_index("Metric"))

with tab_r:
    if df_posts.empty:
        st.info("No posts in the last week.")
    else:
        st.dataframe(df_posts, hide_index=True, use_container_width=True)
