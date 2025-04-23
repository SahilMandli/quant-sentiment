# app.py
import os, time, shutil, base64, textwrap
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
UA            = {"User-Agent":"Mozilla/5.0 (ValueTron/1.2)"}
COLLECT_EVERY = 3*3600        # refresh Reddit cache every 3 h
POST_LIMIT    = 40
POSTS_CSV     = "reddit_posts.csv"
SENTS_CSV     = "reddit_sentiments.csv"
BASE_SENT_CSV = "/mnt/data/reddit_sentiments.csv"  # your original dump
PRICE_TTL     = 900          # 15 min price cache

# ─── PAGE SETUP ────────────────────────────────────────────────────
st.set_page_config("📈 ValueTron", "⚡️", layout="wide")
# Optional Tron background if tron.png exists
if os.path.exists("tron.png"):
    b64 = base64.b64encode(open("tron.png","rb").read()).decode()
    st.markdown(textwrap.dedent(f"""
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
        body, .stApp {{
          background:
            linear-gradient(rgba(0,0,0,.92),rgba(0,0,0,.92)),
            url("data:image/png;base64,{b64}") center/cover fixed;
          color:#fff; font-family:'Orbitron', sans-serif;
        }}
        h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
        .stSidebar{{background:rgba(0,0,30,.95);border-right:2px solid #0ff}}
      </style>"""), unsafe_allow_html=True)
st.markdown("<h1>⚡️ ValueTron</h1>", unsafe_allow_html=True)
st_autorefresh(interval=1_800_000, key="full_refresh")  # reload page every 30 min

# ─── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    tf       = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], index=1)
    tkr      = st.selectbox("Ticker", TICKERS, index=0)
    tech_w   = st.slider("Technical Weight %", 0, 100, 60)
    sent_w   = 100-tech_w

    st.markdown("### Technical Indicators")
    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD",   True)
    show_rsi  = st.checkbox("RSI",    True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("### Fundamental Ratios")
    show_pe = st.checkbox("P/E",           True)
    show_de = st.checkbox("Debt / Equity", True)
    show_ev = st.checkbox("EV / EBITDA",   True)

# ─── 0. REFRESH REDDIT CACHE (silent) ───────────────────────────────
def reddit_rows(sym):
    rows=[]
    # 1) try reddit.com
    for sub in SUBS:
        url = (f"https://www.reddit.com/r/{sub}/search.json"
               f"?q={sym}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1")
        try:
            r=requests.get(url, headers=UA, timeout=8)
            if r.ok:
                for c in r.json().get("data",{}).get("children",[]):
                    d=c["data"]
                    rows.append({
                        "ticker":sym,
                        "title":  d.get("title",""),
                        "text":   d.get("selftext",""),
                        "score":  d.get("score",0)
                    })
        except: pass
        time.sleep(0.3)
    if rows: return rows
    # 2) fall back to Pushshift
    base="https://api.pushshift.io/reddit/search/submission/"
    for sub in SUBS:
        url=f"{base}?q={sym}&subreddit={sub}&after=7d&size={POST_LIMIT}&sort=desc"
        try:
            for d in requests.get(url, timeout=8).json().get("data",[]):
                rows.append({
                    "ticker":sym,
                    "title": d.get("title",""),
                    "text":  d.get("selftext",""),
                    "score": d.get("score",0)
                })
        except: pass
        time.sleep(0.2)
    return rows

def refresh_cache():
    if os.path.exists(SENTS_CSV) and time.time()-os.path.getmtime(SENTS_CSV)<COLLECT_EVERY:
        return
    allr=[r for s in TICKERS for r in reddit_rows(s)]
    if not allr:
        return
    df=pd.DataFrame(allr)
    sia=SentimentIntensityAnalyzer()
    df["sentiment"]=(df["title"].fillna("")+" "+df["text"].fillna("")).apply(
        lambda t:(TextBlob(t).sentiment.polarity + sia.polarity_scores(t)["compound"])/2
    )
    df.to_csv(POSTS_CSV,index=False)
    df.groupby("ticker")["sentiment"].mean().round(4)\
      .reset_index().to_csv(SENTS_CSV,index=False)

refresh_cache()

# ─── 1. LOAD & CLASSIFY SENTIMENT ──────────────────────────────────
sent_val=0.0
if os.path.exists(SENTS_CSV):
    try:
        df_s=pd.read_csv(SENTS_CSV).set_index("ticker")
        sent_val=float(df_s.at[tkr,"sentiment"])
    except: sent_val=0.0

# fallback to original CSV dump
if sent_val==0.0 and os.path.exists(BASE_SENT_CSV):
    try:
        df_b=pd.read_csv(BASE_SENT_CSV)
        sent_val=float(df_b.groupby("ticker")["sentiment_score"].mean().get(tkr,0.0))
    except: sent_val=0.0

if   sent_val>0.20: sent_rating="A"
elif sent_val<-0.20: sent_rating="C"
else:               sent_rating="B"

try:
    df_posts=(pd.read_csv(POSTS_CSV)
                .query("ticker==@tkr")[["title","score"]]
                .head(20))
except:
    df_posts=pd.DataFrame(columns=["title","score"])

# ─── 2. PRICE + INDICATORS ─────────────────────────────────────────
@st.cache_data(ttl=PRICE_TTL)
def load_price(sym,start,end):
    raw = yf.download(sym,
                      start=start,
                      end=end+dt.timedelta(days=1),
                      progress=False,
                      auto_adjust=False)
    if raw.empty: return None

    # flatten any MultiIndex
    if isinstance(raw.columns,pd.MultiIndex):
        raw.columns=raw.columns.get_level_values(1)

    raw.columns=raw.columns.str.replace(" ","").str.lower()
    cols=raw.columns
    if "adjclose" in cols:
        base="adjclose"
    elif any(c.startswith("close") for c in cols):
        base=[c for c in cols if c.startswith("close")][0]
    else:
        base=raw.select_dtypes("number").columns[0]

    df=raw.copy()
    df["Adj Close"]=df[base] if isinstance(df[base],pd.Series) else df[base].iloc[:,0]

    df["SMA_20"]=df["Adj Close"].rolling(20).mean()
    df["MACD"]=df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta=df["Adj Close"].diff()
    rs=(delta.clip(lower=0).rolling(14).mean()/
        (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan))
    df["RSI"]=100-100/(1+rs)
    std=df["Adj Close"].rolling(20).std()
    df["BB_Upper"]=df["SMA_20"]+2*std
    df["BB_Lower"]=df["SMA_20"]-2*std

    return df

today=dt.date.today()
days={"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf,365)
start=dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=days)
price=load_price(tkr,start,today)
if price is None:
    st.error("Price data unavailable."); st.stop()
last=price.iloc[-1]

@st.cache_data(ttl=86400)
def load_fund(sym):
    info=yf.Ticker(sym).info
    return {
        "pe":info.get("trailingPE",np.nan),
        "de":info.get("debtToEquity",np.nan),
        "ev":info.get("enterpriseToEbitda",np.nan)
    }

fund=load_fund(tkr)

# ─── 3. COMPUTE SCORES ─────────────────────────────────────────────
tech=0.0
if show_sma and not np.isnan(last.SMA_20):
    tech+=1 if last["Adj Close"]>last["SMA_20"] else -1
if show_macd and not np.isnan(last.MACD):
    tech+=1 if last["MACD"]>0 else -1
if show_rsi and not np.isnan(last.RSI):
    tech+=1 if 40<last["RSI"]<70 else -1
if show_bb and not (np.isnan(last.BB_Upper) or np.isnan(last.BB_Lower)):
    tech+=0.5 if last["Adj Close"]>last["BB_Upper"] else 0
    tech-=0.5 if last["Adj Close"]<last["BB_Lower"] else 0

if show_pe and not np.isnan(fund["pe"]):
    tech+=1 if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]):
    tech+=0.5 if fund["de"]<1 else -0.5
if show_ev and not np.isnan(fund["ev"]):
    tech+=1 if fund["ev"]<12 else -1

blend=tech_w/100*tech + sent_w/100*sent_val
if blend>2:    ver,color="BUY","springgreen"
elif blend<-2: ver,color="SELL","salmon"
else:          ver,color="HOLD","khaki"

# ─── 4. RENDER TABS ────────────────────────────────────────────────
tab_v,tab_ta,tab_f,tab_r = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"]
)

with tab_v:
    st.markdown(f"<h2 style='color:{color};text-align:center'>{ver}</h2>",
                unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Tech Score",  f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score",  f"{sent_val:.2f}")
    c4.metric("Blended",     f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech • {sent_w}% Sentiment")

with tab_ta:
    dfp=price.loc[start:today]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dfp.index,y=dfp["Adj Close"],name="Price"))
    if show_sma:
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp["SMA_20"],name="SMA-20",
                                 line=dict(dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp["BB_Upper"],name="Upper BB",
                                 line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp["BB_Lower"],name="Lower BB",
                                 line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark",height=340)
    st.plotly_chart(fig,use_container_width=True)
    if show_macd: st.line_chart(dfp["MACD"],height=180)
    if show_rsi:  st.line_chart(dfp["RSI"], height=180)

with tab_f:
    st.table(pd.DataFrame({
        "Metric":["P/E","Debt / Equity","EV / EBITDA"],
        "Value":[fund["pe"],fund["de"],fund["ev"]]
    }).set_index("Metric"))

with tab_r:
    if df_posts.empty:
        st.info("No recent posts.")
    else:
        st.dataframe(df_posts, hide_index=True, use_container_width=True)
