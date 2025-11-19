import os, time
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
import base64, textwrap

# ─── CONFIG ────────────────────────────────────────────────────────
TICKERS     = ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR","GME"]
FULLNAME = {
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "ADBE": "Adobe",
    "VRTX": "Vertex Pharma",
    "SCHW": "Charles Schwab",
    "CROX": "Crocs",
    "DE": "Deere & Co.",
    "FANG": "Diamondback Energy",
    "TMUS": "T-Mobile US",
    "PLTR": "Palantir",
}
SUBS        = ["stocks","investing","wallstreetbets"]
UA          = {"User-Agent":"Mozilla/5.0 (ValueTron/1.5)"}
REFRESH_SEC = 3 * 3600       # refresh Reddit every 3 h
POST_LIMIT  = 40
POSTS_CSV   = "reddit_posts.csv"
PRICE_TTL   = 900            # 15 min price cache

# ─── PAGE SETUP & BACKGROUND ──────────────────────────────────────
st.set_page_config(page_title="📈 ValueTron", page_icon="⚡️", layout="wide")
# Optional Tron-style background
if os.path.exists("tron.png"):
    with open("tron.png","rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(textwrap.dedent(f"""
    <style>
      body, .stApp {{
        background: linear-gradient(rgba(0,0,0,.9), rgba(0,0,0,.9)),
                    url("data:image/png;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Segoe UI', sans-serif;
        color: white;
      }}
    </style>
    """), unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>⚡️ ValueTron</h1>", unsafe_allow_html=True)
st_autorefresh(interval=30 * 60 * 1000, key="reload")  # reload every 30 min

# ─── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    tf     = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], index=1)
    tkr    = st.selectbox("Ticker", TICKERS, index=0)
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    # Show company name and current price
    try:
        info = yf.Ticker(tkr).info
        name = info.get("shortName")
        price_now = info.get("currentPrice")
        if name:
            st.markdown(f"**{tkr} – {name}**")
        if price_now:
            st.markdown(f"💲{price_now:.2f}")
    except:
        pass

    st.markdown("### Technical Indicators")
    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD",   True)
    show_rsi  = st.checkbox("RSI",    True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("### Fundamental Ratios")
    show_pe = st.checkbox("P/E ratio",      True)
    show_de = st.checkbox("Debt / Equity",  True)
    show_ev = st.checkbox("EV / EBITDA",    True)

# ─── 0 | FETCH & CACHE REDDIT POSTS (silent) ───────────────────────
def fetch_reddit(ticker):
    rows = []
    for sub in SUBS:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={ticker}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1"
        )
        try:
            r = requests.get(url, headers=UA, timeout=8)
            if r.ok:
                for c in r.json().get("data", {}).get("children", []):
                    d = c["data"]
                    rows.append({
                        "ticker": ticker,
                        "title":   d.get("title", ""),
                        "text":    d.get("selftext", ""),
                        "score":   d.get("score", 0)
                    })
        except:
            pass
        time.sleep(0.3)
    if rows:
        return rows
    base = "https://api.pushshift.io/reddit/search/submission/"
    for sub in SUBS:
        url = f"{base}?q={ticker}&subreddit={sub}&after=7d&size={POST_LIMIT}&sort=desc"
        try:
            for d in requests.get(url, timeout=8).json().get("data", []):
                rows.append({
                    "ticker": ticker,
                    "title":   d.get("title", ""),
                    "text":    d.get("selftext", ""),
                    "score":   d.get("score", 0)
                })
        except:
            pass
        time.sleep(0.2)
    return rows

def refresh_reddit():
    if os.path.exists(POSTS_CSV):
        age = time.time() - os.path.getmtime(POSTS_CSV)
        if age < REFRESH_SEC:
            return
    all_posts = []
    for tk in TICKERS:
        all_posts += fetch_reddit(tk)
    if all_posts:
        pd.DataFrame(all_posts).to_csv(POSTS_CSV, index=False)

refresh_reddit()

# ─── 1 | LOAD POSTS CSV & COMPUTE SENTIMENT ────────────────────────
posts_all = pd.read_csv(POSTS_CSV) if os.path.exists(POSTS_CSV) else pd.DataFrame(columns=["ticker","title","text","score"])
# Ensure 'text' exists
if 'text' not in posts_all.columns:
    posts_all['text'] = ''

sia = SentimentIntensityAnalyzer()
def hybrid_score(text, upvotes):
    tb = TextBlob(text).sentiment.polarity
    vd = sia.polarity_scores(text)["compound"]
    return ((tb + vd) / 2) * min(upvotes, 100) / 100

posts_all['combined'] = posts_all['title'].fillna('') + ' ' + posts_all['text'].fillna('')
posts_all['sentiment_score'] = posts_all.apply(lambda r: hybrid_score(r['combined'], r['score']), axis=1)
# letter rating
posts_all['rating'] = posts_all['sentiment_score'].apply(lambda x: 'A' if x>0.05 else 'C' if x<-0.05 else 'B')

# save overall rating per ticker
overall = posts_all.groupby('ticker')['rating']\
    .apply(lambda x: x.value_counts().idxmax())\
    .reset_index(name='overall_rating')
overall.to_csv('reddit_sentiment_rating.csv', index=False)

# Filter for current ticker
posts_df = posts_all.loc[posts_all.ticker==tkr, ['title','rating']].head(20)
posts_df.columns = ['Reddit Post','Sentiment']
# Compute average
sub = posts_all.loc[posts_all.ticker==tkr, 'sentiment_score']
sent_val = sub.mean() if not sub.empty else 0.0
sent_rating = 'A' if sent_val>0.05 else 'C' if sent_val<-0.05 else 'B'

# ─── 2 | PRICE + INDICATORS ───────────────────────────────────────
@st.cache_data(ttl=PRICE_TTL)
def load_price(sym, start, end):
    df = yf.download(sym, start=start, end=end+dt.timedelta(days=1), progress=False, auto_adjust=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(-1)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    if 'adj_close' in df.columns: df['Adj Close'] = df['adj_close']
    elif 'close' in df.columns: df['Adj Close'] = df['close']
    else: df['Adj Close'] = df.select_dtypes('number').iloc[:,0]
    df['SMA_20']    = df['Adj Close'].rolling(20).mean()
    df['MACD']      = df['Adj Close'].ewm(span=12).mean() - df['Adj Close'].ewm(span=26).mean()
    delta          = df['Adj Close'].diff()
    rs             = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df['RSI']      = 100 - 100/(1+rs)
    std            = df['Adj Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + 2*std
    df['BB_Lower'] = df['SMA_20'] - 2*std
    return df

today = dt.date.today()
days  = {"1W":7,"1M":30,"6M":180,"1Y":365}[tf]
start = dt.date(today.year,1,1) if tf=='YTD' else today-dt.timedelta(days=days)
price = load_price(tkr, start, today)
if price is None:
    st.error(f"❌ No price data for {tkr}")
    st.stop()
last = price.iloc[-1]

@st.cache_data(ttl=86400)
def load_fund(sym):
    info = yf.Ticker(sym).info
    return {"pe": info.get("trailingPE", np.nan),
            "de": info.get("debtToEquity", np.nan),
            "ev": info.get("enterpriseToEbitda", np.nan)}

fund = load_fund(tkr)

# ─── 3 | TECH + FUND SCORE ─────────────────────────────────────────
tech = 0.0
if show_sma and not pd.isna(last['SMA_20']): tech += 1 if last['Adj Close']>last['SMA_20'] else -1
if show_macd and not pd.isna(last['MACD']): tech += 1 if last['MACD']>0 else -1
if show_rsi and not pd.isna(last['RSI']): tech += 1 if 40<last['RSI']<70 else -1
if show_bb and not (pd.isna(last['BB_Upper']) or pd.isna(last['BB_Lower'])):
    tech += 0.5 if last['Adj Close']>last['BB_Upper'] else 0
    tech -= 0.5 if last['Adj Close']<last['BB_Lower'] else 0
if show_pe and not pd.isna(fund['pe']): tech += 1.0 if fund['pe']<18 else -1.0
if show_de and not pd.isna(fund['de']): tech += 0.5 if fund['de']<1 else -0.5
if show_ev and not pd.isna(fund['ev']): tech += 1.0 if fund['ev']<12 else -1.0

# ─── 4 | BLEND + VERDICT ───────────────────────────────────────────
blend = tech_w/100*tech + sent_w/100*sent_val
if blend>2:      ver,clr = 'BUY','springgreen'
elif blend< -2: ver,clr = 'SELL','salmon'
else:           ver,clr = 'HOLD','khaki'

# ─── 5 | UI TABS ───────────────────────────────────────────────────
tab_v,tab_ta,tab_f,tab_r = st.tabs(["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])
with tab_v:
    st.markdown(f"<h2 style='color:{clr};text-align:center'>{ver}</h2>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score", f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score", f"{sent_val:.2f}")
    c4.metric("Blended", f"{blend:.2f}")
with tab_ta:
    dfp = price.loc[start:today]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp.index, y=dfp['Adj Close'], name='Price'))
    if show_sma: fig.add_trace(go.Scatter(x=dfp.index, y=dfp['SMA_20'], name='SMA-20', line=dict(dash='dash')))
    if show_bb:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp['BB_Upper'], name='Upper BB', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp['BB_Lower'], name='Lower BB', line=dict(dash='dot')))
    fig.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig, use_container_width=True)
    if show_macd: st.line_chart(dfp['MACD'], height=180)
    if show_rsi:  st.line_chart(dfp['RSI'], height=180)
with tab_f:
    rat = pd.DataFrame({ 'Metric':['P/E','Debt / Equity','EV / EBITDA'], 'Value':[fund['pe'],fund['de'],fund['ev']] }).set_index('Metric')
    st.table(rat)
with tab_r:
    if posts_df.empty:
        st.info('No Reddit posts.')
    else:
        st.dataframe(posts_df, hide_index=True, use_container_width=True) 
        
# ── Friendly explanation  ---------------------------------
    explanation_parts = []

    # Technical side
    if tech > 0:
        explanation_parts.append(
            f"Technical indicators are net **bullish** *(+{tech:.1f})*."
        )
    elif tech < 0:
        explanation_parts.append(
            f"Technical indicators are net **bearish** *({tech:.1f})*."
        )
    else:
        explanation_parts.append("Technical indicators are **neutral** *(0)*.")

    # Sentiment side
    if sent_val > 0.05:
        explanation_parts.append(
            f"Reddit sentiment is **positive** *(+{sent_val:.2f})*."
        )
    elif sent_val < -0.05:
        explanation_parts.append(
            f"Reddit sentiment is **negative** *({sent_val:.2f})*."
        )
    else:
        explanation_parts.append("Reddit sentiment is **neutral**.")

    # Blend comment
    explanation_parts.append(
        f"Blending **{tech_w}%** technical with **{sent_w}%** sentiment "
        f"yields a combined score of **{blend:.2f}**, → **{ver}** signal."
    )

    # Render paragraph
    st.markdown(" ".join(explanation_parts), unsafe_allow_html=True)
