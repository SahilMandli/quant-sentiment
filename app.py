# ═══════════════════════════════════════════════════════════════════
#  ValueTron – Quant-Sentiment Dashboard
#  2025-04-23 (KeyError & duplicate-column fixes)
# ═══════════════════════════════════════════════════════════════════
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, requests, time, datetime as dt, os, base64, textwrap
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh

# ─── constants ────────────────────────────────────────────────────
TICKERS       = ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR"]
SUBS          = ["stocks","investing","wallstreetbets"]
UA            = {"User-Agent":"Mozilla/5.0 (ValueTron/1.1)"}
COLLECT_EVERY = 3*3600            # refresh Reddit every 3 h
POST_LIMIT    = 40
POSTS_CSV     = "reddit_posts.csv"
SENTS_CSV     = "reddit_sentiments.csv"
PRICE_TTL     = 900               # cache prices 15 min

# ─── page config & optional Tron CSS ─────────────────────────────
st.set_page_config(page_title="📈 ValueTron", page_icon="⚡️", layout="wide")

if os.path.exists("tron.png"):
    with open("tron.png","rb") as f: bg = base64.b64encode(f.read()).decode()
    st.markdown(textwrap.dedent(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
      body, .stApp {{
        background:
          linear-gradient(rgba(0,0,0,.92),rgba(0,0,0,.92)),
          url("data:image/png;base64,{bg}") center/cover fixed;
        color:#fff; font-family:'Orbitron',sans-serif;
      }}
      h1 {{color:#0ff; text-shadow:0 0 6px #0ff; text-align:center}}
      .stSidebar {{background:rgba(0,0,30,.95); border-right:2px solid #0ff}}
    </style>"""), unsafe_allow_html=True)

st.markdown("<h1>⚡️ ValueTron</h1>", unsafe_allow_html=True)
st_autorefresh(interval=1_800_000, key="auto")   # 30 min auto-refresh

# ─── sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    tf  = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y"], 1)
    tkr = st.selectbox("Ticker", TICKERS, 0)
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    st.markdown("### Technical Indicators")
    show_sma  = st.checkbox("SMA-20", True)
    show_macd = st.checkbox("MACD",   True)
    show_rsi  = st.checkbox("RSI",    True)
    show_bb   = st.checkbox("Bollinger Bands", True)

    st.markdown("### Fundamental Ratios")
    show_pe = st.checkbox("P/E", True)
    show_de = st.checkbox("Debt / Equity", True)
    show_ev = st.checkbox("EV / EBITDA", True)

# ─── 0 · Reddit fetch → sentiment CSV (silent) ────────────────────
def reddit_rows(sym: str):
    rows=[]
    for sub in SUBS:
        url = (f"https://www.reddit.com/r/{sub}/search.json"
               f"?q={sym}&restrict_sr=1&sort=new&limit={POST_LIMIT}&raw_json=1")
        try:
            r = requests.get(url, headers=UA, timeout=10)
            if r.status_code == 200:
                for c in r.json().get("data",{}).get("children",[]):
                    d=c["data"]
                    rows.append({"ticker":sym,
                                 "title":d.get("title",""),
                                 "text": d.get("selftext",""),
                                 "score":d.get("score",0)})
        except: pass
        time.sleep(0.4)
    if rows: return rows
    base="https://api.pushshift.io/reddit/search/submission/"
    for sub in SUBS:
        url = (f"{base}?q={sym}&subreddit={sub}"
               f"&after=7d&size={POST_LIMIT}&sort=desc")
        try:
            for d in requests.get(url,timeout=10).json().get("data",[]):
                rows.append({"ticker":sym,
                             "title":d.get("title",""),
                             "text": d.get("selftext",""),
                             "score":d.get("score",0)})
        except: pass
        time.sleep(0.3)
    return rows

def refresh_reddit_cache():
    if os.path.exists(SENTS_CSV) and time.time()-os.path.getmtime(SENTS_CSV)<COLLECT_EVERY:
        return
    allr=[r for sym in TICKERS for r in reddit_rows(sym)]
    if not allr: return
    df=pd.DataFrame(allr)
    sia=SentimentIntensityAnalyzer()
    df["sentiment"] = (df["title"].fillna("")+" "+df["text"].fillna("")
                      ).apply(lambda t:(TextBlob(t).sentiment.polarity +
                                        sia.polarity_scores(t)["compound"]) / 2)
    df.to_csv(POSTS_CSV,index=False)
    df.groupby("ticker")["sentiment"].mean().round(4).reset_index(
      ).to_csv(SENTS_CSV,index=False)

refresh_reddit_cache()

# ─── 1 · load sentiment (CSV fallback) ────────────────────────────
try:
    sent_val = float(pd.read_csv(SENTS_CSV).set_index("ticker").at[tkr,"sentiment"])
except: sent_val=0.0
sent_rating = "A" if sent_val>0.20 else "C" if sent_val<-0.20 else "B"

try:
    df_posts = (pd.read_csv(POSTS_CSV)
                  .query("ticker==@tkr")[["title","score"]]
                  .head(20))
except: df_posts=pd.DataFrame()

# ─── 2 · load price + add indicators (no KeyError) ────────────────
@st.cache_data(ttl=PRICE_TTL)
def load_price(sym:str, start:dt.date, end:dt.date):
    raw = yf.download(sym,
                      start=start,
                      end=end+dt.timedelta(days=1),
                      progress=False,
                      auto_adjust=False)
    if raw.empty: return None

    # flatten MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.map(lambda x:x[1])

    # normalise to lowercase & no spaces
    raw.columns = raw.columns.str.replace(" ","").str.lower()

    # prefer 'adjclose' if present, else first 'close*', else numeric
    cols = raw.columns
    if "adjclose" in cols:
        base="adjclose"
    elif any(c.startswith("close") for c in cols):
        base=[c for c in cols if c.startswith("close")][0]
    else:
        base = raw.select_dtypes("number").columns[0]

    df = raw.copy()
    df["Adj Close"] = df[base]   # single Series → no duplicates

    # technical indicators
    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["MACD"]   = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs    = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    df["RSI"]   = 100 - 100/(1+rs)
    std = df["Adj Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2*std
    df["BB_Lower"] = df["SMA_20"] - 2*std

    return df

today = dt.date.today()
days  = {"1W":7,"1M":30,"6M":180,"1Y":365}.get(tf,365)
start = dt.date(today.year,1,1) if tf=="YTD" else today-dt.timedelta(days=days)
price = load_price(tkr, start, today)
if price is None:
    st.error("Price data unavailable."); st.stop()
last = price.iloc[-1]

@st.cache_data(ttl=86_400)
def fundamentals(sym:str):
    info = yf.Ticker(sym).info
    return {"pe":info.get("trailingPE",np.nan),
            "de":info.get("debtToEquity",np.nan),
            "ev":info.get("enterpriseToEbitda",np.nan)}
fund = fundamentals(tkr)

# ─── 3 · compute blended tech + sentiment score ────────────────
tech = 0
if show_sma:  tech += 1 if last["Adj Close"]>last["SMA_20"] else -1
if show_macd: tech += 1 if last["MACD"]>0 else -1
if show_rsi:  tech += 1 if 40<last["RSI"]<70 else -1
if show_bb:
    tech += 0.5 if last["Adj Close"]>last["BB_Upper"] else 0
    tech -= 0.5 if last["Adj Close"]<last["BB_Lower"] else 0
if show_pe and not np.isnan(fund["pe"]): tech += 1 if fund["pe"]<18 else -1
if show_de and not np.isnan(fund["de"]): tech += 0.5 if fund["de"]<1 else -0.5
if show_ev and not np.isnan(fund["ev"]): tech += 1 if fund["ev"]<12 else -1

blend = tech_w/100 * tech + sent_w/100 * sent_val
ver, color = ("BUY","springgreen") if blend>2 else ("SELL","salmon") if blend<-2 else ("HOLD","khaki")

# ─── 4 · render UI tabs ───────────────────────────────────────────
tab_v,tab_ta,tab_f,tab_r = st.tabs(
    ["🏁 Verdict","📈 Technical","📊 Fundamentals","🗣️ Reddit"])

with tab_v:
    st.markdown(f"<h2 style='color:{color};text-align:center'>{ver}</h2>",unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tech Score",f"{tech:.2f}")
    c2.metric("Sent Rating",sent_rating)
    c3.metric("Sent Score",f"{sent_val:.2f}")
    c4.metric("Blended",f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech • {sent_w}% Sentiment")

with tab_ta:
    df=price.loc[start:today]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Adj Close"],name="Price"))
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index,y=df["SMA_20"],name="SMA-20",line=dict(dash="dash")))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],name="Upper BB",line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],name="Lower BB",line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark",height=340)
    st.plotly_chart(fig,use_container_width=True)
    if show_macd: st.line_chart(df["MACD"],height=180)
    if show_rsi:  st.line_chart(df["RSI"],height=180)

with tab_f:
    st.table(pd.DataFrame({
        "Metric":["P/E","Debt / Equity","EV / EBITDA"],
        "Value":[fund["pe"],fund["de"],fund["ev"]]
    }).set_index("Metric"))

with tab_r:
    if not df_posts.empty:
        st.dataframe(df_posts,hide_index=True,use_container_width=True)
