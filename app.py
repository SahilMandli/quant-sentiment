import streamlit as st
import pandas as pd, numpy as np, plotly.graph_objects as go
import yfinance as yf, datetime as dt, requests, base64, os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ────────────────── PAGE CONFIG & STYLE ──────────────────────────────
st.set_page_config(page_title="⚡️ Quant Sentiment", page_icon="📈", layout="wide")

if os.path.exists("tron.png"):
    bg64 = base64.b64encode(open("tron.png", "rb").read()).decode()
    st.markdown(
        f"""
        <style>
          body,.stApp{{background:
            linear-gradient(rgba(0,0,0,.9),rgba(0,0,0,.9)),
            url("data:image/png;base64,{bg64}") center/cover fixed;
            color:#fff;font-family:Arial}}
          h1{{color:#0ff;text-align:center;text-shadow:0 0 6px #0ff}}
          .stSidebar{{background:rgba(0,0,30,.93);border-right:2px solid #0ff}}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)

# ───────────────────── SIDEBAR ───────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    tf = st.selectbox("Timeframe", ["1W", "1M", "6M", "YTD", "1Y"], index=1)
    tkr = st.text_input("Ticker", "NVDA").strip().upper()
    tech_w = st.slider("Technical Weight %", 0, 100, 60)
    sent_w = 100 - tech_w

    # technical factors
    show_sma = st.checkbox("SMA‑20", True)
    show_macd = st.checkbox("MACD", True)
    show_rsi = st.checkbox("RSI", True)
    show_bb = st.checkbox("Bollinger Bands", True)

    st.markdown("---")
    # fundamental factors
    show_pe = st.checkbox("P/E ratio", True)
    show_de = st.checkbox("Debt / Equity", True)
    show_ev = st.checkbox("EV / EBITDA", True)

    run = st.button("🚀 Analyze")

if not run or not tkr:
    st.stop()

# ─────────── DATE RANGE ──────────────────────────────────────────────
today = dt.date.today()
start = dt.date(today.year, 1, 1) if tf == "YTD" else today - dt.timedelta(
    days={"1W": 7, "1M": 30, "6M": 180, "1Y": 365}[tf]
)

# ─────────── PRICE & INDICATORS ──────────────────────────────────────
@st.cache_data(ttl=900)
def load_price(tkr, start, end):
    raw = yf.download(tkr, start=start, end=end + dt.timedelta(days=1),
                      progress=False, group_by="ticker")

    # flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(tkr, level=0, axis=1)

    if raw.empty:
        return None

    df = raw.copy()
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["MACD"] = df["Adj Close"].ewm(12).mean() - df["Adj Close"].ewm(26).mean()
    delta = df["Adj Close"].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (
        -delta.clip(upper=0).rolling(14).mean()
    ).replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    std = df["Adj Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * std
    df["BB_Lower"] = df["SMA_20"] - 2 * std
    return df


price = load_price(tkr, start, today)
if price is None:
    st.error("No price data."); st.stop()

price = price.dropna(subset=["Adj Close"])
if price.empty:
    st.error("No usable rows for this period."); st.stop()
last = price.iloc[-1]

# ─────────── FUNDAMENTALS ────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fundamentals(tkr):
    tick = yf.Ticker(tkr)
    info = tick.fast_info or {}
    pe = info.get("trailingPe", np.nan)
    de = info.get("debtToEquity", np.nan)
    ev = info.get("evToEbitda", np.nan)

    if np.isnan(pe) or np.isnan(de) or np.isnan(ev):
        try:
            full = tick.info
            pe = full.get("trailingPE", pe)
            de = full.get("debtToEquity", de)
            ev = full.get("enterpriseToEbitda", ev)
        except Exception:
            pass
    return dict(pe=pe, de=de, ev=ev)


fund = fundamentals(tkr)

# ─────────── REDDIT SENTIMENT (Pushshift only) ───────────────────────
@st.cache_data(ttl=300)
def reddit_sentiment(tkr):
    """
    Fetch latest 60 submissions mentioning the ticker via Reddit’s own
    JSON search (no auth needed).  Returns avg‑sentiment, A/B/C rating,
    and DataFrame of titles & scores.
    """
    headers = {"User-Agent": "Mozilla/5.0 (QuantDash)"}
    subs    = ["stocks", "investing", "wallstreetbets"]
    rows    = []

    for sub in subs:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={tkr}&restrict_sr=1&sort=new&limit=20"
        )
        try:
            r = requests.get(url, headers=headers, timeout=10)
            for child in r.json().get("data", {}).get("children", []):
                d = child["data"]
                rows.append(
                    {
                        "title": d.get("title", ""),
                        "text": d.get("selftext", ""),
                        "score": d.get("score", 0),
                    }
                )
        except Exception:
            pass  # ignore sub‑level errors

    if not rows:
        return 0.0, "B", pd.DataFrame()

    sia = SentimentIntensityAnalyzer()

    def hybrid(x):
        txt = f"{x['title']} {x['text']}"
        base = (TextBlob(txt).sentiment.polarity + sia.polarity_scores(txt)["compound"]) / 2
        weight = min(x["score"], 100) / 100
        return base * weight

    avg = sum(hybrid(x) for x in rows) / len(rows)
    rating = "A" if avg > 0.2 else "C" if avg < -0.2 else "B"
    df = pd.DataFrame([{"title": x["title"], "score": x["score"]} for x in rows])

    return avg, rating, df

# ─────────── TECH + FUND SCORE ───────────────────────────────────────
tech = 0.0
if show_sma and "SMA_20" in last:
    tech += 1 if last["Adj Close"] > last["SMA_20"] else -1
if show_macd and "MACD" in last:
    tech += 1 if last["MACD"] > 0 else -1
if show_rsi and "RSI" in last:
    tech += 1 if 40 < last["RSI"] < 70 else -1
if show_bb and {"BB_Upper", "BB_Lower"}.issubset(last.index):
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

blend = tech_w / 100 * tech + sent_w / 100 * sent_val
ver, color = (
    ("BUY", "springgreen") if blend > 2 else
    ("SELL", "salmon") if blend < -2 else
    ("HOLD", "khaki")
)

# ─────────── DISPLAY TABS ────────────────────────────────────────────
tab_v, tab_ta, tab_f, tab_r = st.tabs(
    ["🏁 Verdict", "📈 Technical", "📊 Fundamentals", "🗣️ Reddit"]
)

with tab_v:
    st.header("Overall Verdict")
    st.markdown(
        f"<h1 style='color:{color};text-align:center'>{ver}</h1>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tech Score", f"{tech:.2f}")
    c2.metric("Sent Rating", sent_rating)
    c3.metric("Sent Score", f"{sent_val:.2f}")
    c4.metric("Blended", f"{blend:.2f}")
    st.caption(f"{tech_w}% Tech + {sent_w}% Sentiment")

with tab_ta:
    df = price.loc[start:today]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Adj Close"], name="Price", line=dict(color="#0ff"))
    )
    if show_sma and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="SMA‑20",
                line=dict(color="#ff0", dash="dash"),
            )
        )
    if show_bb and {"BB_Upper", "BB_Lower"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="Upper BB",
                line=dict(color="#0f0", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="Lower BB",
                line=dict(color="#0f0", dash="dot"),
            )
        )
    fig.update_layout(template="plotly_dark", height=350,
                      title="Price / SMA / Bollinger")
    st.plotly_chart(fig, use_container_width=True)

    if show_macd and "MACD" in df.columns:
        st.line_chart(df["MACD"], height=200)
    if show_rsi and "RSI" in df.columns:
        st.line_chart(df["RSI"], height=200)

    st.subheader("Candlestick")
    candle = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="#0ff",
                decreasing_line_color="#f44",
            )
        ]
    )
    candle.update_layout(
        template="plotly_dark", height=420, xaxis_rangeslider_visible=False
    )
    st.plotly_chart(candle, use_container_width=True)

with tab_f:
    st.header("Key Ratios")
    st.table(
        pd.DataFrame(
            {
                "Metric": ["P/E", "Debt / Equity", "EV / EBITDA"],
                "Value": [fund["pe"], fund["de"], fund["ev"]],
            }
        ).set_index("Metric")
    )

with tab_r:
    st.header("Latest Reddit Mentions")
    if df_posts.empty:
        st.info("No recent posts.")
    else:
        st.dataframe(df_posts.head(20), use_container_width=True)
