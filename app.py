import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import base64

# ─── Must be first Streamlit command ─────────────
st.set_page_config(
    page_title="⚡️ Quant Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Background: TRON PNG & Styles ───────────────
with open("tron.png", "rb") as img:
    b64 = base64.b64encode(img.read()).decode()

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
<style>
  body, .stApp {{
    background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url("data:image/png;base64,{b64}") no-repeat center center fixed;
    background-size: cover;
    font-family: 'Orbitron', sans-serif;
    color: #FFF;
    margin: 0; padding: 0;
  }}
  header, footer {{ visibility: hidden; }}
  h1 {{ font-size:2.5rem; color:#0ff; text-align:center; text-shadow:0 0 4px #0ff; margin-bottom:0.5rem; }}
  .stSidebar {{ background: rgba(0,0,20,0.9); border-right:2px solid #0ff; }}
  h2 {{ font-size:1.8rem; color:#ff0; border-bottom:2px solid #0ff; padding-bottom:4px; margin-top:1.5rem; text-shadow:0 0 4px #ff0; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡️ Quant Sentiment Dashboard</h1>", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    timeframe = st.selectbox("Timeframe", ["1W","1M","6M","YTD","1Y","Custom"])
    if timeframe != "Custom":
        today = pd.Timestamp.today().date()
        if timeframe == "1W": dmin = today - pd.Timedelta(days=7)
        elif timeframe == "1M": dmin = today - pd.Timedelta(days=30)
        elif timeframe == "6M": dmin = today - pd.Timedelta(days=180)
        elif timeframe == "YTD": dmin = pd.Timestamp(today.year,1,1).date()
        else: dmin = today - pd.Timedelta(days=365)
        dmax = today
    else:
        dmin, dmax = st.date_input("Custom date range", [pd.Timestamp.today().date()-pd.Timedelta(days=180), pd.Timestamp.today().date()])
    st.markdown("---")
    st.markdown("### Display Indicators")
    show_sma  = st.checkbox("SMA20", True)
    show_macd = st.checkbox("MACD", True)
    show_rsi  = st.checkbox("RSI", True)
    show_bb   = st.checkbox("Bollinger", True)
    st.markdown("---")
    ta_pct = st.slider("Technical Weight (%)", 0, 100, 60, 5)
    sa_pct = 100 - ta_pct
    stock = st.selectbox("Select Ticker", ["NVDA","AMD","ADBE","VRTX","SCHW","CROX","DE","FANG","TMUS","PLTR"])
    st.markdown(f"**Blend:** {ta_pct}% Tech + {sa_pct}% Sentiment")
    st.markdown("> Data via **YFinance**, **AlphaVantage**, **PushShift**")

# ─── Load Data ─────────────────────────────────────
sent_df  = pd.read_csv("reddit_sentiments.csv")
tech_raw = pd.read_csv("technical_indicators.csv", header=[0,1], index_col=0, parse_dates=True)
blend    = pd.read_csv("final_recommendations.csv", index_col=0)
fund_df  = pd.read_csv("fundamentals.csv", index_col="ticker")

# ─── Data Access ───────────────────────────────────
def series(tkr, col):
    df = tech_raw[tkr][col]
    return df.loc[dmin:dmax]

@st.cache_data(ttl=300)
def fetch_intraday(tkr):
    with st.spinner("Fetching intraday data…"):
        today = pd.Timestamp.today().date()
        return yf.download(tkr, start=today, end=today+pd.Timedelta(days=1), interval="5m", progress=False)

# ─── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏁 Verdict","🔧 Tech Breakdown","📊 Fundamentals & Pulse"])

def show_verdict():
    st.markdown("<h2>Overall Investment Verdict</h2>", unsafe_allow_html=True)
    verdict = blend.at[stock,"verdict"]
    color = {"BUY":"#0ff","HOLD":"#ff0","SELL":"#f0f"}[verdict]
    st.markdown(f"<h1 style='color:{color}'>{verdict}</h1>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Tech Score", blend.at[stock,"technical_score"] )
    c2.metric("Sentiment", round(blend.at[stock,"sentiment_score"],2) )
    c3.metric("Blended", round(blend.at[stock,"blended_score"],2) )
    st.caption(f"BUY > +2, SELL < –2 | Blend {ta_pct}% / {sa_pct}%")
    st.download_button("Download Verdict CSV", blend.loc[[stock]].to_csv().encode(), f"{stock}_verdict.csv")

def show_tech():
    st.markdown("<h2>Technical Indicators Breakdown</h2>", unsafe_allow_html=True)
    price = series(stock,"close"); fig = go.Figure(layout=dict(template='plotly_dark', height=300))
    fig.add_trace(go.Scatter(x=price.index,y=price,name="Price",line=dict(color="#0ff")))
    if show_sma:
        sma=series(stock,"SMA20")
        fig.add_trace(go.Scatter(x=sma.index,y=sma,name="SMA20",line=dict(color="#f0f",dash='dash')))
    if show_bb:
        up=series(stock,"BB_upper"); lo=series(stock,"BB_lower")
        fig.add_trace(go.Scatter(x=up.index,y=up,name="BB Upper",line=dict(color="#08f",dash='dot')))
        fig.add_trace(go.Scatter(x=lo.index,y=lo,name="BB Lower",line=dict(color="#08f",dash='dot')))
    fig.update_traces(hovertemplate="%{y:.2f} @ %{x|%Y-%m-%d}<extra></extra>")
    st.plotly_chart(fig,use_container_width=True)
    if show_macd:
        macd=series(stock,"MACD")
        st.markdown("#### MACD")
        fig2=px.line(macd,template="plotly_dark"); fig2.update_traces(hovertemplate="%{y:.2f} @ %{x}<extra></extra>")
        st.plotly_chart(fig2,use_container_width=True)
    if show_rsi:
        rsi=series(stock,"RSI")
        st.markdown("#### RSI")
        fig3=px.line(rsi,template="plotly_dark"); fig3.update_traces(hovertemplate="%{y:.1f} @ %{x}<extra></extra>")
        st.plotly_chart(fig3,use_container_width=True)

def show_fund():
    st.markdown("<h2>Fundamental Metrics & Community Pulse</h2>", unsafe_allow_html=True)
    if stock in fund_df.index:
        m=fund_df.loc[stock,["pe","de_ratio","ev_ebitda"]]
        m.index=["P/E","D/E","EV/EBITDA"]
        dfm=m.reset_index().rename(columns={"index":"Metric",stock:"Value"})
        fig4=px.bar(dfm,x="Metric",y="Value",text="Value",template="plotly_dark")
        st.plotly_chart(fig4,use_container_width=True)
        st.markdown("*Lower values suggest potential undervaluation.*")
    posts=sent_df[sent_df["ticker"]==stock]
    st.markdown("#### Community Pulse (Reddit)")
    if not posts.empty:
        st.dataframe(posts["title"].head(10),use_container_width=True)
    else: st.info("No recent Reddit mentions found.")

# Run all
with tab1: show_verdict()
with tab2: show_tech()
with tab3: show_fund()


