# app.py
import sys
from pathlib import Path
import os
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---- Page Config (must be first Streamlit call) ----
st.set_page_config(
    page_title="NSE Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- TensorFlow check ----
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ---- Import models conditionally ----
if TF_AVAILABLE:
    from models.lstm_model import train_model, predict_next_days
else:
    from models.simple_model import train_simple_model as train_model, predict_simple as predict_next_days

# ---- Import everything else ----
from data.fetcher import get_stock_data, get_live_price
from analysis.technical import add_technical_indicators, get_signal_summary
from analysis.sentiment import analyze_sentiment
from ai.insights import generate_insights

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📈 NSE Stock Predictor")
st.sidebar.markdown(
    "Powered by LSTM + Groq AI" if TF_AVAILABLE else "Powered by GradientBoost + Groq AI"
)

# ---- Load full NSE stock list ----
@st.cache_data(ttl=86400)
def load_nse_symbols():
    import requests
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_nse = pd.read_csv(url)
        stocks = {}
        for _, row in df_nse.iterrows():
            name = str(row['NAME OF COMPANY']).strip().title()
            sym = str(row['SYMBOL']).strip() + '.NS'
            stocks[f"{name} ({sym})"] = sym
        return stocks
    except Exception:
        return {
            'Reliance Industries (RELIANCE.NS)': 'RELIANCE.NS',
            'TCS (TCS.NS)': 'TCS.NS',
            'HDFC Bank (HDFCBANK.NS)': 'HDFCBANK.NS',
            'Infosys (INFY.NS)': 'INFY.NS',
            'Wipro (WIPRO.NS)': 'WIPRO.NS',
            'ICICI Bank (ICICIBANK.NS)': 'ICICIBANK.NS',
            'SBI (SBIN.NS)': 'SBIN.NS',
            'Bajaj Finance (BAJFINANCE.NS)': 'BAJFINANCE.NS',
            'Tata Motors (TATAMOTORS.NS)': 'TATAMOTORS.NS',
            'Adani Enterprises (ADANIENT.NS)': 'ADANIENT.NS',
            'Zomato (ZOMATO.NS)': 'ZOMATO.NS',
            'Paytm (PAYTM.NS)': 'PAYTM.NS',
            'Nykaa (NYKAA.NS)': 'NYKAA.NS',
        }

# ---- Stock search ----
with st.sidebar:
    st.markdown("### 🔍 Search Any NSE Stock")
    all_stocks = load_nse_symbols()
    symbol_to_name = {v: k for k, v in all_stocks.items()}

    search_query = st.text_input(
        "Type company name or symbol",
        placeholder="e.g. Zomato, HDFC, Tata..."
    ).strip().lower()

    if search_query:
        filtered = {k: v for k, v in all_stocks.items() if search_query in k.lower()}
    else:
        popular = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS',
                   'WIPRO.NS','ICICIBANK.NS','SBIN.NS','ZOMATO.NS',
                   'TATAMOTORS.NS','BAJFINANCE.NS']
        filtered = {k: v for k, v in all_stocks.items() if v in popular}

    if not filtered:
        st.warning("No stocks found. Try a different search.")
        filtered = all_stocks

    selected_name = st.selectbox(
        f"Select ({len(filtered)} results)",
        list(filtered.keys())
    )
    symbol = filtered[selected_name]

    st.markdown("**Or enter symbol directly:**")
    custom_symbol = st.text_input(
        "NSE/BSE symbol",
        placeholder="e.g. ZOMATO.NS or ZOMATO.BO"
    ).strip().upper()

    if custom_symbol:
        if not custom_symbol.endswith('.NS') and not custom_symbol.endswith('.BO'):
            custom_symbol += '.NS'
        symbol = custom_symbol

    st.markdown(f"**Selected:** `{symbol}`")
    st.markdown("---")

    period = st.slider("Historical data (days)", 90, 730, 365)
    forecast_days = st.slider("Forecast days ahead", 1, 14, 5)

    # ---- Auto train if no model saved ----
    model_path = f"models/saved/{symbol.replace('.', '_')}.keras"
    model_exists = os.path.exists(model_path)
    train_model_btn = st.button(
        "🔁 Retrain Model" if model_exists else "🧠 Train Model (First Time)"
    )

    st.markdown("---")
    st.markdown("⚠️ For educational use only. Not financial advice.")

if not TF_AVAILABLE:
    st.sidebar.warning("⚠️ Running lightweight model — LSTM unavailable on cloud")

# ============================================================
# LOAD DATA
# ============================================================
st.title(f"📊 {symbol_to_name.get(symbol, symbol)}")
st.markdown(f"*Analysis as of {datetime.now().strftime('%d %b %Y, %I:%M %p')}*")

with st.spinner("Fetching live NSE data..."):
    df_raw = get_stock_data(symbol, period_days=period)

if df_raw is None or df_raw.empty:
    st.error(f"Could not fetch data for {symbol}. Check the symbol and try again.")
    st.stop()

df = add_technical_indicators(df_raw)
live = get_live_price(symbol)

# ============================================================
# LIVE PRICE ROW
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price", f"₹{live['current_price']}")
col2.metric("Day High",      f"₹{live['day_high']}")
col3.metric("Day Low",       f"₹{live['day_low']}")
col4.metric("Prev Close",    f"₹{live['previous_close']}")
col5.metric("P/E Ratio",     live['pe_ratio'])

st.markdown("---")

# ============================================================
# PRICE CHART
# ============================================================
st.subheader("📈 Price Chart with Indicators")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='Price'
))
fig.add_trace(go.Scatter(
    x=df.index, y=df['EMA_20'], name='EMA 20',
    line=dict(color='orange', width=1.5)
))
fig.add_trace(go.Scatter(
    x=df.index, y=df['EMA_50'], name='EMA 50',
    line=dict(color='blue', width=1.5)
))
fig.add_trace(go.Scatter(
    x=df.index, y=df['BB_upper'], name='BB Upper',
    line=dict(color='gray', dash='dash', width=1)
))
fig.add_trace(go.Scatter(
    x=df.index, y=df['BB_lower'], name='BB Lower',
    line=dict(color='gray', dash='dash', width=1),
    fill='tonexty', fillcolor='rgba(200,200,200,0.1)'
))
fig.update_layout(height=500, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TECHNICAL SIGNALS + SENTIMENT
# ============================================================
col_tech, col_sent = st.columns(2)

with col_tech:
    st.subheader("🚦 Technical Signals")
    signals = get_signal_summary(df)

    signal_emoji = {
        'STRONG BUY': '🟢', 'BUY': '🟩',
        'HOLD': '🟡', 'SELL': '🟧', 'STRONG SELL': '🔴'
    }
    emoji = signal_emoji.get(signals['overall_signal'], '⚪')
    st.markdown(f"### {emoji} {signals['overall_signal']}")

    metrics_df = pd.DataFrame({
        'Indicator': ['RSI', 'MACD', 'Bollinger', 'vs EMA20', 'vs EMA50'],
        'Value': [
            f"{signals['RSI']} ({signals['RSI_signal']})",
            signals['MACD_direction'],
            signals['BB_signal'],
            signals['price_vs_EMA20'],
            signals['price_vs_EMA50'],
        ]
    })
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    fig_rsi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=signals['RSI'],
        title={'text': "RSI"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30],  'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100],'color': "salmon"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'value': signals['RSI']
            }
        }
    ))
    fig_rsi.update_layout(height=250)
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_sent:
    st.subheader("📰 News Sentiment")
    with st.spinner("Analyzing news..."):
        sentiment = analyze_sentiment(symbol)

    score = sentiment['sentiment_score']
    label = sentiment['sentiment_label']
    color = "green" if score > 0.05 else "red" if score < -0.05 else "gray"

    st.markdown(f"### Sentiment: :{color}[{label}] ({score:+.3f})")
    st.markdown(f"*{sentiment['articles_analyzed']} articles analyzed*")

    s1, s2, s3 = st.columns(3)
    s1.metric("Positive", sentiment['positive_count'])
    s2.metric("Neutral",  sentiment['neutral_count'])
    s3.metric("Negative", sentiment['negative_count'])

    if sentiment['headlines']:
        st.markdown("**Top Headlines:**")
        for h in sentiment['headlines'][:5]:
            icon = "🟢" if h['score'] > 0.05 else "🔴" if h['score'] < -0.05 else "⚪"
            st.markdown(f"{icon} {h['title'][:90]}...")

# ============================================================
# MODEL TRAINING + PREDICTION
# ============================================================
st.markdown("---")
st.subheader("🔮 Price Prediction")

# Auto train on first load if no model exists
if not model_exists and f'model_{symbol}' not in st.session_state:
    with st.spinner(f"Auto-training model for {symbol} for the first time... (~2 mins)"):
        model, scaler, features, metrics = train_model(symbol, df, epochs=50)
        st.session_state[f'model_{symbol}'] = (model, scaler, features)
        st.success(f"✅ Model ready! Accuracy: ~{metrics['accuracy']:.1f}% | MAE: ₹{metrics['mae']:.2f}")

elif train_model_btn:
    with st.spinner(f"Retraining model for {symbol}..."):
        model, scaler, features, metrics = train_model(symbol, df, epochs=50)
        st.session_state[f'model_{symbol}'] = (model, scaler, features)
        st.success(f"✅ Retrained! Accuracy: ~{metrics['accuracy']:.1f}% | MAE: ₹{metrics['mae']:.2f}")

if f'model_{symbol}' in st.session_state:
    model, scaler, features = st.session_state[f'model_{symbol}']
    prediction = predict_next_days(symbol, df, scaler, features, days_ahead=forecast_days)

    if prediction:
        trend_col, pred_col = st.columns([1, 2])

        with trend_col:
            st.metric("Predicted Trend",  prediction['predicted_trend'])
            st.metric("Expected Change",  f"{prediction['expected_change_pct']:+.2f}%")
            st.metric("Current Price",    f"₹{prediction['current_price']}")

        with pred_col:
            pred_df = pd.DataFrame({
                'Day': [f"Day {i+1}" for i in range(len(prediction['predictions']))],
                'Predicted Price (₹)': prediction['predictions']
            })
            fig_pred = px.line(
                pred_df, x='Day', y='Predicted Price (₹)',
                markers=True,
                title=f"{symbol} — {forecast_days}-Day Forecast"
            )
            fig_pred.update_traces(line_color='#00CC96', line_width=2.5)
            st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("👆 Model will auto-train when you select a stock for the first time.")

# ============================================================
# AI INSIGHTS
# ============================================================
st.markdown("---")
st.subheader("🧠 AI Analysis (Groq)")

if st.button("✨ Generate AI Insights"):
    pred_result = None
    if f'model_{symbol}' in st.session_state:
        model, scaler, features = st.session_state[f'model_{symbol}']
        pred_result = predict_next_days(symbol, df, scaler, features, days_ahead=5)

    with st.spinner("AI is analyzing all signals..."):
        insight = generate_insights(symbol, signals, sentiment, pred_result)

    st.markdown(f"""
    <div style="background:#f0f4ff;padding:20px;border-radius:12px;border-left:4px solid #4B6BFB;">
    {insight.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# BACKTESTING
# ============================================================
st.markdown("---")
st.subheader("🎯 Model Accuracy & Backtesting")

if f'model_{symbol}' in st.session_state:
    from utils.backtester import (
        run_backtest, plot_backtest,
        plot_error_distribution, plot_directional_accuracy
    )

    if st.button("📊 Run Full Backtest"):
        model, scaler, features = st.session_state[f'model_{symbol}']

        with st.spinner("Running backtest on historical data..."):
            results = run_backtest(df, scaler, features, model)

        m = results['metrics']
        st.markdown("### 📐 Performance Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Directional Accuracy", f"{m['Directional Accuracy']}%",
                  help="How often it predicted UP/DOWN correctly")
        c2.metric("Within 2% Error",      f"{m['Within 2%']}%",
                  help="Predictions within 2% of actual price")
        c3.metric("Within 5% Error",      f"{m['Within 5%']}%",
                  help="Predictions within 5% of actual price")
        c4.metric("Correlation",          f"{m['Correlation']}",
                  help="How closely predicted tracks actual (1.0 = perfect)")

        c5, c6, c7 = st.columns(3)
        c5.metric("MAE",  f"₹{m['MAE']}",  help="Mean Absolute Error in rupees")
        c6.metric("RMSE", f"₹{m['RMSE']}", help="Root Mean Square Error")
        c7.metric("MAPE", f"{m['MAPE']}%", help="Mean Absolute Percentage Error")

        dir_acc = m['Directional Accuracy']
        grade = (
            "🟢 Excellent" if dir_acc >= 65 else
            "🟡 Good"      if dir_acc >= 55 else
            "🟠 Average"   if dir_acc >= 50 else
            "🔴 Needs more training data"
        )

        st.info(f"""
        **Directional Accuracy: {dir_acc}% — {grade}**
        - Above 50% = better than random guessing
        - Above 60% = good for stock prediction
        - Above 65% = excellent — hedge funds aim for this

        **MAE of ₹{m['MAE']}** means predictions are on average ₹{m['MAE']} away from actual price.
        **Within 2%: {m['Within 2%']}%** of predictions were very close to actual price.
        """)

        st.plotly_chart(plot_backtest(results, symbol), use_container_width=True)

        col_err, col_dir = st.columns(2)
        with col_err:
            st.plotly_chart(plot_error_distribution(results),    use_container_width=True)
        with col_dir:
            st.plotly_chart(plot_directional_accuracy(results),  use_container_width=True)

        with st.expander("📋 View raw backtest data"):
            st.dataframe(
                results['results_df'].style.format({
                    'Actual':    '₹{:.2f}',
                    'Predicted': '₹{:.2f}',
                    'Error':     '₹{:.2f}',
                    'Error_Pct': '{:.2f}%'
                }),
                use_container_width=True
            )
else:
    st.info("👆 Train the model first using the sidebar button, then run the backtest.")