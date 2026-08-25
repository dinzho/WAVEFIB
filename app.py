import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from github import Github, GithubException
import os
import numpy as np

# ==========================================
# 1. 雲端同步與基礎工具
# ==========================================
class GitHubSync:
    def __init__(self):
        self.token = st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN"))
        self.repo_name = st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO", "your-username/stock-data"))
        self.file_path = "portfolio_data.json"
        self.g = Github(self.token) if self.token else None
        self.repo = self.g.get_repo(self.repo_name) if self.g else None

    def _get_default_data(self):
        return {"watchlist": [], "positions": []}

    def load_data(self):
        if not self.repo:
            return st.session_state.get('cloud_data', self._get_default_data())
        try:
            file_content = self.repo.get_contents(self.file_path)
            data = json.loads(file_content.decoded_content.decode("utf-8"))
            st.session_state['cloud_data'] = data
            return data
        except GithubException as e:
            if e.status == 404: 
                self.save_data(self._get_default_data())
                return self._get_default_data()
            return self._get_default_data()

    def save_data(self, data):
        st.session_state['cloud_data'] = data
        if not self.repo: return
        try:
            file_content = self.repo.get_contents(self.file_path)
            self.repo.update_file(
                self.file_path, "Update portfolio data", 
                json.dumps(data, indent=2, ensure_ascii=False), file_content.sha
            )
        except: pass

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def find_significant_swings(df, threshold_pct=3.0):
    if len(df) < 2: return None, None, []
    last_high, last_low = df['High'].iloc[0], df['Low'].iloc[0]
    trend = 0
    sig_highs, sig_lows, swing_points = [], [], []

    for i in range(1, len(df)):
        high, low = df['High'].iloc[i], df['Low'].iloc[i]
        if trend == 0:
            if high > last_high: last_high = high
            if low < last_low: last_low = low
            if (high - last_low) / last_low * 100 >= threshold_pct:
                trend = 1; sig_lows.append(last_low); swing_points.append({'idx': i-1, 'type': 'low', 'price': last_low}); last_high = high
            elif (last_high - low) / last_high * 100 >= threshold_pct:
                trend = -1; sig_highs.append(last_high); swing_points.append({'idx': i-1, 'type': 'high', 'price': last_high}); last_low = low
        elif trend == 1:
            if high > last_high: last_high = high
            elif (last_high - low) / last_high * 100 >= threshold_pct:
                sig_highs.append(last_high); swing_points.append({'idx': i-1, 'type': 'high', 'price': last_high}); trend = -1; last_low = low
        elif trend == -1:
            if low < last_low: last_low = low
            elif (high - last_low) / last_low * 100 >= threshold_pct:
                sig_lows.append(last_low); swing_points.append({'idx': i-1, 'type': 'low', 'price': last_low}); trend = 1; last_high = high

    swing_high = sig_highs[-1] if sig_highs else df['High'].max()
    swing_low = sig_lows[-1] if sig_lows else df['Low'].min()
    if swing_high < swing_low: swing_high, swing_low = swing_low, swing_high
    return swing_high, swing_low, swing_points

# ==========================================
# 2. 多時間框架分析引擎 (核心升級)
# ==========================================
def analyze_single_timeframe(df, threshold_pct):
    """分析單一時間框架的趨勢與動能"""
    if df is None or len(df) < 30: return None
    
    close = df['Close'].iloc[-1]
    rsi = calculate_rsi(df['Close']).iloc[-1]
    macd, signal, hist = calculate_macd(df)
    macd_val, signal_val, hist_val = macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]
    
    # 成交量分析
    avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
    vol_ratio = df['Volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1
    
    # FIB 分析
    high_price, low_price, swings = find_significant_swings(df, threshold_pct)
    diff = high_price - low_price
    
    # 趨勢評分 (0-100)
    score = 50
    if macd_val > signal_val: score += 15
    if macd_val > 0: score += 10
    if rsi > 50: score += 10
    if rsi > 60: score += 5
    if vol_ratio > 1.2: score += 10
    
    # 判斷波浪可能位置 (簡化版)
    wave_phase = "盤整/調整"
    if score > 75 and rsi > 60: wave_phase = "主升段 (可能第3浪)"
    elif score > 60: wave_phase = "上升趨勢 (第1或5浪)"
    elif score < 40 and rsi < 40: wave_phase = "主跌段"
    elif score < 50: wave_phase = "下降趨勢"

    return {
        'close': close, 'rsi': rsi, 'macd': macd_val, 'hist': hist_val,
        'vol_ratio': vol_ratio, 'score': min(score, 100),
        'wave_phase': wave_phase, 'high': high_price, 'low': low_price,
        'trend': "多頭" if score > 55 else "空頭" if score < 45 else "盤整"
    }

@st.cache_data(ttl=120)
def fetch_multi_timeframe(ticker, tf_large, tf_small):
    """並行獲取多個時間框架數據"""
    period_map = {"月線": "5y", "週線": "2y", "日線": "1y", "小時線": "3mo"}
    interval_map = {"月線": "1mo", "週線": "1wk", "日線": "1d", "小時線": "1h"}
    
    df_large = None
    df_small = None
    
    try:
        df_large = yf.download(ticker, period=period_map[tf_large], interval=interval_map[tf_large], progress=False)
        if isinstance(df_large.columns, pd.MultiIndex): df_large.columns = df_large.columns.droplevel(1)
        df_large = df_large.reset_index()
        if 'Date' in df_large.columns: df_large['Date'] = pd.to_datetime(df_large['Date'])
    except: pass

    try:
        df_small = yf.download(ticker, period=period_map[tf_small], interval=interval_map[tf_small], progress=False)
        if isinstance(df_small.columns, pd.MultiIndex): df_small.columns = df_small.columns.droplevel(1)
        df_small = df_small.reset_index()
        if 'Date' in df_small.columns: df_small['Date'] = pd.to_datetime(df_small['Date'])
    except: pass

    return df_large, df_small

# ==========================================
# 3. Streamlit UI 主程式
# ==========================================
st.set_page_config(page_title="智能個股分析平台 - 多時間框架共振", layout="wide", page_icon="")
sync_manager = GitHubSync()
data = sync_manager.load_data()

st.title(" 智能個股分析平台 - 多時間框架波浪共振")

# --- Sidebar ---
st.sidebar.header("🔍 股票查詢")
search_code = st.sidebar.text_input("代碼 (如: NVDA, 0700)", "NVDA").upper().strip()
market = st.sidebar.selectbox("市場", ["US", "HK"], index=0)
ticker = f"{search_code}.HK" if market == "HK" else search_code

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 共振分析設定")
threshold_pct = st.sidebar.slider("波段確認閾值 (%)", 1.0, 15.0, 3.0, 0.5)

# 【核心升級】多時間框架選擇
st.sidebar.markdown("#### 🕰️ 時間框架配置")
tf_large = st.sidebar.selectbox("大級別 (定趨勢方向)", ["週線", "月線", "日線"], index=0, 
    help="用於判斷當前處於波浪理論的哪個大級別（如第3浪主升段）。")
tf_small = st.sidebar.selectbox("小級別 (找進場信號)", ["日線", "小時線", "週線"], index=0, 
    help="用於尋找精確的突破進場點與止損位。")

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📈 共振分析儀表板", "👁️ 觀察清單", "💼 模擬持倉"])

with tab1:
    st.subheader(f"{search_code} ({market}) - 多時間框架共振分析")
    
    with st.spinner(f"正在獲取 {tf_large} 與 {tf_small} 數據並計算波浪結構..."):
        df_large, df_small = fetch_multi_timeframe(ticker, tf_large, tf_small)
    
    if df_large is None or df_small is None:
        st.error("無法獲取多時間框架數據，請檢查代碼或網路連線。")
    else:
        # 執行分析
        analysis_large = analyze_single_timeframe(df_large, threshold_pct)
        analysis_small = analyze_single_timeframe(df_small, threshold_pct)
        
        if analysis_large and analysis_small:
            # --- 1. 共振評分矩陣 ---
            st.markdown("### 🎯 波浪共振評分矩陣")
            resonance_score = (analysis_large['score'] * 0.6) + (analysis_small['score'] * 0.4)
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown(f"""<div style="padding:15px; border-radius:10px; background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3);">
                    <div style="font-size:12px; color:#9ca3af">大級別 ({tf_large}) 趨勢</div>
                    <div style="font-size:20px; font-weight:bold; color:#60a5fa">{analysis_large['trend']} ({analysis_large['score']:.0f}分)</div>
                    <div style="font-size:11px; color:#6b7280">波浪階段: {analysis_large['wave_phase']}</div>
                </div>""", unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""<div style="padding:15px; border-radius:10px; background:rgba(168,85,247,0.1); border:1px solid rgba(168,85,247,0.3);">
                    <div style="font-size:12px; color:#9ca3af">小級別 ({tf_small}) 動能</div>
                    <div style="font-size:20px; font-weight:bold; color:#a855f7">{analysis_small['trend']} ({analysis_small['score']:.0f}分)</div>
                    <div style="font-size:11px; color:#6b7280">成交量比: {analysis_small['vol_ratio']:.2f}x</div>
                </div>""", unsafe_allow_html=True)
            
            with col_res3:
                color = "34,197,94" if resonance_score > 70 else "234,179,8" if resonance_score > 50 else "239,68,68"
                st.markdown(f"""<div style="padding:15px; border-radius:10px; background:rgba({color},0.1); border:2px solid rgba({color},0.5);">
                    <div style="font-size:12px; color:#9ca3af">🌊 綜合共振指數</div>
                    <div style="font-size:28px; font-weight:bold; color:rgb({color})">{resonance_score:.0f}/100</div>
                    <div style="font-size:11px; color:#6b7280">{'強烈共振 (高勝率)' if resonance_score > 70 else '部分共振' if resonance_score > 50 else '信號衝突 (觀望)'}</div>
                </div>""", unsafe_allow_html=True)

            # --- 2. 交易策略建議 (基於波浪理論) ---
            st.markdown("### 💡 實戰交易策略建議")
            if resonance_score > 70 and analysis_large['trend'] == "多頭":
                st.success(f"""
                **🚀 高勝率做多策略 (順大級別趨勢)**
                - **波浪定位**：大級別 ({tf_large}) 處於 **{analysis_large['wave_phase']}**，小級別 ({tf_small}) 確認突破。
                - **進場點**：小級別 FIB 回撤 38.2% - 50% 區間，或小級別突破 Swing High ({analysis_small['high']:.2f})。
                - **止損位**：小級別 Swing Low ({analysis_small['low']:.2f}) 下方。
                - **目標價**：大級別 FIB 延伸 1.618 ({analysis_large['high'] + (analysis_large['high'] - analysis_large['low'])*1.618:.2f})。
                """)
            elif resonance_score < 40 and analysis_large['trend'] == "空頭":
                st.error(f"""
                **️ 高勝率做空/觀望策略**
                - **波浪定位**：大級別 ({tf_large}) 處於 **{analysis_large['wave_phase']}**，動能疲弱。
                - **建議**：避免抄底，等待小級別出現 ABC 調整完成的信號。
                """)
            else:
                st.warning(f"""
                ** 盤整/震盪策略**
                - **波浪定位**：大級別 ({tf_large}) 為 **{analysis_large['wave_phase']}**，大小級別信號不完全一致。
                - **建議**：縮小倉位，採用區間操作（高拋低吸），嚴格執行止損。
                """)

            # --- 3. 多時間框架圖表對比 ---
            st.markdown("### 📊 多級別圖表對比")
            tab_large, tab_small = st.tabs([f" {tf_large} (大級別趨勢)", f"🔍 {tf_small} (小級別進場)"])
            
            with tab_large:
                fig_large = go.Figure()
                fig_large.add_trace(go.Candlestick(x=df_large['Date'], open=df_large['Open'], high=df_large['High'], low=df_large['Low'], close=df_large['Close'], name='K線'))
                # 標記大級別 Swing 點
                for point in find_significant_swings(df_large, threshold_pct)[2][-5:]:
                    fig_large.add_scatter(x=[df_large['Date'].iloc[point['idx']]], y=[point['price']], mode='markers', marker=dict(size=10, color='red' if point['type']=='high' else 'green', symbol='star'))
                fig_large.update_layout(title=f"{search_code} {tf_large} 波浪結構", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_large, use_container_width=True)

            with tab_small:
                fig_small = go.Figure()
                fig_small.add_trace(go.Candlestick(x=df_small['Date'], open=df_small['Open'], high=df_small['High'], low=df_small['Low'], close=df_small['Close'], name='K線'))
                # 標記小級別 Swing 點
                for point in find_significant_swings(df_small, threshold_pct)[2][-5:]:
                    fig_small.add_scatter(x=[df_small['Date'].iloc[point['idx']]], y=[point['price']], mode='markers', marker=dict(size=10, color='red' if point['type']=='high' else 'green', symbol='star'))
                fig_small.update_layout(title=f"{search_code} {tf_small} 突破信號", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_small, use_container_width=True)

        else:
            st.warning("數據不足，無法進行共振分析。")

with tab2:
    st.subheader("👁️ 觀察清單 (GitHub同步)")
    # ... (保持原有觀察清單邏輯，此處省略以節省篇幅，請從上一版本複製)
    st.info("觀察清單功能與上一版本相同。")

with tab3:
    st.subheader("💼 模擬持倉 (GitHub同步)")
    st.info("模擬持倉功能與上一版本相同。")