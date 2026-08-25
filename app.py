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
# 1. GitHub 同步管理 (支援用戶自行配置)
# ==========================================
class GitHubSync:
    def __init__(self, token=None, repo_name=None):
        self.token = token or st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN"))
        self.repo_name = repo_name or st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO", ""))
        self.file_path = "portfolio_data.json"
        self.is_configured = False
        self.repo = None
        
        if self.token and self.repo_name:
            try:
                self.g = Github(self.token)
                self.repo = self.g.get_repo(self.repo_name)
                self.is_configured = True
            except:
                self.is_configured = False

    def _get_default_data(self):
        return {"watchlist": [], "positions": []}

    def load_data(self):
        if not self.is_configured:
            return st.session_state.get('app_data', self._get_default_data())
        
        try:
            file_content = self.repo.get_contents(self.file_path)
            data = json.loads(file_content.decoded_content.decode("utf-8"))
            st.session_state['app_data'] = data
            return data
        except GithubException as e:
            if e.status == 404:
                self.save_data(self._get_default_data())
                return self._get_default_data()
            return self._get_default_data()

    def save_data(self, data):
        st.session_state['app_data'] = data
        if not self.is_configured:
            return
        
        try:
            file_content = self.repo.get_contents(self.file_path)
            self.repo.update_file(
                self.file_path, "Update portfolio", 
                json.dumps(data, indent=2, ensure_ascii=False), file_content.sha
            )
        except: pass

# ==========================================
# 2. 技術分析核心函數 (增強穩定性)
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def find_swings(df, threshold_pct=3.0):
    """【修復 1】尋找顯著波段高低點 - 保證永遠返回有效數值"""
    if df is None or len(df) < 2:
        # 返回安全預設值
        return 100.0, 90.0, []
    
    # 確保數據類型正確
    df = df.copy()
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df = df.dropna(subset=['High', 'Low'])
    
    if len(df) < 2:
        return float(df['High'].max()) if len(df) > 0 else 100.0, float(df['Low'].min()) if len(df) > 0 else 90.0, []

    last_high = float(df['High'].iloc[0])
    last_low = float(df['Low'].iloc[0])
    trend = 0
    swings = []
    
    for i in range(1, len(df)):
        high = float(df['High'].iloc[i])
        low = float(df['Low'].iloc[i])
        
        if trend == 0:
            if high > last_high: last_high = high
            if low < last_low: last_low = low
            if last_low > 0 and (high - last_low) / last_low * 100 >= threshold_pct:
                trend = 1; swings.append({'idx': i-1, 'type': 'low', 'price': last_low}); last_high = high
            elif last_high > 0 and (last_high - low) / last_high * 100 >= threshold_pct:
                trend = -1; swings.append({'idx': i-1, 'type': 'high', 'price': last_high}); last_low = low
        elif trend == 1:
            if high > last_high: last_high = high
            elif last_high > 0 and (last_high - low) / last_high * 100 >= threshold_pct:
                swings.append({'idx': i-1, 'type': 'high', 'price': last_high}); trend = -1; last_low = low
        else:
            if low < last_low: last_low = low
            elif last_low > 0 and (high - last_low) / last_low * 100 >= threshold_pct:
                swings.append({'idx': i-1, 'type': 'low', 'price': last_low}); trend = 1; last_high = high
    
    # 【關鍵修復】永遠返回有效的 high 和 low
    high_price = float(df['High'].max())
    low_price = float(df['Low'].min())
    
    # 確保 high > low
    if high_price <= low_price:
        high_price = low_price * 1.05  # 給一個 5% 的緩衝
    
    return high_price, low_price, swings

def identify_wave_pattern(swings, df=None):
    """自動識別浪型"""
    if not swings or len(swings) < 2: 
        return "趨勢初期", "數據不足，建議切換到日線或降低閾值至 1-2%"
    
    last_swings = swings[-5:] if len(swings) >= 5 else swings
    highs = [s['price'] for s in last_swings if s['type'] == 'high']
    lows = [s['price'] for s in last_swings if s['type'] == 'low']
    
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "第3浪 (主升段)", "強勁上升趨勢，成交量配合，適合持倉或加倉"
        elif highs[-1] < highs[-2] and lows[-1] > lows[-2]:
            return "第5浪 (尾聲)", "上升動能減弱，注意背離，準備獲利了結"
        elif highs[-1] > highs[-2] and lows[-1] < lows[-2]:
            return "第2浪或第4浪 (回調)", "回調階段，關注FIB 38.2%-61.8%支撐位"
        else:
            return "調整浪 (ABC)", "盤整或下跌趨勢，觀望或輕倉操作"
    
    return "趨勢初期", "等待明確信號確認"

def analyze_dow_theory(df):
    """道氏理論分析"""
    if len(df) < 50: return "數據不足", "需要更多歷史數據"
    
    recent_highs = df['High'].rolling(window=20).max().iloc[-20:]
    recent_lows = df['Low'].rolling(window=20).min().iloc[-20:]
    
    higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[0]
    higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[0]
    
    if higher_highs and higher_lows:
        return "上升趨勢", "HH & HL 持續確認，主要趨勢向上"
    elif not higher_highs and not higher_lows:
        return "下降趨勢", "LL & LH 確認，主要趨勢向下"
    else:
        return "盤整趨勢", "高低點混亂，等待方向突破"

def calculate_fib_zones(high, low):
    """【修復 2】計算斐波那契區間 - 增加防禦性編程"""
    if high is None or low is None:
        high, low = 100.0, 90.0  # 安全預設值
    
    if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
        high, low = 100.0, 90.0
    
    if high == low or high <= 0 or low <= 0:
        # 避免除零錯誤
        high = max(high, low * 1.05)
    
    diff = abs(high - low)
    
    return {
        "阻力位": [high + diff * 0.618, high + diff * 1.0, high + diff * 1.618],
        "支撐位": [high - diff * 0.382, high - diff * 0.5, high - diff * 0.618],
        "關鍵位": [low, high],
        "high": high,
        "low": low
    }

def analyze_pattern(df):
    """形態與趨勢分析"""
    if len(df) < 30: return "數據不足", "中性"
    
    ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
    ma50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else ma20
    current_price = df['Close'].iloc[-1]
    
    if current_price > ma20 > ma50:
        return "多頭排列", "MA20 > MA50，趨勢強勁向上"
    elif current_price < ma20 < ma50:
        return "空頭排列", "MA20 < MA50，趨勢向下"
    else:
        return "均線糾纏", "均線交錯，等待方向選擇"

def generate_strategy(current_price, fib_zones, wave_type, trend):
    """生成三種情境策略"""
    high = fib_zones.get('high', current_price)
    low = fib_zones.get('low', current_price * 0.9)
    
    if "上升" in trend and ("第3浪" in wave_type or "第1浪" in wave_type):
        return {
            "樂觀": {"目標": high + (high-low) * 0.618, "概率": "30%", "策略": "突破前高後加倉，目標看1.618延伸"},
            "基準": {"目標": high, "概率": "50%", "策略": "持倉觀望，關注成交量變化"},
            "悲觀": {"目標": low, "概率": "20%", "策略": "跌破FIB 61.8%止損離場"}
        }
    elif "回調" in wave_type:
        return {
            "樂觀": {"目標": high - (high-low) * 0.382, "概率": "40%", "策略": "在38.2%支撐位接多，博反彈"},
            "基準": {"目標": high - (high-low) * 0.5, "概率": "40%", "策略": "在50%位置分批建倉"},
            "悲觀": {"目標": low, "概率": "20%", "策略": "跌破61.8%放棄做多，等待新信號"}
        }
    
    return {
        "樂觀": {"目標": current_price * 1.05, "概率": "20%", "策略": "輕倉試探"},
        "基準": {"目標": current_price, "概率": "50%", "策略": "觀望為主"},
        "悲觀": {"目標": current_price * 0.95, "概率": "30%", "策略": "跌破支撐止損"}
    }

# ==========================================
# 3. 數據獲取
# ==========================================
@st.cache_data(ttl=120)
def fetch_data(ticker, period="2y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        if 'Datetime' in df.columns: df = df.rename(columns={'Datetime': 'Date'})
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'])
        return df
    except: return None

# ==========================================
# 4. UI 主程式
# ==========================================
st.set_page_config(page_title="智能個股分析平台", layout="wide", page_icon="🌊")

# 【修復 3】GitHub 配置放在 Sidebar 讓用戶自行輸入
with st.sidebar:
    st.title("⚙️ 系統設定")
    
    st.markdown("### 🔑 GitHub 配置")
    st.markdown("*輸入 Token 和倉庫以啟用雲端同步*")
    
    github_token = st.text_input("GitHub Token", type="password", 
                                  help="在 GitHub Settings > Developer settings > Personal access tokens 生成")
    github_repo = st.text_input("GitHub 倉庫", placeholder="username/repo",
                                 help="格式: 你的用戶名/倉庫名")
    
    if github_token and github_repo:
        # 創建帶有用戶配置的 sync 對象
        sync = GitHubSync(token=github_token, repo_name=github_repo)
        if sync.is_configured:
            st.success("✅ GitHub 同步已啟用")
        else:
            st.error("❌ 配置失敗，請檢查 Token 和倉庫名稱")
    else:
        sync = GitHubSync()
        if sync.is_configured:
            st.success("✅ 使用環境變量配置")
        else:
            st.warning("⚠️ 未配置 GitHub，數據僅暫存")
    
    st.markdown("---")
    st.title("🔍 股票查詢")
    search_code = st.text_input("股票代碼", "NVDA").upper()
    market = st.selectbox("市場", ["US", "HK"])
    ticker = f"{search_code}.HK" if market == "HK" else search_code
    
    st.markdown("---")
    st.title("️ 分析參數")
    threshold = st.slider("波段閾值 (%)", 0.5, 15.0, 3.0, 0.5, 
                         help="小時線建議 1-2%，日線建議 3-5%")
    tf_large = st.selectbox("大級別", ["週線", "月線", "日線"])
    tf_small = st.selectbox("小級別", ["日線", "小時線", "週線"])

# 主標題
st.title("🌊 智能個股分析平台 - 波浪理論專業版")

# 加載數據
data = sync.load_data()
period_map = {"月線": "5y", "週線": "2y", "日線": "1y", "小時線": "3mo"}
interval_map = {"月線": "1mo", "週線": "1wk", "日線": "1d", "小時線": "1h"}

df_large = fetch_data(ticker, period_map[tf_large], interval_map[tf_large])
df_small = fetch_data(ticker, period_map[tf_small], interval_map[tf_small])

if df_large is None or df_small is None:
    st.error("無法獲取數據，請檢查代碼或嘗試切換時間框架。")
    st.stop()

# 當前價格
current_price = df_small['Close'].iloc[-1]
prev_close = df_small['Close'].iloc[-2] if len(df_small) > 1 else current_price
change = current_price - prev_close
change_pct = (change / prev_close) * 100

# 顯示指標卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("當前價格", f"{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
with col2:
    rsi = calculate_rsi(df_small['Close']).iloc[-1]
    st.metric("RSI (14)", f"{rsi:.1f}", "超買" if rsi > 70 else "超賣" if rsi < 30 else "中性")
with col3:
    macd, _, _ = calculate_macd(df_small)
    st.metric("MACD", f"{macd.iloc[-1]:.2f}", "多頭" if macd.iloc[-1] > 0 else "空頭")
with col4:
    if 'Volume' in df_small.columns:
        vol_ma = df_small['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df_small['Volume'].iloc[-1] / vol_ma if vol_ma > 0 else 1
        st.metric("成交量比", f"{vol_ratio:.2f}x", "放量" if vol_ratio > 1.5 else "正常")
    else:
        st.metric("成交量", "無數據")

# 技術分析
high_price, low_price, swings = find_swings(df_small, threshold)
wave_type, wave_desc = identify_wave_pattern(swings, df_small)
dow_trend, dow_desc = analyze_dow_theory(df_small)
pattern, pattern_desc = analyze_pattern(df_small)
fib_zones = calculate_fib_zones(high_price, low_price)
strategies = generate_strategy(current_price, fib_zones, wave_type, dow_trend)

# 標籤頁
tab1, tab2, tab3 = st.tabs([" 共振分析儀表板", "️ 觀察清單", "💼 模擬持倉"])

with tab1:
    # 道氏理論 + 波浪定位
    st.markdown("### 📐 道氏理論與波浪分析")
    col_dow1, col_dow2 = st.columns(2)
    with col_dow1:
        st.markdown(f"""<div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(147,51,234,0.1)); border: 2px solid rgba(59,130,246,0.4); box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #60a5fa; margin: 0 0 10px 0;">📊 趨勢方向</h3>
            <p style="font-size: 26px; font-weight: bold; color: #60a5fa; margin: 10px 0;">{dow_trend}</p>
            <p style="color: #9ca3af; margin: 0; font-size: 14px;">{dow_desc}</p></div>""", unsafe_allow_html=True)
    with col_dow2:
        st.markdown(f"""<div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(168,85,247,0.1)); border: 2px solid rgba(34,197,94,0.4); box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #22c55e; margin: 0 0 10px 0;">🎯 波浪定位</h3>
            <p style="font-size: 26px; font-weight: bold; color: #22c55e; margin: 10px 0;">{wave_type}</p>
            <p style="color: #9ca3af; margin: 0; font-size: 14px;">{wave_desc}</p></div>""", unsafe_allow_html=True)
    
    # 【修復 4】斐波那契區間 - 美化 UI
    st.markdown("### 📐 斐波那契區間")
    col_fib1, col_fib2, col_fib3 = st.columns(3)
    
    with col_fib1:
        st.markdown("""<div style="background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05)); padding: 15px; border-radius: 10px; border-left: 4px solid #ef4444;">
            <h4 style="color: #ef4444; margin: 0 0 10px 0;">📈 阻力位</h4>
        </div>""", unsafe_allow_html=True)
        st.metric("1.618 延伸", f"{fib_zones['阻力位'][2]:.2f}")
        st.metric("1.000 等長", f"{fib_zones['阻力位'][1]:.2f}")
        st.metric("0.618 阻力", f"{fib_zones['阻力位'][0]:.2f}")
    
    with col_fib2:
        st.markdown("""<div style="background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05)); padding: 15px; border-radius: 10px; border-left: 4px solid #22c55e;">
            <h4 style="color: #22c55e; margin: 0 0 10px 0;"> 支撐位</h4>
        </div>""", unsafe_allow_html=True)
        st.metric("38.2% 支撐", f"{fib_zones['支撐位'][0]:.2f}")
        st.metric("50.0% 中軸", f"{fib_zones['支撐位'][1]:.2f}")
        st.metric("61.8% 強支撐", f"{fib_zones['支撐位'][2]:.2f}")
    
    with col_fib3:
        st.markdown("""<div style="background: linear-gradient(135deg, rgba(234,179,8,0.1), rgba(234,179,8,0.05)); padding: 15px; border-radius: 10px; border-left: 4px solid #eab308;">
            <h4 style="color: #eab308; margin: 0 0 10px 0;">🎯 關鍵位</h4>
        </div>""", unsafe_allow_html=True)
        st.metric("波段高點", f"{fib_zones['關鍵位'][1]:.2f}")
        st.metric("波段低點", f"{fib_zones['關鍵位'][0]:.2f}")
        st.metric("波動幅度", f"{fib_zones['high'] - fib_zones['low']:.2f}")
    
    # 形態分析
    st.markdown("### 📊 形態與趨勢分析")
    st.markdown(f"""<div style="padding: 15px; border-radius: 10px; background: linear-gradient(135deg, rgba(234,179,8,0.1), rgba(234,179,8,0.05)); border-left: 4px solid #eab308;">
        <p style="margin: 0; font-size: 16px;"><strong>當前形態:</strong> <span style="color: #eab308; font-weight: bold;">{pattern}</span></p>
        <p style="margin: 5px 0 0 0; color: #9ca3af; font-size: 14px;">{pattern_desc}</p></div>""", unsafe_allow_html=True)
    
    # 三種情境策略
    st.markdown("### 🎯 建議策略與三種情況點位")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown(f"""<div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.05)); border: 2px solid rgba(34,197,94,0.5); box-shadow: 0 4px 6px rgba(34,197,94,0.2);">
            <h3 style="color: #22c55e; margin: 0;">🚀 樂觀情境</h3>
            <p style="font-size: 32px; font-weight: bold; color: #22c55e; margin: 15px 0;">{strategies['樂觀']['目標']:.2f}</p>
            <div style="background: rgba(34,197,94,0.2); padding: 8px; border-radius: 6px; margin: 10px 0;">
                <p style="color: #9ca3af; margin: 0; font-size: 13px;">概率: {strategies['樂觀']['概率']}</p>
            </div>
            <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 13px; line-height: 1.5;">{strategies['樂觀']['策略']}</p></div>""", unsafe_allow_html=True)
    
    with col_s2:
        st.markdown(f"""<div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(59,130,246,0.05)); border: 2px solid rgba(59,130,246,0.5); box-shadow: 0 4px 6px rgba(59,130,246,0.2);">
            <h3 style="color: #60a5fa; margin: 0;">📊 基準情境</h3>
            <p style="font-size: 32px; font-weight: bold; color: #60a5fa; margin: 15px 0;">{strategies['基準']['目標']:.2f}</p>
            <div style="background: rgba(59,130,246,0.2); padding: 8px; border-radius: 6px; margin: 10px 0;">
                <p style="color: #9ca3af; margin: 0; font-size: 13px;">概率: {strategies['基準']['概率']}</p>
            </div>
            <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 13px; line-height: 1.5;">{strategies['基準']['策略']}</p></div>""", unsafe_allow_html=True)
    
    with col_s3:
        st.markdown(f"""<div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05)); border: 2px solid rgba(239,68,68,0.5); box-shadow: 0 4px 6px rgba(239,68,68,0.2);">
            <h3 style="color: #f87171; margin: 0;">⚠️ 悲觀情境</h3>
            <p style="font-size: 32px; font-weight: bold; color: #f87171; margin: 15px 0;">{strategies['悲觀']['目標']:.2f}</p>
            <div style="background: rgba(239,68,68,0.2); padding: 8px; border-radius: 6px; margin: 10px 0;">
                <p style="color: #9ca3af; margin: 0; font-size: 13px;">概率: {strategies['悲觀']['概率']}</p>
            </div>
            <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 13px; line-height: 1.5;">{strategies['悲觀']['策略']}</p></div>""", unsafe_allow_html=True)
    
    # 【修復 4】多時間框架圖表 - 加入浪型標記
    st.markdown("### 📊 多時間框架圖表")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # K線圖
    fig.add_trace(go.Candlestick(
        x=df_small['Date'], open=df_small['Open'], high=df_small['High'],
        low=df_small['Low'], close=df_small['Close'], name='K線',
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444'
    ), row=1, col=1)
    
    # 【新增】標記 Swing 高低點
    if swings:
        swing_x_high = [df_small['Date'].iloc[s['idx']] for s in swings if s['type'] == 'high']
        swing_y_high = [s['price'] for s in swings if s['type'] == 'high']
        swing_x_low = [df_small['Date'].iloc[s['idx']] for s in swings if s['type'] == 'low']
        swing_y_low = [s['price'] for s in swings if s['type'] == 'low']
        
        fig.add_trace(go.Scatter(
            x=swing_x_high, y=swing_y_high, mode='markers+text',
            marker=dict(size=12, color='#ef4444', symbol='triangle-down'),
            text=['H'] * len(swing_x_high), textposition='top center',
            name='Swing High'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=swing_x_low, y=swing_y_low, mode='markers+text',
            marker=dict(size=12, color='#22c55e', symbol='triangle-up'),
            text=['L'] * len(swing_x_low), textposition='bottom center',
            name='Swing Low'
        ), row=1, col=1)
    
    # FIB 線
    for i, price in enumerate(fib_zones['支撐位']):
        if price > 0:
            fig.add_hline(y=price, line_dash="dash", line_color="#22c55e", 
                         opacity=0.5, annotation_text=f"支撐{i+1}", row=1, col=1)
    for i, price in enumerate(fib_zones['阻力位']):
        if price > 0:
            fig.add_hline(y=price, line_dash="dash", line_color="#ef4444", 
                         opacity=0.5, annotation_text=f"阻力{i+1}", row=1, col=1)
    
    # 成交量
    if 'Volume' in df_small.columns:
        colors = ['#22c55e' if df_small['Close'].iloc[i] >= df_small['Open'].iloc[i] else '#ef4444' 
                 for i in range(len(df_small))]
        fig.add_trace(go.Bar(
            x=df_small['Date'], y=df_small['Volume'], 
            marker_color=colors, name='成交量', opacity=0.7
        ), row=2, col=1)
    
    fig.update_layout(
        height=700, template="plotly_dark", showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # 【修復 5】觀察清單 - 美化 UI
    st.title("️ 觀察清單 (自動分析)")
    
    # 添加股票表單
    with st.form("add_stock", clear_on_submit=False):
        st.markdown("### ➕ 添加股票到觀察清單")
        col1, col2 = st.columns([3, 1])
        with col1: 
            new_code = st.text_input("輸入股票代碼", placeholder="例如: AAPL, 0700", key="watch_code").upper()
        with col2:
            submit_btn = st.form_submit_button("➕ 加入清單", use_container_width=True)
        
        if submit_btn and new_code:
            if not any(w['code'] == new_code for w in data['watchlist']):
                new_ticker = f"{new_code}.HK" if market == "HK" else new_code
                new_df = fetch_data(new_ticker, "1y", "1d")
                
                if new_df is not None:
                    h, l, s = find_swings(new_df, threshold)
                    w_type, w_desc = identify_wave_pattern(s, new_df)
                    d_trend, d_desc = analyze_dow_theory(new_df)
                    fib = calculate_fib_zones(h, l)
                    strat = generate_strategy(new_df['Close'].iloc[-1], fib, w_type, d_trend)
                    
                    data['watchlist'].append({
                        "code": new_code,
                        "wave_type": w_type,
                        "trend": d_trend,
                        "strategy": strat,
                        "fib_support": fib['支撐位'][2],
                        "fib_resist": fib['阻力位'][1],
                        "current_price": new_df['Close'].iloc[-1],
                        "added_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                    })
                    sync.save_data(data)
                    st.success(f"✅ 已加入 {new_code} 並自動分析！")
                    st.rerun()
                else:
                    st.error("❌ 無法獲取數據，請檢查代碼")
            else:
                st.warning(f"⚠️ {new_code} 已在清單中")
    
    # 顯示清單 - 美化卡片
    if data['watchlist']:
        st.markdown(f"### 📋 已追蹤 {len(data['watchlist'])} 支股票")
        
        for i, item in enumerate(data['watchlist']):
            # 根據浪型選擇顏色
            if "第3浪" in item.get('wave_type', ''):
                card_color = "22,197,94"  # 綠色
                icon = "🚀"
            elif "第5浪" in item.get('wave_type', ''):
                card_color = "234,179,8"  # 黃色
                icon = "⚠️"
            elif "回調" in item.get('wave_type', ''):
                card_color = "59,130,246"  # 藍色
                icon = "📉"
            else:
                card_color = "147,51,234"  # 紫色
                icon = "📊"
            
            with st.expander(f"{icon} **{item['code']}** - {item['wave_type']} - 加入: {item['added_at']}", expanded=True):
                # 價格資訊
                col_price1, col_price2, col_price3 = st.columns(3)
                with col_price1:
                    st.metric("當前價格", f"{item.get('current_price', 'N/A'):.2f}" if isinstance(item.get('current_price'), (int, float)) else "N/A")
                with col_price2:
                    st.metric("支撐位", f"{item['fib_support']:.2f}")
                with col_price3:
                    st.metric("阻力位", f"{item['fib_resist']:.2f}")
                
                # 趨勢和浪型
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"** 趨勢:** {item['trend']}")
                with col_info2:
                    st.markdown(f"**🌊 浪型:** {item['wave_type']}")
                
                # 策略 (JSON 美化顯示)
                st.markdown("**🎯 操作策略：**")
                strategy_col1, strategy_col2, strategy_col3 = st.columns(3)
                with strategy_col1:
                    st.markdown(f"""<div style="background: rgba(34,197,94,0.1); padding: 10px; border-radius: 8px; border-left: 3px solid #22c55e;">
                        <strong style="color: #22c55e;"> 樂觀</strong><br>
                        <small>{item['strategy']['樂觀']['策略']}</small><br>
                        <strong>{item['strategy']['樂觀']['目標']:.2f}</strong> ({item['strategy']['樂觀']['概率']})
                    </div>""", unsafe_allow_html=True)
                with strategy_col2:
                    st.markdown(f"""<div style="background: rgba(59,130,246,0.1); padding: 10px; border-radius: 8px; border-left: 3px solid #60a5fa;">
                        <strong style="color: #60a5fa;">📊 基準</strong><br>
                        <small>{item['strategy']['基準']['策略']}</small><br>
                        <strong>{item['strategy']['基準']['目標']:.2f}</strong> ({item['strategy']['基準']['概率']})
                    </div>""", unsafe_allow_html=True)
                with strategy_col3:
                    st.markdown(f"""<div style="background: rgba(239,68,68,0.1); padding: 10px; border-radius: 8px; border-left: 3px solid #f87171;">
                        <strong style="color: #f87171;">⚠️ 悲觀</strong><br>
                        <small>{item['strategy']['悲觀']['策略']}</small><br>
                        <strong>{item['strategy']['悲觀']['目標']:.2f}</strong> ({item['strategy']['悲觀']['概率']})
                    </div>""", unsafe_allow_html=True)
                
                # 刪除按鈕
                if st.button(f"🗑️ 刪除 {item['code']}", key=f"del_wl_{i}", type="secondary"):
                    data['watchlist'].pop(i)
                    sync.save_data(data)
                    st.rerun()
                st.markdown("---")
    else:
        st.info("📭 觀察清單為空，請在上方添加股票代碼開始追蹤")

with tab3:
    st.title("💼 模擬持倉 (實時盈虧)")
    
    # 添加持倉表單
    with st.form("add_position", clear_on_submit=False):
        st.markdown("### ➕ 新增持倉")
        col1, col2, col3 = st.columns(3)
        with col1:
            pos_code = st.text_input("代碼", placeholder="例如: NVDA", key="pos_code_input").upper()
            pos_qty = st.number_input("數量 (股)", min_value=1, value=100, key="pos_qty_input")
        with col2:
            pos_entry = st.number_input("買入價", min_value=0.01, step=0.01, key="pos_entry_input")
            pos_dir = st.selectbox("方向", ["做多", "做空"], key="pos_dir_input")
        with col3:
            st.markdown("<div style='padding-top: 22px;'></div>", unsafe_allow_html=True)
            if st.form_submit_button("💼 記錄持倉", use_container_width=True, type="primary"):
                if pos_code and pos_entry > 0:
                    pos_ticker = f"{pos_code}.HK" if market == "HK" else pos_code
                    pos_df = fetch_data(pos_ticker, "1d", "1d")
                    
                    if pos_df is not None:
                        current = pos_df['Close'].iloc[-1]
                        pnl = (current - pos_entry) * pos_qty if pos_dir == "做多" else (pos_entry - current) * pos_qty
                        
                        h, l, s = find_swings(pos_df, threshold)
                        fib = calculate_fib_zones(h, l)
                        stop_loss = fib['支撐位'][2] if pos_dir == "做多" else fib['阻力位'][0]
                        
                        data['positions'].append({
                            "code": pos_code,
                            "entry": pos_entry,
                            "qty": pos_qty,
                            "dir": pos_dir,
                            "current": current,
                            "pnl": pnl,
                            "stop_loss": stop_loss,
                            "suggestion": f"止損位: {stop_loss:.2f}",
                            "added_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                        })
                        sync.save_data(data)
                        st.success(f"✅ 已記錄 {pos_code} 持倉")
                        st.rerun()
                    else:
                        st.error("❌ 無法獲取數據")
                else:
                    st.warning("⚠️ 請填寫完整資訊")
    
    # 顯示持倉
    if data['positions']:
        total_pnl = sum(p.get('pnl', 0) for p in data['positions'])
        st.metric(" 總模擬盈虧", f"{total_pnl:+,.2f}", f"{total_pnl:+,.2f}")
        st.markdown("---")
        
        for i, p in enumerate(data['positions']):
            # 更新當前價格
            p_ticker = f"{p['code']}.HK" if market == "HK" else p['code']
            p_df = fetch_data(p_ticker, "1d", "1d")
            if p_df is not None:
                p['current'] = p_df['Close'].iloc[-1]
                p['pnl'] = (p['current'] - p['entry']) * p['qty'] if p['dir'] == "做多" else (p['entry'] - p['current']) * p['qty']
            
            pnl = p.get('pnl', 0)
            color = "green" if pnl >= 0 else "red"
            icon = "" if pnl >= 0 else "📉"
            
            st.markdown(f"""<div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba({34 if pnl>=0 else 239}, {197 if pnl>=0 else 68}, {94 if pnl>=0 else 68}, 0.1), rgba({34 if pnl>=0 else 239}, {197 if pnl>=0 else 68}, {94 if pnl>=0 else 68}, 0.05)); border-left: 5px solid {'#22c55e' if pnl>=0 else '#ef4444'}; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px;">
                        <h3 style="margin: 0; color: {'#22c55e' if pnl>=0 else '#ef4444'}; font-size: 24px;">{icon} {p['code']} ({p['dir']})</h3>
                        <p style="margin: 8px 0; color: #9ca3af; font-size: 14px;">
                            買入: <strong>{p['entry']}</strong> x {p['qty']}股 | 
                            現價: <strong>{p['current']:.2f}</strong> | 
                            加入: {p.get('added_at', 'N/A')}
                        </p>
                        <p style="margin: 5px 0; color: #6b7280; font-size: 13px;">
                            {p.get('suggestion', '無止損建議')}
                        </p>
                    </div>
                    <div style="text-align: right; min-width: 150px;">
                        <p style="font-size: 32px; font-weight: bold; color: {color}; margin: 0;">{pnl:+,.2f}</p>
                        <p style="font-size: 14px; color: #9ca3af; margin: 5px 0;">
                            回報率: {((p['current'] - p['entry']) / p['entry'] * 100 if p['dir'] == '做多' else (p['entry'] - p['current']) / p['entry'] * 100):+.2f}%
                        </p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            col_btn1, col_btn2 = st.columns([4, 1])
            with col_btn2:
                if st.button(f"平倉", key=f"close_{i}", type="secondary", use_container_width=True):
                    data['positions'].pop(i)
                    sync.save_data(data)
                    st.rerun()
        
        st.markdown("---")
    else:
        st.info("💭 尚無持倉記錄，開始建立你的模擬投資組合吧！")
