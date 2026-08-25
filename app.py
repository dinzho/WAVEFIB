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
# 1. GitHub 同步管理
# ==========================================
class GitHubSync:
    def __init__(self):
        self.token = st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN"))
        self.repo_name = st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO", ""))
        self.file_path = "portfolio_data.json"
        
        if self.token and self.repo_name:
            try:
                self.g = Github(self.token)
                self.repo = self.g.get_repo(self.repo_name)
                self.is_configured = True
            except:
                self.is_configured = False
        else:
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
# 2. 技術分析核心函數
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
    if len(df) < 2: return None, None, []
    last_high, last_low = df['High'].iloc[0], df['Low'].iloc[0]
    trend = 0
    swings = []
    
    for i in range(1, len(df)):
        high, low = df['High'].iloc[i], df['Low'].iloc[i]
        if trend == 0:
            if high > last_high: last_high = high
            if low < last_low: last_low = low
            if (high - last_low) / last_low * 100 >= threshold_pct:
                trend = 1; swings.append({'idx': i-1, 'type': 'low', 'price': last_low}); last_high = high
            elif (last_high - low) / last_high * 100 >= threshold_pct:
                trend = -1; swings.append({'idx': i-1, 'type': 'high', 'price': last_high}); last_low = low
        elif trend == 1:
            if high > last_high: last_high = high
            elif (last_high - low) / last_high * 100 >= threshold_pct:
                swings.append({'idx': i-1, 'type': 'high', 'price': last_high}); trend = -1; last_low = low
        else:
            if low < last_low: last_low = low
            elif (high - last_low) / last_low * 100 >= threshold_pct:
                swings.append({'idx': i-1, 'type': 'low', 'price': last_low}); trend = 1; last_high = high
    
    high_price = swings[-1]['price'] if swings and swings[-1]['type'] == 'high' else df['High'].max()
    low_price = swings[-1]['price'] if swings and swings[-1]['type'] == 'low' else df['Low'].min()
    return high_price, low_price, swings

def identify_wave_pattern(swings):
    """自動識別浪型"""
    if len(swings) < 2: return "數據不足", "等待更多數據確認趨勢"
    
    # 檢查最近走勢
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
    if len(df) < 50: return None
    
    # 計算高低點
    recent_highs = df['High'].rolling(window=20).max().iloc[-20:]
    recent_lows = df['Low'].rolling(window=20).min().iloc[-20:]
    
    # 判斷趨勢
    higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[0]
    higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[0]
    
    if higher_highs and higher_lows:
        return "上升趨勢", "HH & HL 持續確認，主要趨勢向上"
    elif not higher_highs and not higher_lows:
        return "下降趨勢", "LL & LH 確認，主要趨勢向下"
    else:
        return "盤整趨勢", "高低點混亂，等待方向突破"

def calculate_fib_zones(high, low):
    """計算斐波那契區間"""
    diff = high - low
    return {
        "阻力位": [high + diff * 0.618, high + diff * 1.0, high + diff * 1.618],
        "支撐位": [high - diff * 0.382, high - diff * 0.5, high - diff * 0.618],
        "關鍵位": [low, high]
    }

def analyze_pattern(df):
    """形態與趨勢分析"""
    if len(df) < 30: return "數據不足", "中性"
    
    # 簡化形態識別
    recent = df.iloc[-20:]
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
    if "上升" in trend and ("第3浪" in wave_type or "第1浪" in wave_type):
        return {
            "樂觀": {
                "目標": fib_zones["阻力位"][2],
                "概率": "30%",
                "策略": "突破前高後加倉，目標看1.618延伸"
            },
            "基準": {
                "目標": fib_zones["阻力位"][1],
                "概率": "50%",
                "策略": "持倉觀望，關注成交量變化"
            },
            "悲觀": {
                "目標": fib_zones["支撐位"][2],
                "概率": "20%",
                "策略": "跌破FIB 61.8%止損離場"
            }
        }
    elif "回調" in wave_type:
        return {
            "樂觀": {
                "目標": fib_zones["支撐位"][0],
                "概率": "40%",
                "策略": "在38.2%支撐位接多，博反彈"
            },
            "基準": {
                "目標": fib_zones["支撐位"][1],
                "概率": "40%",
                "策略": "在50%位置分批建倉"
            },
            "悲觀": {
                "目標": fib_zones["支撐位"][2],
                "概率": "20%",
                "策略": "跌破61.8%放棄做多，等待新信號"
            }
        }
    else:
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
sync = GitHubSync()
data = sync.load_data()

# Sidebar - GitHub 配置提示
with st.sidebar:
    st.title("⚙️ 系統設定")
    
    if not sync.is_configured:
        st.warning("""
        **⚠️ 未配置 GitHub Token**
        
        數據僅在當前瀏覽器暫存，重新整理會消失。
        
        **配置步驟：**
        1. 在 Streamlit Cloud 點擊 "Manage app"
        2. 點擊 "Secrets"
        3. 貼上：
        ```toml
        GITHUB_TOKEN = "ghp_xxx"
        GITHUB_REPO = "username/repo"
        ```
        """)
    else:
        st.success("✅ GitHub 同步已啟用")
    
    st.markdown("---")
    st.title(" 股票查詢")
    search_code = st.text_input("股票代碼", "NVDA").upper()
    market = st.selectbox("市場", ["US", "HK"])
    ticker = f"{search_code}.HK" if market == "HK" else search_code
    
    st.markdown("---")
    st.title("️ 分析參數")
    threshold = st.slider("波段閾值 (%)", 1.0, 15.0, 3.0, 0.5)
    tf_large = st.selectbox("大級別", ["週線", "月線", "日線"])
    tf_small = st.selectbox("小級別", ["日線", "小時線", "週線"])

# 主標題
st.title("🌊 智能個股分析平台 - 波浪理論專業版")

# 獲取數據
period_map = {"月線": "5y", "週線": "2y", "日線": "1y", "小時線": "3mo"}
interval_map = {"月線": "1mo", "週線": "1wk", "日線": "1d", "小時線": "1h"}

df_large = fetch_data(ticker, period_map[tf_large], interval_map[tf_large])
df_small = fetch_data(ticker, period_map[tf_small], interval_map[tf_small])

if df_large is None or df_small is None:
    st.error("無法獲取數據，請檢查代碼")
    st.stop()

# 當前價格
current_price = df_small['Close'].iloc[-1]
prev_close = df_small['Close'].iloc[-2] if len(df_small) > 1 else current_price
change = current_price - prev_close
change_pct = (change / prev_close) * 100

# 顯示價格卡片
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
    vol_ratio = df_small['Volume'].iloc[-1] / df_small['Volume'].rolling(20).mean().iloc[-1]
    st.metric("成交量比", f"{vol_ratio:.2f}x", "放量" if vol_ratio > 1.5 else "正常")

# 技術分析
high_price, low_price, swings = find_swings(df_small, threshold)
wave_type, wave_desc = identify_wave_pattern(swings)
dow_trend, dow_desc = analyze_dow_theory(df_small)
pattern, pattern_desc = analyze_pattern(df_small)
fib_zones = calculate_fib_zones(high_price, low_price)
strategies = generate_strategy(current_price, fib_zones, wave_type, dow_trend)

# 標籤頁
tab1, tab2, tab3 = st.tabs([" 共振分析儀表板", "️ 觀察清單", "💼 模擬持倉"])

with tab1:
    # 道氏理論卡片
    st.markdown("### 📐 道氏理論分析")
    col_dow1, col_dow2 = st.columns(2)
    with col_dow1:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(147,51,234,0.1)); border: 2px solid rgba(59,130,246,0.3);">
            <h3 style="color: #60a5fa; margin: 0;">📊 趨勢方向</h3>
            <p style="font-size: 24px; font-weight: bold; color: #60a5fa; margin: 10px 0;">{dow_trend}</p>
            <p style="color: #9ca3af; margin: 0;">{dow_desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_dow2:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(168,85,247,0.1)); border: 2px solid rgba(34,197,94,0.3);">
            <h3 style="color: #22c55e; margin: 0;">🎯 波浪定位</h3>
            <p style="font-size: 24px; font-weight: bold; color: #22c55e; margin: 10px 0;">{wave_type}</p>
            <p style="color: #9ca3af; margin: 0;">{wave_desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 斐波那契區間
    st.markdown("###  斐波那契區間")
    col_fib1, col_fib2, col_fib3 = st.columns(3)
    with col_fib1:
        st.markdown("**📈 阻力位**")
        st.write(f"1.618: {fib_zones['阻力位'][2]:.2f}")
        st.write(f"1.000: {fib_zones['阻力位'][1]:.2f}")
        st.write(f"0.618: {fib_zones['阻力位'][0]:.2f}")
    with col_fib2:
        st.markdown("**📉 支撐位**")
        st.write(f"38.2%: {fib_zones['支撐位'][0]:.2f}")
        st.write(f"50.0%: {fib_zones['支撐位'][1]:.2f}")
        st.write(f"61.8%: {fib_zones['支撐位'][2]:.2f}")
    with col_fib3:
        st.markdown("**🎯 關鍵位**")
        st.write(f"高點: {fib_zones['關鍵位'][1]:.2f}")
        st.write(f"低點: {fib_zones['關鍵位'][0]:.2f}")
    
    # 形態與趨勢
    st.markdown("### 📈 形態與趨勢分析")
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 8px; background: rgba(234,179,8,0.1); border-left: 4px solid #eab308;">
        <p style="margin: 0;"><strong>當前形態:</strong> {pattern}</p>
        <p style="margin: 5px 0 0 0; color: #9ca3af;">{pattern_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 三種情境策略
    st.markdown("###  建議策略與三種情況點位")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05)); border: 2px solid rgba(34,197,94,0.4);">
            <h3 style="color: #22c55e; margin: 0;">🚀 樂觀情境</h3>
            <p style="font-size: 28px; font-weight: bold; color: #22c55e; margin: 10px 0;">{strategies['樂觀']['目標']:.2f}</p>
            <p style="color: #9ca3af; margin: 5px 0;">概率: {strategies['樂觀']['概率']}</p>
            <p style="color: #6b7280; margin: 5px 0; font-size: 13px;">{strategies['樂觀']['策略']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s2:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.05)); border: 2px solid rgba(59,130,246,0.4);">
            <h3 style="color: #60a5fa; margin: 0;">📊 基準情境</h3>
            <p style="font-size: 28px; font-weight: bold; color: #60a5fa; margin: 10px 0;">{strategies['基準']['目標']:.2f}</p>
            <p style="color: #9ca3af; margin: 5px 0;">概率: {strategies['基準']['概率']}</p>
            <p style="color: #6b7280; margin: 5px 0; font-size: 13px;">{strategies['基準']['策略']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s3:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); border: 2px solid rgba(239,68,68,0.4);">
            <h3 style="color: #f87171; margin: 0;">⚠️ 悲觀情境</h3>
            <p style="font-size: 28px; font-weight: bold; color: #f87171; margin: 10px 0;">{strategies['悲觀']['目標']:.2f}</p>
            <p style="color: #9ca3af; margin: 5px 0;">概率: {strategies['悲觀']['概率']}</p>
            <p style="color: #6b7280; margin: 5px 0; font-size: 13px;">{strategies['悲觀']['策略']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 圖表
    st.markdown("### 📊 多時間框架圖表")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # K線
    fig.add_trace(go.Candlestick(
        x=df_small['Date'], open=df_small['Open'], high=df_small['High'],
        low=df_small['Low'], close=df_small['Close'], name='K線'
    ), row=1, col=1)
    
    # FIB 線
    for i, price in enumerate(fib_zones['支撐位']):
        fig.add_hline(y=price, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    for i, price in enumerate(fib_zones['阻力位']):
        fig.add_hline(y=price, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    
    # 成交量
    colors = ['#22c55e' if df_small['Close'].iloc[i] >= df_small['Open'].iloc[i] else '#ef4444' for i in range(len(df_small))]
    fig.add_trace(go.Bar(x=df_small['Date'], y=df_small['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.title("👁️ 觀察清單 (自動分析)")
    
    # 添加股票
    with st.form("add_stock"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_code = st.text_input("輸入股票代碼", "").upper()
        with col2:
            if st.form_submit_button(" 加入清單", use_container_width=True):
                if new_code and not any(w['code'] == new_code for w in data['watchlist']):
                    # 自動分析
                    new_ticker = f"{new_code}.HK" if market == "HK" else new_code
                    new_df = fetch_data(new_ticker, "1y", "1d")
                    
                    if new_df is not None:
                        h, l, s = find_swings(new_df, threshold)
                        w_type, w_desc = identify_wave_pattern(s)
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
                            "added_at": pd.Timestamp.now().strftime("%Y-%m-%d")
                        })
                        sync.save_data(data)
                        st.success(f"已加入 {new_code} 並自動分析！")
                        st.rerun()
                    else:
                        st.error("無法獲取數據")
                elif new_code:
                    st.warning("已在清單中")
    
    # 顯示清單
    if data['watchlist']:
        for i, item in enumerate(data['watchlist']):
            with st.expander(f"📌 {item['code']} - {item['wave_type']} - 加入日期: {item['added_at']}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**趨勢:** {item['trend']}")
                    st.markdown(f"**支撐位:** {item['fib_support']:.2f}")
                with col2:
                    st.markdown(f"**阻力位:** {item['fib_resist']:.2f}")
                    st.markdown("**操作策略:**")
                    st.json(item['strategy'])
                
                if st.button(f"🗑️ 刪除", key=f"del_wl_{i}"):
                    data['watchlist'].pop(i)
                    sync.save_data(data)
                    st.rerun()
    else:
        st.info("觀察清單為空，請在上方添加股票代碼")

with tab3:
    st.title(" 模擬持倉 (實時盈虧)")
    
    # 添加持倉
    with st.form("add_position"):
        col1, col2, col3 = st.columns(3)
        with col1:
            pos_code = st.text_input("代碼", "").upper()
            pos_qty = st.number_input("數量 (股)", min_value=1, value=100)
        with col2:
            pos_entry = st.number_input("買入價", min_value=0.01, step=0.01)
            pos_dir = st.selectbox("方向", ["做多", "做空"])
        with col3:
            if st.form_submit_button("💼 記錄持倉", use_container_width=True):
                if pos_code:
                    # 獲取當前價格
                    pos_ticker = f"{pos_code}.HK" if market == "HK" else pos_code
                    pos_df = fetch_data(pos_ticker, "1d", "1d")
                    
                    if pos_df is not None:
                        current = pos_df['Close'].iloc[-1]
                        pnl = (current - pos_entry) * pos_qty if pos_dir == "做多" else (pos_entry - current) * pos_qty
                        
                        # 自動生成建議
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
                            "suggestion": f"止損位: {stop_loss:.2f}"
                        })
                        sync.save_data(data)
                        st.rerun()
    
    # 顯示持倉
    if data['positions']:
        total_pnl = sum(p['pnl'] for p in data['positions'])
        st.metric("總模擬盈虧", f"{total_pnl:+,.2f}", f"{total_pnl:+,.2f}")
        
        for i, p in enumerate(data['positions']):
            # 更新當前價格
            p_ticker = f"{p['code']}.HK" if market == "HK" else p['code']
            p_df = fetch_data(p_ticker, "1d", "1d")
            if p_df is not None:
                p['current'] = p_df['Close'].iloc[-1]
                p['pnl'] = (p['current'] - p['entry']) * p['qty'] if p['dir'] == "做多" else (p['entry'] - p['current']) * p['qty']
            
            color = "green" if p['pnl'] >= 0 else "red"
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background: rgba({34 if p['pnl']>=0 else 239}, {197 if p['pnl']>=0 else 68}, {94 if p['pnl']>=0 else 68}, 0.1); border-left: 4px solid {'#22c55e' if p['pnl']>=0 else '#ef4444'}; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0; color: {'#22c55e' if p['pnl']>=0 else '#ef4444'};">{p['code']} ({p['dir']})</h3>
                        <p style="margin: 5px 0; color: #9ca3af;">買入: {p['entry']} x {p['qty']}股 | 現價: {p['current']:.2f}</p>
                        <p style="margin: 5px 0; color: #6b7280;">{p['suggestion']}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 24px; font-weight: bold; color: {color}; margin: 0;">{p['pnl']:+,.2f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"平倉", key=f"close_{i}"):
                data['positions'].pop(i)
                sync.save_data(data)
                st.rerun()
    else:
        st.info("尚無持倉記錄")
