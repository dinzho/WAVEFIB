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
            raw_content = file_content.decoded_content.decode("utf-8").strip()
            
            if not raw_content:
                data = self._get_default_data()
            else:
                data = json.loads(raw_content)
                
            st.session_state['app_data'] = data
            return data
        except GithubException as e:
            if e.status == 404:
                self.save_data(self._get_default_data())
                return self._get_default_data()
            return self._get_default_data()
        except json.JSONDecodeError:
            st.warning("⚠️ 雲端數據格式錯誤，已自動重置為預設數據。")
            default_data = self._get_default_data()
            self.save_data(default_data)
            return default_data

    def save_data(self, data):
        st.session_state['app_data'] = data
        if not self.is_configured:
            return
        
        try:
            if not isinstance(data, dict):
                data = self._get_default_data()
                
            file_content = self.repo.get_contents(self.file_path)
            self.repo.update_file(
                self.file_path, "Update portfolio", 
                json.dumps(data, indent=2, ensure_ascii=False), file_content.sha
            )
        except GithubException as e:
            if e.status == 404:
                try:
                    self.repo.create_file(
                        self.file_path, "Create portfolio data", 
                        json.dumps(data, indent=2, ensure_ascii=False)
                    )
                except:
                    pass

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
    if df is None or len(df) < 2:
        return 100.0, 90.0, []
    
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
    
    high_price = float(df['High'].max())
    low_price = float(df['Low'].min())
    
    if high_price <= low_price:
        high_price = low_price * 1.05
    
    return high_price, low_price, swings

def identify_wave_pattern(swings, df=None):
    if not swings or len(swings) < 2: 
        return "趨勢初期", "數據不足"
    
    last_swings = swings[-5:] if len(swings) >= 5 else swings
    highs = [s['price'] for s in last_swings if s['type'] == 'high']
    lows = [s['price'] for s in last_swings if s['type'] == 'low']
    
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "第3浪 (主升段)", "強勁上升趨勢"
        elif highs[-1] < highs[-2] and lows[-1] > lows[-2]:
            return "第5浪 (尾聲)", "上升動能減弱"
        elif highs[-1] > highs[-2] and lows[-1] < lows[-2]:
            return "第2/4浪 (回調)", "回調階段"
        else:
            return "調整浪 (ABC)", "盤整或下跌"
    
    return "趨勢初期", "等待信號"

def calculate_fib_zones(high, low):
    if high is None or low is None:
        high, low = 100.0, 90.0
    
    if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
        high, low = 100.0, 90.0
    
    if high == low or high <= 0 or low <= 0:
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
    if len(df) < 30: return "數據不足", "中性"
    
    ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
    ma50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else ma20
    current_price = df['Close'].iloc[-1]
    
    if current_price > ma20 > ma50:
        return "多頭排列", "MA20 > MA50，趨勢向上"
    elif current_price < ma20 < ma50:
        return "空頭排列", "MA20 < MA50，趨勢向下"
    else:
        return "均線糾纏", "均線交錯"

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

with st.sidebar:
    st.title("️ 系統設定")
    
    st.markdown("### 🔑 GitHub 配置")
    github_token = st.text_input("GitHub Token", type="password", help="Settings > Developer settings > Personal access tokens")
    github_repo = st.text_input("GitHub 倉庫", placeholder="username/repo", help="格式: 你的用戶名/倉庫名")
    
    if github_token and github_repo:
        sync = GitHubSync(token=github_token, repo_name=github_repo)
        if sync.is_configured:
            st.success("✅ GitHub 同步已啟用")
        else:
            st.error("❌ 配置失敗")
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
    threshold = st.slider("波段閾值 (%)", 0.5, 15.0, 3.0, 0.5, help="小時線建議 1-2%，日線建議 3-5%")
    tf_large = st.selectbox("大級別", ["週線", "月線", "日線"])
    tf_small = st.selectbox("小級別", ["日線", "小時線", "週線"])

st.title("🌊 智能個股分析平台 - 波浪理論專業版")

data = sync.load_data()
period_map = {"月線": "5y", "週線": "2y", "日線": "1y", "小時線": "3mo"}
interval_map = {"月線": "1mo", "週線": "1wk", "日線": "1d", "小時線": "1h"}

df_large = fetch_data(ticker, period_map[tf_large], interval_map[tf_large])
df_small = fetch_data(ticker, period_map[tf_small], interval_map[tf_small])

if df_large is None or df_small is None:
    st.error("無法獲取數據，請檢查代碼或嘗試切換時間框架。")
    st.stop()

current_price = df_small['Close'].iloc[-1]
prev_close = df_small['Close'].iloc[-2] if len(df_small) > 1 else current_price
change = current_price - prev_close
change_pct = (change / prev_close) * 100

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

high_price, low_price, swings = find_swings(df_small, threshold)
wave_type, wave_desc = identify_wave_pattern(swings, df_small)
pattern, pattern_desc = analyze_pattern(df_small)
fib_zones = calculate_fib_zones(high_price, low_price)

tab1, tab2, tab3, tab4 = st.tabs(["📊 共振分析儀表板", "👁️ 觀察清單", " 模擬持倉 (Excel)", "📅 每日持倉日報"])

with tab1:
    st.markdown("### 📐 道氏理論與波浪分析")
    col_dow1, col_dow2 = st.columns(2)
    with col_dow1:
        st.markdown(f"""<div style="padding: 15px; border-radius: 10px; background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(147,51,234,0.1)); border: 2px solid rgba(59,130,246,0.4);">
            <h3 style="color: #60a5fa; margin: 0 0 10px 0;">📊 趨勢方向</h3>
            <p style="font-size: 22px; font-weight: bold; color: #60a5fa; margin: 10px 0;">{wave_type}</p>
            <p style="color: #9ca3af; margin: 0; font-size: 13px;">{wave_desc}</p></div>""", unsafe_allow_html=True)
    with col_dow2:
        st.markdown(f"""<div style="padding: 15px; border-radius: 10px; background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(168,85,247,0.1)); border: 2px solid rgba(34,197,94,0.4);">
            <h3 style="color: #22c55e; margin: 0 0 10px 0;">🎯 形態分析</h3>
            <p style="font-size: 22px; font-weight: bold; color: #22c55e; margin: 10px 0;">{pattern}</p>
            <p style="color: #9ca3af; margin: 0; font-size: 13px;">{pattern_desc}</p></div>""", unsafe_allow_html=True)
    
    st.markdown("###  斐波那契區間")
    col_fib1, col_fib2, col_fib3 = st.columns(3)
    
    with col_fib1:
        st.markdown("""<div style="background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05)); padding: 12px; border-radius: 8px; border-left: 4px solid #ef4444;">
            <h4 style="color: #ef4444; margin: 0 0 8px 0; font-size: 16px;">📈 阻力位</h4>
        </div>""", unsafe_allow_html=True)
        st.metric("1.618 延伸", f"{fib_zones['阻力位'][2]:.2f}")
        st.metric("1.000 等長", f"{fib_zones['阻力位'][1]:.2f}")
        st.metric("0.618 阻力", f"{fib_zones['阻力位'][0]:.2f}")
    
    with col_fib2:
        st.markdown("""<div style="background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05)); padding: 12px; border-radius: 8px; border-left: 4px solid #22c55e;">
            <h4 style="color: #22c55e; margin: 0 0 8px 0; font-size: 16px;">📉 支撐位</h4>
        </div>""", unsafe_allow_html=True)
        st.metric("38.2% 支撐", f"{fib_zones['支撐位'][0]:.2f}")
        st.metric("50.0% 中軸", f"{fib_zones['支撐位'][1]:.2f}")
        st.metric("61.8% 強支撐", f"{fib_zones['支撐位'][2]:.2f}")
    
    with col_fib3:
        st.markdown("""<div style="background: linear-gradient(135deg, rgba(234,179,8,0.1), rgba(234,179,8,0.05)); padding: 12px; border-radius: 8px; border-left: 4px solid #eab308;">
            <h4 style="color: #eab308; margin: 0 0 8px 0; font-size: 16px;"> 關鍵位</h4>
        </div>""", unsafe_allow_html=True)
        st.metric("波段高點", f"{fib_zones['關鍵位'][1]:.2f}")
        st.metric("波段低點", f"{fib_zones['關鍵位'][0]:.2f}")
        st.metric("波動幅度", f"{fib_zones['high'] - fib_zones['low']:.2f}")
    
    st.markdown("### 📊 多時間框架圖表")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df_small['Date'], open=df_small['Open'], high=df_small['High'],
        low=df_small['Low'], close=df_small['Close'], name='K線',
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444'
    ), row=1, col=1)
    
    if swings:
        swing_x_high = [df_small['Date'].iloc[s['idx']] for s in swings if s['type'] == 'high']
        swing_y_high = [s['price'] for s in swings if s['type'] == 'high']
        swing_x_low = [df_small['Date'].iloc[s['idx']] for s in swings if s['type'] == 'low']
        swing_y_low = [s['price'] for s in swings if s['type'] == 'low']
        
        fig.add_trace(go.Scatter(
            x=swing_x_high, y=swing_y_high, mode='markers+text',
            marker=dict(size=10, color='#ef4444', symbol='triangle-down'),
            text=['H'] * len(swing_x_high), textposition='top center',
            name='Swing High'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=swing_x_low, y=swing_y_low, mode='markers+text',
            marker=dict(size=10, color='#22c55e', symbol='triangle-up'),
            text=['L'] * len(swing_x_low), textposition='bottom center',
            name='Swing Low'
        ), row=1, col=1)
    
    for i, price in enumerate(fib_zones['支撐位']):
        if price > 0:
            fig.add_hline(y=price, line_dash="dash", line_color="#22c55e", 
                         opacity=0.5, annotation_text=f"支撐{i+1}", row=1, col=1)
    for i, price in enumerate(fib_zones['阻力位']):
        if price > 0:
            fig.add_hline(y=price, line_dash="dash", line_color="#ef4444", 
                         opacity=0.5, annotation_text=f"阻力{i+1}", row=1, col=1)
    
    if 'Volume' in df_small.columns:
        colors = ['#22c55e' if df_small['Close'].iloc[i] >= df_small['Open'].iloc[i] else '#ef4444' 
                 for i in range(len(df_small))]
        fig.add_trace(go.Bar(
            x=df_small['Date'], y=df_small['Volume'], 
            marker_color=colors, name='成交量', opacity=0.7
        ), row=2, col=1)
    
    fig.update_layout(
        height=600, template="plotly_dark", showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.title("👁️ 觀察清單 (自動分析)")
    
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
                    fib = calculate_fib_zones(h, l)
                    
                    data['watchlist'].append({
                        "code": new_code,
                        "wave_type": w_type,
                        "fib_support": fib['支撐位'][2],
                        "fib_resist": fib['阻力位'][1],
                        "current_price": new_df['Close'].iloc[-1],
                        "added_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                    })
                    sync.save_data(data)
                    st.success(f"✅ 已加入 {new_code}")
                    st.rerun()
                else:
                    st.error("❌ 無法獲取數據")
            else:
                st.warning(f"⚠️ {new_code} 已在清單中")
    
    if data['watchlist']:
        st.markdown(f"### 📋 已追蹤 {len(data['watchlist'])} 支股票")
        
        for i, item in enumerate(data['watchlist']):
            with st.expander(f"**{item['code']}** - {item['wave_type']} - 加入: {item['added_at']}", expanded=True):
                col_price1, col_price2, col_price3 = st.columns(3)
                with col_price1:
                    st.metric("當前價格", f"{item.get('current_price', 'N/A'):.2f}" if isinstance(item.get('current_price'), (int, float)) else "N/A")
                with col_price2:
                    st.metric("支撐位", f"{item['fib_support']:.2f}")
                with col_price3:
                    st.metric("阻力位", f"{item['fib_resist']:.2f}")
                
                if st.button(f"🗑️ 刪除 {item['code']}", key=f"del_wl_{i}", type="secondary"):
                    data['watchlist'].pop(i)
                    sync.save_data(data)
                    st.rerun()
                st.markdown("---")
    else:
        st.info(" 觀察清單為空")

# ==========================================
# Tab 3: 模擬持倉 (Excel 風格)
# ==========================================
with tab3:
    st.title("💼 模擬持倉管理")
    
    # 新增持倉表單 (使用 expander 折疊，保持界面整潔)
    with st.expander("➕ 新增持倉 (點擊展開)", expanded=False):
        with st.form("add_position", clear_on_submit=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pos_code = st.text_input("代碼", placeholder="NVDA", key="pos_code_input").upper()
            with col2:
                pos_entry = st.number_input("買入價", min_value=0.01, step=0.01, key="pos_entry_input")
            with col3:
                pos_qty = st.number_input("數量 (股)", min_value=1, value=100, key="pos_qty_input")
            with col4:
                pos_dir = st.selectbox("方向", ["做多", "做空"], key="pos_dir_input")
            
            submit_pos = st.form_submit_button("💼 記錄持倉", use_container_width=True, type="primary")
            
            if submit_pos and pos_code and pos_entry > 0:
                pos_ticker = f"{pos_code}.HK" if market == "HK" else pos_code
                pos_df = fetch_data(pos_ticker, "1d", "1d")
                
                if pos_df is not None:
                    current = pos_df['Close'].iloc[-1]
                    h, l, s = find_swings(pos_df, threshold)
                    fib = calculate_fib_zones(h, l)
                    stop_loss = fib['支撐位'][2] if pos_dir == "做多" else fib['阻力位'][0]
                    
                    data['positions'].append({
                        "code": pos_code,
                        "entry": pos_entry,
                        "qty": pos_qty,
                        "dir": pos_dir,
                        "current": current,
                        "stop_loss": stop_loss,
                        "added_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                    })
                    sync.save_data(data)
                    st.success(f"✅ 已記錄 {pos_code} 持倉")
                    st.rerun()
                else:
                    st.error("❌ 無法獲取數據")

    # Excel 風格表格展示
    if data['positions']:
        # 準備表格數據
        table_data = []
        total_pnl = 0
        total_cost = 0
        
        for p in data['positions']:
            # 更新現價
            p_ticker = f"{p['code']}.HK" if market == "HK" else p['code']
            p_df = fetch_data(p_ticker, "1d", "1d")
            if p_df is not None:
                p['current'] = p_df['Close'].iloc[-1]
            
            current = p.get('current', p['entry'])
            pnl = (current - p['entry']) * p['qty'] if p['dir'] == "做多" else (p['entry'] - current) * p['qty']
            pnl_pct = ((current - p['entry']) / p['entry'] * 100) if p['dir'] == "做多" else ((p['entry'] - current) / p['entry'] * 100)
            
            total_pnl += pnl
            total_cost += p['entry'] * p['qty']
            
            table_data.append({
                "代碼": p['code'],
                "方向": p['dir'],
                "買入價": p['entry'],
                "數量": p['qty'],
                "現價": current,
                "盈虧": pnl,
                "回報率%": pnl_pct,
                "止損位": p.get('stop_loss', 0)
            })
        
        # 顯示總盈虧
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric(" 總持倉成本", f"${total_cost:,.2f}")
        with col_t2:
            color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("💰 總模擬盈虧", f"${total_pnl:+,.2f}", f"{total_pnl:+,.2f}", delta_color=color)
        
        st.markdown("---")
        
        # 顯示 DataFrame (Excel 風格)
        df_positions = pd.DataFrame(table_data)
        
        # 格式化顯示
        st.dataframe(
            df_positions.style.format({
                "買入價": "${:.2f}",
                "現價": "${:.2f}",
                "盈虧": "${:+,.2f}",
                "回報率%": "{:+.2f}%",
                "止損位": "${:.2f}"
            }).background_gradient(subset=["盈虧"], cmap="RdYlGn"),
            use_container_width=True,
            height=300
        )
        
        st.markdown("---")
        st.markdown("### ️ 持倉操作")
        
        # 平倉按鈕 (放在表格下方，保持整潔)
        cols = st.columns(len(data['positions']))
        for i, p in enumerate(data['positions']):
            with cols[i % len(cols)]:
                if st.button(f"平倉 {p['code']}", key=f"close_{i}", type="secondary", use_container_width=True):
                    data['positions'].pop(i)
                    sync.save_data(data)
                    st.rerun()
    else:
        st.info("💭 尚無持倉記錄，請在上方展開表單新增持倉。")

# ==========================================
# Tab 4: 每日持倉日報
# ==========================================
with tab4:
    st.title("📅 每日持倉日報生成器")
    st.markdown("一鍵生成所有持倉的技術分析與操作建議，適合每日開盤前查看。")
    
    if st.button("🔄 生成今日持倉日報", type="primary", use_container_width=True):
        if not data['positions']:
            st.warning("⚠️ 尚無持倉記錄，請先到「模擬持倉」頁面新增。")
        else:
            st.markdown("---")
            st.markdown(f"### 📊 日報生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
            st.markdown("---")
            
            for i, p in enumerate(data['positions']):
                p_ticker = f"{p['code']}.HK" if market == "HK" else p['code']
                p_df = fetch_data(p_ticker, "1y", "1d")
                
                if p_df is not None:
                    current = p_df['Close'].iloc[-1]
                    p['current'] = current  # 更新持倉現價
                    
                    # 技術分析
                    h, l, s = find_swings(p_df, threshold)
                    fib = calculate_fib_zones(h, l)
                    wave_type, wave_desc = identify_wave_pattern(s, p_df)
                    pattern, pattern_desc = analyze_pattern(p_df)
                    rsi = calculate_rsi(p_df['Close']).iloc[-1]
                    
                    # 計算盈虧
                    pnl = (current - p['entry']) * p['qty'] if p['dir'] == "做多" else (p['entry'] - current) * p['qty']
                    pnl_pct = ((current - p['entry']) / p['entry'] * 100) if p['dir'] == "做多" else ((p['entry'] - current) / p['entry'] * 100)
                    
                    # 生成建議邏輯
                    suggestion = ""
                    risk_level = "🟢 低風險"
                    
                    stop_loss = p.get('stop_loss', fib['支撐位'][2] if p['dir'] == "做多" else fib['阻力位'][0])
                    
                    if p['dir'] == "做多":
                        if current < stop_loss:
                            suggestion = "⚠️ **觸發止損**：現價已跌破止損位，建議立即平倉或嚴格執行止損紀律。"
                            risk_level = "🔴 高風險"
                        elif current > p['entry'] * 1.1:
                            suggestion = "📈 **獲利豐厚**：建議上移止損位至成本價，鎖定利潤。可考慮部分獲利了結。"
                            risk_level = " 中風險"
                        elif rsi > 70:
                            suggestion = "️ **RSI 超買**：短期可能回調，建議持有但暫停加倉，關注 FIB 38.2% 支撐。"
                            risk_level = "🟡 中風險"
                        else:
                            suggestion = "✅ **趨勢良好**：建議繼續持有，止損位設在 " + f"{stop_loss:.2f}" + "。"
                    else:  # 做空
                        if current > stop_loss:
                            suggestion = "⚠️ **觸發止損**：現價已突破止損位，建議立即平倉。"
                            risk_level = "🔴 高風險"
                        elif current < p['entry'] * 0.9:
                            suggestion = "📉 **獲利豐厚**：建議下移止損位至成本價，鎖定利潤。"
                            risk_level = "🟡 中風險"
                        elif rsi < 30:
                            suggestion = "⚠️ **RSI 超賣**：短期可能反彈，建議持有但暫停加倉，關注 FIB 61.8% 阻力。"
                            risk_level = "🟡 中風險"
                        else:
                            suggestion = "✅ **趨勢良好**：建議繼續持有，止損位設在 " + f"{stop_loss:.2f}" + "。"
                    
                    # 顯示日報卡片
                    with st.expander(f"📌 {p['code']} ({p['dir']}) - 盈虧: {pnl:+,.2f} ({pnl_pct:+.2f}%)", expanded=True):
                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            st.markdown(f"**🌊 浪型與形態**")
                            st.markdown(f"- 浪型: {wave_type}")
                            st.markdown(f"- 形態: {pattern}")
                        with col_r2:
                            st.markdown(f"**📊 技術指標**")
                            st.markdown(f"- RSI: {rsi:.1f}")
                            st.markdown(f"- 現價: {current:.2f}")
                        with col_r3:
                            st.markdown(f"**🎯 關鍵價位**")
                            st.markdown(f"- 支撐: {fib['支撐位'][2]:.2f}")
                            st.markdown(f"- 阻力: {fib['阻力位'][1]:.2f}")
                        
                        st.markdown("---")
                        st.markdown(f"**💡 操作建議 ({risk_level})**")
                        st.markdown(suggestion)
                        
                        # 更新止損位到持倉數據
                        p['stop_loss'] = stop_loss
                    
                    st.markdown("---")
            
            # 保存更新後的持倉數據 (包含最新現價和止損位)
            sync.save_data(data)
            st.success("✅ 日報生成完畢，持倉數據已更新！")
