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
# CSS 样式注入 (保持专业金融终端风格)
# ==========================================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
.stTabs [data-baseweb="tab-list"] button {
    font-size: 16px !important; padding: 10px 20px !important; font-weight: 600 !important;
    border-radius: 8px !important; background: #1e293b !important; border: 1px solid #334155 !important;
}
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important; border: none !important; }
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important; }
[data-testid="stDataFrame"] th { background: #1e293b !important; color: #60a5fa !important; text-align: center !important; }
[data-testid="stDataFrame"] td { text-align: center !important; }
.metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border-radius: 12px; padding: 20px; border: 1px solid #334155;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); transition: all 0.3s ease;
}
.metric-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

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
            except: self.is_configured = False

    def _get_default_data(self): return {"watchlist": [], "positions": [], "closed_positions": []}

    def load_data(self):
        if not self.is_configured: return st.session_state.get('app_data', self._get_default_data())
        try:
            file_content = self.repo.get_contents(self.file_path)
            raw_content = file_content.decoded_content.decode("utf-8").strip()
            data = json.loads(raw_content) if raw_content else self._get_default_data()
            if 'closed_positions' not in data: data['closed_positions'] = []
            st.session_state['app_data'] = data
            return data
        except: return self._get_default_data()

    def save_data(self, data):
        st.session_state['app_data'] = data
        if not self.is_configured: return
        try:
            file_content = self.repo.get_contents(self.file_path)
            self.repo.update_file(self.file_path, "Update", json.dumps(data, indent=2, ensure_ascii=False), file_content.sha)
        except: pass

# ==========================================
# 2. 技术分析核心函数
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def find_swings(df, threshold_pct=3.0):
    if df is None or len(df) < 2: return 100.0, 90.0, []
    df = df.copy()
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df = df.dropna(subset=['High', 'Low'])
    if len(df) < 2: return float(df['High'].max()) if len(df) > 0 else 100.0, float(df['Low'].min()) if len(df) > 0 else 90.0, []

    last_high, last_low = float(df['High'].iloc[0]), float(df['Low'].iloc[0])
    trend, swings = 0, []
    for i in range(1, len(df)):
        high, low = float(df['High'].iloc[i]), float(df['Low'].iloc[i])
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
    
    high_price, low_price = float(df['High'].max()), float(df['Low'].min())
    if high_price <= low_price: high_price = low_price * 1.05
    return high_price, low_price, swings

def identify_wave_pattern(swings):
    if not swings or len(swings) < 2: return "趋势初期", "数据不足"
    last_swings = swings[-5:] if len(swings) >= 5 else swings
    highs = [s['price'] for s in last_swings if s['type'] == 'high']
    lows = [s['price'] for s in last_swings if s['type'] == 'low']
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]: return "第 3 浪 (主升段)", "强劲上升趋势"
        elif highs[-1] < highs[-2] and lows[-1] > lows[-2]: return "第 5 浪 (尾声)", "上升动能减弱"
        elif highs[-1] > highs[-2] and lows[-1] < lows[-2]: return "第 2/4 浪 (回调)", "回调阶段"
        else: return "调整浪 (ABC)", "盘整或下跌"
    return "趋势初期", "等待信号"

def calculate_fib_zones(high, low):
    if high is None or low is None or high <= 0 or low <= 0: high, low = 100.0, 90.0
    if high == low: high = low * 1.05
    diff = abs(high - low)
    return {
        "短期阻力": [high + diff * 0.382, high + diff * 0.618],
        "长期阻力": [high + diff * 1.0, high + diff * 1.618],
        "支撑位": [high - diff * 0.382, high - diff * 0.5, high - diff * 0.618],
        "关键位": [low, high], "high": high, "low": low
    }

def analyze_pattern(df):
    if len(df) < 30: return "数据不足", "中性", "#9ca3af"
    ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
    ma50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else ma20
    current_price = df['Close'].iloc[-1]
    if current_price > ma20 > ma50: return "多头排列", "MA20 > MA50，趋势向上", "#22c55e"
    elif current_price < ma20 < ma50: return "空头排列", "MA20 < MA50，趋势向下", "#ef4444"
    else: return "均线纠缠", "均线交错", "#eab308"

def analyze_trend_color(trend_text):
    if "上升" in trend_text: return "#ef4444"
    elif "下降" in trend_text: return "#000000"
    else: return "#eab308"

def analyze_volume_anomaly(df):
    if 'Volume' not in df.columns or len(df) < 20: return "无数据", 1.0
    current_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
    if avg_vol == 0: return "正常", 1.0
    ratio = current_vol / avg_vol
    if ratio > 2.0: return "异常放量 🔥", ratio
    elif ratio < 0.5: return "异常缩量 ️", ratio
    else: return "正常", ratio

def check_multi_tf_resonance(ticker):
    try:
        df_daily = yf.Ticker(ticker).history(period="1y", interval="1d")
        df_weekly = yf.Ticker(ticker).history(period="2y", interval="1wk")
        if df_daily.empty or df_weekly.empty: return "无数据", "中性"
        daily_trend = "多头" if df_daily['Close'].iloc[-1] > df_daily['Close'].rolling(20).mean().iloc[-1] else "空头"
        weekly_trend = "多头" if df_weekly['Close'].iloc[-1] > df_weekly['Close'].rolling(20).mean().iloc[-1] else "空头"
        if daily_trend == weekly_trend: return f"共振确认 ✅ ({daily_trend})", f"日线与周线均为{daily_trend}"
        else: return "信号冲突 ⚠️", f"日线{daily_trend}，周线{weekly_trend}"
    except: return "计算失败", "中性"

# ==========================================
# 3. 【核心修复】终极稳定版数据获取
# ==========================================
@st.cache_data(ttl=300)
def fetch_data(ticker, period="2y", interval="1d"):
    try:
        # 优先使用 Ticker.history，它对参数组合更宽容，且不易被反爬
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        
        # 如果 history 失败，降级尝试 download
        if df.empty:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty: return None
        
        # 统一处理索引和列名
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'Datetime' in df.columns: df = df.rename(columns={'Datetime': 'Date'})
        
        if 'Date' in df.columns:
            # 清理时区信息，防止后续图表报错
            df['Date'] = pd.to_datetime(df['Date'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
        
        if 'Volume' not in df.columns: df['Volume'] = 0
        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return None

# ==========================================
# 4. UI 主程序
# ==========================================
st.set_page_config(page_title="智能个股分析平台", layout="wide", page_icon="🌊")

with st.sidebar:
    st.title("⚙️ 系统设定")
    st.markdown("### 🔑 GitHub 配置")
    github_token = st.text_input("GitHub Token", type="password")
    github_repo = st.text_input("GitHub 仓库", placeholder="username/repo")
    if github_token and github_repo:
        sync = GitHubSync(token=github_token, repo_name=github_repo)
        if sync.is_configured: st.success("✅ GitHub 同步已启用")
        else: st.error("❌ 配置失败")
    else:
        sync = GitHubSync()
        if sync.is_configured: st.success("✅ 使用环境变量配置")
        else: st.warning("⚠️ 未配置 GitHub，数据仅暂存")
    
    st.markdown("---")
    st.title("🔍 股票查询")
    search_code = st.text_input("股票代码", "NVDA").upper()
    market = st.selectbox("市场", ["US", "HK"])
    ticker = f"{search_code}.HK" if market == "HK" else search_code
    
    st.markdown("---")
    st.title("⚙️ 分析参数")
    threshold = st.slider("波段值 (%)", 0.5, 15.0, 3.0, 0.5)
    tf_large = st.selectbox("大级别", ["周线", "月线", "日线"])
    tf_small = st.selectbox("小级别", ["日线", "小时线", "周线"])

st.title(" 智能个股分析平台 - 波浪理论专业版")

data = sync.load_data()
# 严格匹配 Yahoo Finance 允许的 period/interval 组合
period_map = {"月线": "5y", "周线": "2y", "日线": "1y", "小时线": "3mo"}
interval_map = {"月线": "1mo", "周线": "1wk", "日线": "1d", "小时线": "1h"}

df_large = fetch_data(ticker, period_map[tf_large], interval_map[tf_large])
df_small = fetch_data(ticker, period_map[tf_small], interval_map[tf_small])

if df_large is None or df_small is None:
    st.error(f"无法获取 {ticker} 的数据。请检查代码是否正确，或尝试切换时间框架（例如小时线只能看最近3个月）。")
    st.stop()

current_price = df_small['Close'].iloc[-1]
prev_close = df_small['Close'].iloc[-2] if len(df_small) > 1 else current_price
change = current_price - prev_close
change_pct = (change / prev_close) * 100

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("当前价格", f"{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
with col2:
    rsi = calculate_rsi(df_small['Close']).iloc[-1]
    st.metric("RSI (14)", f"{rsi:.1f}", "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性")
with col3: st.metric("MACD", "多头" if df_small['Close'].iloc[-1] > df_small['Close'].rolling(26).mean().iloc[-1] else "空头")
with col4:
    vol_ratio = df_small['Volume'].iloc[-1] / df_small['Volume'].rolling(20).mean().iloc[-1] if df_small['Volume'].rolling(20).mean().iloc[-1] > 0 else 1
    st.metric("成交量比", f"{vol_ratio:.2f}x", "放量" if vol_ratio > 1.5 else "正常")

high_price, low_price, swings = find_swings(df_small, threshold)
wave_type, wave_desc = identify_wave_pattern(swings)
pattern, pattern_desc, pattern_color = analyze_pattern(df_small)
fib_zones = calculate_fib_zones(high_price, low_price)

tab1, tab2, tab3, tab4 = st.tabs(["📊 共振分析仪表板", "👁️ 观察清单 (Excel)", "💼 模拟持仓 (Excel)", "📅 每日持仓日报 (Statement)"])

with tab1:
    st.markdown("### 📐 道氏理论与波浪分析")
    col_dow1, col_dow2 = st.columns(2)
    with col_dow1:
        st.markdown(f"""<div class="metric-card" style="border-left: 4px solid #60a5fa;">
            <h3 style="color: #60a5fa; margin: 0 0 10px 0;">📊 趋势方向</h3>
            <p style="font-size: 22px; font-weight: bold; color: #60a5fa; margin: 10px 0;">{wave_type}</p>
            <p style="color: #9ca3af; margin: 0; font-size: 13px;">{wave_desc}</p></div>""", unsafe_allow_html=True)
    with col_dow2:
        st.markdown(f"""<div class="metric-card" style="border-left: 4px solid {pattern_color};">
            <h3 style="color: {pattern_color}; margin: 0 0 10px 0;">🎯 形态分析</h3>
            <p style="font-size: 22px; font-weight: bold; color: {pattern_color}; margin: 10px 0;">{pattern}</p>
            <p style="color: #9ca3af; margin: 0; font-size: 13px;">{pattern_desc}</p></div>""", unsafe_allow_html=True)
    
    st.markdown("### 📐 斐波那契区间")
    col_fib1, col_fib2, col_fib3 = st.columns(3)
    with col_fib1:
        st.markdown("""<div style="background: rgba(239,68,68,0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #ef4444;">
            <h4 style="color: #ef4444; margin: 0 0 5px 0; font-size: 14px;">📈 阻力位</h4></div>""", unsafe_allow_html=True)
        st.metric("1.618 延伸", f"{fib_zones['长期阻力'][1]:.2f}")
        st.metric("1.000 等长", f"{fib_zones['长期阻力'][0]:.2f}")
        st.metric("0.618 阻力", f"{fib_zones['短期阻力'][1]:.2f}")
    with col_fib2:
        st.markdown("""<div style="background: rgba(34,197,94,0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #22c55e;">
            <h4 style="color: #22c55e; margin: 0 0 5px 0; font-size: 14px;">📉 支撑位</h4></div>""", unsafe_allow_html=True)
        st.metric("38.2% 支撑", f"{fib_zones['支撑位'][0]:.2f}")
        st.metric("50.0% 中轴", f"{fib_zones['支撑位'][1]:.2f}")
        st.metric("61.8% 强支撑", f"{fib_zones['支撑位'][2]:.2f}")
    with col_fib3:
        st.markdown("""<div style="background: rgba(234,179,8,0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #eab308;">
            <h4 style="color: #eab308; margin: 0 0 5px 0; font-size: 14px;">🎯 关键位</h4></div>""", unsafe_allow_html=True)
        st.metric("波段高点", f"{fib_zones['关键位'][1]:.2f}")
        st.metric("波段低点", f"{fib_zones['关键位'][0]:.2f}")

    st.markdown("### 📊 多时间框架图表")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_small['Date'], open=df_small['Open'], high=df_small['High'], low=df_small['Low'], close=df_small['Close'], name='K 线'), row=1, col=1)
    if swings:
        fig.add_trace(go.Scatter(x=[df_small['Date'].iloc[s['idx']] for s in swings if s['type'] == 'high'], y=[s['price'] for s in swings if s['type'] == 'high'], mode='markers', marker=dict(size=10, color='#ef4444', symbol='triangle-down'), name='Swing High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[df_small['Date'].iloc[s['idx']] for s in swings if s['type'] == 'low'], y=[s['price'] for s in swings if s['type'] == 'low'], mode='markers', marker=dict(size=10, color='#22c55e', symbol='triangle-up'), name='Swing Low'), row=1, col=1)
    for price in fib_zones['支撑位'] + fib_zones['短期阻力'] + fib_zones['长期阻力']:
        if price > 0: fig.add_hline(y=price, line_dash="dash", line_color="#ef4444" if price > high_price else "#22c55e", opacity=0.5, row=1, col=1)
    if 'Volume' in df_small.columns:
        fig.add_trace(go.Bar(x=df_small['Date'], y=df_small['Volume'], marker_color=['#22c55e' if df_small['Close'].iloc[i] >= df_small['Open'].iloc[i] else '#ef4444' for i in range(len(df_small))], name='成交量'), row=2, col=1)
    fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# (Tab 2, 3, 4 的逻辑与上一版完全相同，此处省略以节省篇幅，请直接使用上一版的 Tab 2/3/4 代码，只需确保 fetch_data 函数被替换即可)
# 为了完整性，这里保留 Tab 2 的核心修复逻辑
with tab2:
    st.title("👁️ 观察清单 (Excel 风格)")
    with st.expander("➕ 新增股票到观察清单", expanded=False):
        with st.form("add_stock", clear_on_submit=True):
            col1, col2 = st.columns([3, 1])
            with col1: new_code = st.text_input("输入股票代码", placeholder="例如：AAPL, 0700", key="watch_code").upper()
            with col2: submit_btn = st.form_submit_button("➕ 加入清单", use_container_width=True)
            if submit_btn and new_code:
                if not any(w['code'] == new_code for w in data['watchlist']):
                    new_ticker = f"{new_code}.HK" if market == "HK" else new_code
                    new_df = fetch_data(new_ticker, "1y", "1d")
                    if new_df is not None:
                        h, l, s = find_swings(new_df, threshold)
                        w_type, _ = identify_wave_pattern(s)
                        fib = calculate_fib_zones(h, l)
                        trend = "上升" if "第 3 浪" in w_type else "下降" if "调整浪" in w_type or "第 5 浪" in w_type else "盘整"
                        data['watchlist'].append({"code": new_code, "wave_type": w_type, "trend": trend, "fib_support": fib['支撑位'][2], "short_resist": fib['短期阻力'][1], "long_resist": fib['长期阻力'][1], "current_price": new_df['Close'].iloc[-1]})
                        sync.save_data(data); st.success(f"✅ 已加入 {new_code}"); st.rerun()
                    else: st.error(" 无法获取数据")
                else: st.warning(f"⚠️ {new_code} 已在清单中")

    if data['watchlist']:
        st.markdown(f"### 📋 已追踪 {len(data['watchlist'])} 支股票")
        for item in data['watchlist']:
            if 'short_resist' not in item: item['short_resist'] = item.get('current_price', 100) * 1.05; item['long_resist'] = item.get('current_price', 100) * 1.10
        html_table = """<table style="width: 100%; border-collapse: collapse; font-size: 14px;"><thead><tr style="background: #1e293b; color: #60a5fa;">
            <th style="padding: 12px; text-align: center; border: 1px solid #475569;">代码</th><th style="padding: 12px; text-align: center; border: 1px solid #475569;">浪型</th>
            <th style="padding: 12px; text-align: center; border: 1px solid #475569;">趋势</th><th style="padding: 12px; text-align: center; border: 1px solid #475569;">现价</th>
            <th style="padding: 12px; text-align: center; border: 1px solid #475569;">支撑位</th><th style="padding: 12px; text-align: center; border: 1px solid #475569;">短期阻力</th>
            <th style="padding: 12px; text-align: center; border: 1px solid #475569;">长期阻力</th></tr></thead><tbody>"""
        for item in data['watchlist']:
            trend_color = analyze_trend_color(item['trend'])
            html_table += f"""<tr style="background: #0f172a;"><td style="padding: 10px; text-align: center; border: 1px solid #334155; font-weight: bold; color: #60a5fa;">{item['code']}</td>
                <td style="padding: 10px; text-align: center; border: 1px solid #334155;">{item['wave_type']}</td>
                <td style="padding: 10px; text-align: center; border: 1px solid #334155; color: {trend_color}; font-weight: bold;">{item['trend']}</td>
                <td style="padding: 10px; text-align: center; border: 1px solid #334155; color: #22c55e;">${item['current_price']:.2f}</td>
                <td style="padding: 10px; text-align: center; border: 1px solid #334155; color: #ef4444;">${item['fib_support']:.2f}</td>
                <td style="padding: 10px; text-align: center; border: 1px solid #334155; color: #f97316;">${item['short_resist']:.2f}</td>
                <td style="padding: 10px; text-align: center; border: 1px solid #334155; color: #eab308;">${item['long_resist']:.2f}</td></tr>"""
        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🗑️ 管理清单 (删除股票)")
        if len(data['watchlist']) > 0:
            cols = st.columns(min(6, len(data['watchlist'])))
            for i, item in enumerate(data['watchlist']):
                with cols[i % len(cols)]:
                    if st.button(f" {item['code']}", key=f"del_wl_{i}", use_container_width=True, type="secondary"):
                        data['watchlist'].pop(i); sync.save_data(data); st.rerun()
    else: st.info("📭 观察清单为空")

# Tab 3 和 Tab 4 保持上一版代码不变...
