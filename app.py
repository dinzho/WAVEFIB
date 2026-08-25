import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import json
from github import Github, GithubException
import os

# ==========================================
# 1. 雲端同步管理 (GitHub 作為免費雲端資料庫)
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
            st.warning("⚠️ 未配置 GitHub Token，數據僅在本次瀏覽時暫存。")
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
            st.error(f"GitHub 讀取失敗: {e.data}")
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
        except Exception as e:
            st.error(f"GitHub 寫入失敗: {e}")

# ==========================================
# 2. 技術分析與波浪標記演算法
# ==========================================
def find_significant_swings(df, threshold_pct=3.0):
    """尋找顯著波段高低點 (ZigZag)"""
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
                trend = 1; sig_lows.append({'idx': i-1, 'price': last_low}); swing_points.append({'idx': i-1, 'type': 'low', 'price': last_low}); last_high = high
            elif (last_high - low) / last_high * 100 >= threshold_pct:
                trend = -1; sig_highs.append({'idx': i-1, 'price': last_high}); swing_points.append({'idx': i-1, 'type': 'high', 'price': last_high}); last_low = low
        elif trend == 1:
            if high > last_high: last_high = high
            elif (last_high - low) / last_high * 100 >= threshold_pct:
                sig_highs.append({'idx': i-1, 'price': last_high}); swing_points.append({'idx': i-1, 'type': 'high', 'price': last_high}); trend = -1; last_low = low
        elif trend == -1:
            if low < last_low: last_low = low
            elif (high - last_low) / last_low * 100 >= threshold_pct:
                sig_lows.append({'idx': i-1, 'price': last_low}); swing_points.append({'idx': i-1, 'type': 'low', 'price': last_low}); trend = 1; last_high = high

    swing_high = sig_highs[-1]['price'] if sig_highs else df['High'].max()
    swing_low = sig_lows[-1]['price'] if sig_lows else df['Low'].min()
    if swing_high < swing_low: swing_high, swing_low = swing_low, swing_high
    return swing_high, swing_low, swing_points

def identify_elliott_waves(swing_points):
    """
    嘗試識別 1-2-3-4-5 推動浪結構
    規則：2不低於1，3不是最短，4不重疊1
    """
    # 我們需要最近的 6 個轉折點 (低-高-低-高-低-高) 來構成 5 浪
    if len(swing_points) < 6: return []
    
    # 取最近 6 個點
    recent = swing_points[-6:]
    
    # 檢查是否為上升推動浪 (低-高-低-高-低-高)
    if recent[0]['type'] != 'low' or recent[1]['type'] != 'high': return []
    
    p1, p2, p3, p4, p5, p6 = recent[0]['price'], recent[1]['price'], recent[2]['price'], recent[3]['price'], recent[4]['price'], recent[5]['price']
    idx1, idx2, idx3, idx4, idx5, idx6 = recent[0]['idx'], recent[1]['idx'], recent[2]['idx'], recent[3]['idx'], recent[4]['idx'], recent[5]['idx']
    
    # 波浪鐵律檢查
    rule1 = p2 > p1 # 浪2不低於浪1起點
    rule2 = (p3 - p2) > (p2 - p1) # 浪3通常最長 (簡化版：浪3幅度 > 浪1幅度)
    rule3 = p4 > p1 # 浪4不重疊浪1 (浪4低點 > 浪1高點)
    
    if rule1 and rule3: # 規則 1 和 3 是鐵律，必須遵守
        return [
            {'idx': idx1, 'price': p1, 'wave': '1'},
            {'idx': idx2, 'price': p2, 'wave': '2'},
            {'idx': idx3, 'price': p3, 'wave': '3'},
            {'idx': idx4, 'price': p4, 'wave': '4'},
            {'idx': idx5, 'price': p5, 'wave': '5'},
        ]
    return []

# ==========================================
# 3. 數據獲取 (修復小時線 KeyError)
# ==========================================
@st.cache_data(ttl=120)
def fetch_multi_timeframe(ticker, tf_large, tf_small):
    period_map = {"月線": "5y", "週線": "2y", "日線": "1y", "小時線": "3mo"}
    interval_map = {"月線": "1mo", "週線": "1wk", "日線": "1d", "小時線": "1h"}
    
    def process_df(df):
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        # 【修復點】統一處理 Date/Datetime 欄位
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        elif 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df

    df_large = process_df(yf.download(ticker, period=period_map[tf_large], interval=interval_map[tf_large], progress=False))
    df_small = process_df(yf.download(ticker, period=period_map[tf_small], interval=interval_map[tf_small], progress=False))
    return df_large, df_small

# ==========================================
# 4. Streamlit UI 主程式
# ==========================================
st.set_page_config(page_title="智能個股分析平台 - 波浪共振", layout="wide", page_icon="🌊")
sync_manager = GitHubSync()
data = sync_manager.load_data()

st.title("🌊 智能個股分析平台 - 多時間框架波浪共振")

# --- Sidebar ---
st.sidebar.header("🔍 股票查詢")
search_code = st.sidebar.text_input("代碼 (如: NVDA, 0700)", "NVDA").upper().strip()
market = st.sidebar.selectbox("市場", ["US", "HK"], index=0)
ticker = f"{search_code}.HK" if market == "HK" else search_code

st.sidebar.markdown("---")
st.sidebar.header("️ 共振分析設定")
threshold_pct = st.sidebar.slider("波段確認閾值 (%)", 1.0, 15.0, 3.0, 0.5)

tf_large = st.sidebar.selectbox("大級別 (定趨勢)", ["週線", "月線", "日線"], index=0)
tf_small = st.sidebar.selectbox("小級別 (找進場)", ["日線", "小時線", "週線"], index=0)

# --- Main Content ---
tab1, tab2, tab3 = st.tabs([" 共振分析儀表板", "️ 觀察清單 (交易計畫)", " 模擬持倉"])

with tab1:
    st.subheader(f"{search_code} ({market}) - 多時間框架共振分析")
    
    with st.spinner(f"正在獲取 {tf_large} 與 {tf_small} 數據..."):
        df_large, df_small = fetch_multi_timeframe(ticker, tf_large, tf_small)
    
    if df_large is None or df_small is None:
        st.error("無法獲取數據，請檢查代碼或網路。")
    else:
        # 【修復點 3】顯示現價
        current_price = df_small['Close'].iloc[-1]
        prev_close = df_small['Close'].iloc[-2] if len(df_small) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        st.metric(" 當前價格", f"{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
        
        # 計算波浪結構
        high_price, low_price, swings_small = find_significant_swings(df_small, threshold_pct)
        waves = identify_elliott_waves(swings_small)
        
        # 繪製小級別圖表 (含 1-2-3-4-5 標記)
        st.markdown(f"### 🔍 {tf_small} 波浪結構與進場信號")
        fig_small = go.Figure()
        fig_small.add_trace(go.Candlestick(
            x=df_small['Date'], open=df_small['Open'], high=df_small['High'], 
            low=df_small['Low'], close=df_small['Close'], name='K線'
        ))
        
        # 【修復點 2】在圖上標記 1-2-3-4-5
        if waves:
            wave_x = [df_small['Date'].iloc[w['idx']] for w in waves]
            wave_y = [w['price'] for w in waves]
            wave_text = [f"浪{w['wave']}" for w in waves]
            
            fig_small.add_trace(go.Scatter(
                x=wave_x, y=wave_y, mode='markers+text',
                marker=dict(size=15, color='yellow', symbol='star', line=dict(width=2, color='black')),
                text=wave_text, textposition='top center',
                name='Elliott Waves', textfont=dict(size=14, color='white', family='Arial Black')
            ))
            st.success("✅ 偵測到符合鐵律的 1-2-3-4-5 推動浪結構！")
        else:
            st.info("️ 當前走勢未形成標準的 5 浪推動結構，可能處於調整浪或盤整中。")
            
        # 標記 Swing High/Low
        for point in swings_small[-3:]:
            fig_small.add_scatter(x=[df_small['Date'].iloc[point['idx']]], y=[point['price']], mode='markers', 
                                  marker=dict(size=8, color='red' if point['type']=='high' else 'green', symbol='diamond'))

        fig_small.update_layout(title=f"{search_code} {tf_small} 波浪分析", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_small, use_container_width=True)

with tab2:
    st.subheader("👁️ 觀察清單 (交易計畫)")
    st.info("💡 此清單已同步至 GitHub。請在此記錄你的浪型判斷與策略，手機電腦隨時查看。")
    
    with st.form("add_watchlist_form"):
        col1, col2 = st.columns(2)
        with col1:
            wl_code = st.text_input("代碼", key="wl_code").upper()
            wl_name = st.text_input("名稱", key="wl_name")
        with col2:
            wl_market = st.selectbox("市場", ["US", "HK"], key="wl_market")
            wl_wave = st.selectbox("現處浪型", ["第1浪(起漲)", "第2浪(回調)", "第3浪(主升)", "第4浪(盤整)", "第5浪(尾聲)", "A浪(下跌)", "B浪(反彈)", "C浪(主跌)", "不明確"], key="wl_wave")
        
        wl_strategy = st.text_input("操作策略 (如: 突破買入, 回調38.2%接多)", key="wl_strategy")
        wl_suggestion = st.text_area("具體建議/止損位", key="wl_suggestion", height=100)
        
        submitted = st.form_submit_button(" 加入觀察清單")
        if submitted and wl_code:
            if not any(item['code'] == wl_code for item in data['watchlist']):
                data['watchlist'].append({
                    "code": wl_code, "name": wl_name, "market": wl_market,
                    "wave": wl_wave, "strategy": wl_strategy, "suggestion": wl_suggestion
                })
                sync_manager.save_data(data)
                st.success(f"已將 {wl_code} 加入並同步！"); st.rerun()
            else: st.warning("該代碼已在清單中。")

    if data['watchlist']:
        for i, item in enumerate(data['watchlist']):
            with st.expander(f"📌 {item['code']} - {item['name']} ({item['market']}) - 浪型: {item['wave']}", expanded=True):
                st.markdown(f"** 現處浪型**: {item['wave']}")
                st.markdown(f"** 操作策略**: {item['strategy']}")
                st.markdown(f"**💡 具體建議**: {item['suggestion']}")
                if st.button(f"🗑️ 刪除 {item['code']}", key=f"del_wl_{i}"):
                    data['watchlist'].pop(i)
                    sync_manager.save_data(data)
                    st.rerun()
    else: st.info("觀察清單為空。")

with tab3:
    st.subheader("💼 模擬持倉")
    st.info("💡 你的持倉記錄已安全保存在 GitHub 雲端，換裝置登入即可看到。")
    
    with st.form("add_position_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            pos_code = st.text_input("代碼", key="pos_code").upper()
            pos_name = st.text_input("名稱", key="pos_name")
        with col2:
            pos_entry = st.number_input("買入價", min_value=0.0, step=0.01, key="pos_entry")
            pos_qty = st.number_input("數量 (股)", min_value=1, step=1, key="pos_qty")
        with col3:
            pos_dir = st.selectbox("方向", ["做多 Long", "做空 Short"], key="pos_dir")
            pos_current = st.number_input("當前價/目標價", min_value=0.0, step=0.01, key="pos_current")
            
        submitted = st.form_submit_button("💼 記錄持倉")
        if submitted and pos_code:
            pnl = (pos_current - pos_entry) * pos_qty if pos_dir == "做多 Long" else (pos_entry - pos_current) * pos_qty
            data['positions'].append({
                "code": pos_code, "name": pos_name, "entry": pos_entry, 
                "qty": pos_qty, "dir": pos_dir, "current": pos_current, "pnl": pnl
            })
            sync_manager.save_data(data)
            st.rerun()

    if data['positions']:
        total_pnl = sum(p['pnl'] for p in data['positions'])
        st.metric("總模擬盈虧", f"{total_pnl:+,.2f}", delta=f"{total_pnl:+,.2f}")
        
        for i, p in enumerate(data['positions']):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1: st.write(f"**{p['code']}** ({p['dir']})")
            with col2: st.write(f"買入: {p['entry']} x {p['qty']}")
            with col3: 
                color = "green" if p['pnl'] >= 0 else "red"
                st.markdown(f"<span style='color:{color}; font-weight:bold'>盈虧: {p['pnl']:+,.2f}</span>", unsafe_allow_html=True)
            with col4:
                if st.button("平倉/刪除", key=f"del_pos_{i}"):
                    data['positions'].pop(i)
                    sync_manager.save_data(data)
                    st.rerun()
    else: st.info("尚無持倉記錄。")
