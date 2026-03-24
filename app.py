"""
WQ101 Alpha Explorer — Consumer-grade backtesting tool for WorldQuant 101 Formulaic Alphas.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from alphas import ALPHA_CATALOG, list_alphas, compute_alpha
from engine import fetch_data, backtest_alpha, SP100


# ── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="WQ101 Alpha Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Palette ──────────────────────────────────────────────────
       #FE9321  orange   — primary / equity curve / CTA
       #6FE3CC  mint     — positive / good
       #185D7A  teal     — surfaces / borders
       #C8DB2A  lime     — warning / neutral
       #EF4687  pink     — negative / bad / drawdown
    ── */

    /* ── Metric cards (st.metric fallback) ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d2535 0%, #112f45 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #185D7A;
    }
    div[data-testid="stMetric"] label {
        color: #7ec8d8 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Color-coded metric cards ── */
    .metric-good {
        background: linear-gradient(135deg, #082820 0%, #051f18 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1a5a40;
        text-align: center;
    }
    .metric-bad {
        background: linear-gradient(135deg, #2a0a1a 0%, #1f0813 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #6b1535;
        text-align: center;
    }
    .metric-neutral {
        background: linear-gradient(135deg, #0d2535 0%, #112f45 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #185D7A;
        text-align: center;
    }
    .metric-warn {
        background: linear-gradient(135deg, #1e2208 0%, #161a05 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #4a5010;
        text-align: center;
    }
    .metric-value-good    { font-size: 26px; font-weight: 700; color: #6FE3CC; }
    .metric-value-bad     { font-size: 26px; font-weight: 700; color: #EF4687; }
    .metric-value-warn    { font-size: 26px; font-weight: 700; color: #C8DB2A; }
    .metric-value-neutral { font-size: 26px; font-weight: 700; color: #e8f4f8; }
    .metric-label-style {
        font-size: 11px;
        color: #7ec8d8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .metric-hint {
        font-size: 10px;
        color: #4a8a9a;
        margin-top: 2px;
    }

    /* ── Category pills ── */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .pill-momentum       { background: #0d2535; color: #FE9321; border: 1px solid #FE9321; }
    .pill-mean-reversion { background: #0d2535; color: #6FE3CC; border: 1px solid #6FE3CC; }
    .pill-volatility     { background: #0d2535; color: #EF4687; border: 1px solid #EF4687; }
    .pill-default        { background: #0d2535; color: #C8DB2A; border: 1px solid #C8DB2A; }

    /* ── Alpha cards ── */
    .alpha-num {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        /* readable on both light and dark via native Streamlit caption */
    }
    /* formula block — always forced-dark so colors are predictable */
    .formula-block {
        background: #0d2535;
        border: 1px solid #185D7A;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0 4px 0;
        overflow-x: auto;
    }
    .formula-block code {
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 11px;
        color: #6FE3CC;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.6;
    }
    /* section header matching Factor Profile style */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 4px 0;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #112f45 0%, #0d2535 60%, #07181f 100%);
        border-radius: 16px;
        padding: 32px 36px;
        border: 1px solid #185D7A;
        margin-bottom: 24px;
    }
    .hero h1 { color: #e8f4f8; font-size: 28px; margin: 0 0 8px 0; }
    .hero p  { color: #7ec8d8; font-size: 15px; margin: 0; line-height: 1.6; }
    .hero p strong { color: #FE9321; }
    .hero-badge {
        display: inline-block;
        background: #185D7A;
        color: #6FE3CC;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
        border: 1px solid #6FE3CC;
    }

    /* ── Result count badge ── */
    .result-count {
        display: inline-block;
        background: #0d2535;
        color: #7ec8d8;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid #185D7A;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #4a8a9a;
    }
    .empty-state-icon { font-size: 48px; margin-bottom: 12px; }
    .empty-state-text { font-size: 16px; font-weight: 500; color: #7ec8d8; }
    .empty-state-sub  { font-size: 14px; color: #4a8a9a; margin-top: 6px; }

    /* ── Holdings header ── */
    .holdings-long  { color: #6FE3CC; font-weight: 600; }
    .holdings-short { color: #EF4687; font-weight: 600; }

    /* ── Ticker chip ── */
    .ticker-chip {
        display: inline-block;
        background: #112f45;
        color: #e8f4f8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 600;
        font-family: monospace;
        border: 1px solid #185D7A;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "📋 Catalog"
if "jump_to_alpha" not in st.session_state:
    st.session_state.jump_to_alpha = None


# ── Sidebar ────────────────────────────────────────────────────────

st.sidebar.title("📊 WQ101 Alpha Explorer")
st.sidebar.markdown("*WorldQuant 101 Formulaic Alphas*")
st.sidebar.divider()

# Mode selection (uses session state so catalog cards can switch modes)
mode = st.sidebar.radio(
    "Mode",
    ["📋 Catalog", "🔬 Single Alpha", "⚔️ Compare Alphas"],
    index=["📋 Catalog", "🔬 Single Alpha", "⚔️ Compare Alphas"].index(st.session_state.mode),
    label_visibility="collapsed",
)
st.session_state.mode = mode

alpha_list = list_alphas()
alpha_nums = [a["num"] for a in alpha_list]
alpha_labels = {a["num"]: f"Alpha #{a['num']} — {a['name']}" for a in alpha_list}

# Settings — only shown when a backtest is needed
if mode != "📋 Catalog":
    st.sidebar.divider()
    st.sidebar.subheader("Settings")

    universe = st.sidebar.selectbox(
        "Universe",
        ["S&P 100 (Full)", "S&P 100 (Top 30)", "Custom"],
        index=1,
    )

    if universe == "Custom":
        custom_tickers = st.sidebar.text_input(
            "Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,NVDA,META,TSLA"
        )
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    elif universe == "S&P 100 (Top 30)":
        tickers = SP100[:30]
    else:
        tickers = SP100

    col_start, col_end = st.sidebar.columns(2)
    start_date = col_start.date_input("Start", datetime(2022, 1, 1))
    end_date = col_end.date_input("End", datetime.now())
else:
    # Defaults so data loading functions don't error if needed
    tickers = SP100[:30]
    start_date = datetime(2022, 1, 1).date()
    end_date = datetime.now().date()


# ── Data Loading ───────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data(tickers_tuple, start, end):
    return fetch_data(list(tickers_tuple), start=str(start), end=str(end))


@st.cache_data(ttl=3600, show_spinner="Computing alpha signals...")
def run_single_alpha(alpha_num, tickers_tuple, start, end):
    data = load_data(tickers_tuple, start, end)
    signals = compute_alpha(alpha_num, data)
    result = backtest_alpha(signals, data["returns"])
    return result


# ── Helper: Category Pill ─────────────────────────────────────────

def category_pill(category):
    cls = {
        "momentum":       "pill-momentum",
        "mean-reversion": "pill-mean-reversion",
        "volatility":     "pill-volatility",
    }.get(category, "pill-default")
    return f'<span class="pill {cls}">{category}</span>'


# ── Helper: Color-Coded Metric Cards ─────────────────────────────

def _metric_class_and_val(key, raw_val):
    """Return (card_class, value_class, display_val, hint)."""
    try:
        if "%" in raw_val:
            num = float(raw_val.strip("%")) / 100
        else:
            num = float(raw_val.split()[0])
    except Exception:
        return "metric-neutral", "metric-value-neutral", raw_val, ""

    hints = {
        "Sharpe Ratio":      ("Target: > 1.0", lambda v: "good" if v >= 1.0 else ("warn" if v >= 0.5 else "bad")),
        "Annual Return":     ("Annualized",     lambda v: "good" if v > 0 else "bad"),
        "Max Drawdown":      ("Lower is better", lambda v: "warn" if v > -0.1 else ("bad" if v < -0.2 else "warn")),
        "Win Rate":          ("Days > 0",        lambda v: "good" if v >= 0.5 else "warn"),
        "Calmar Ratio":      ("Return/Drawdown", lambda v: "good" if v >= 1.0 else ("warn" if v >= 0.5 else "bad")),
        "Annual Volatility": ("Annualized vol",  lambda v: "warn" if v > 0.15 else "neutral"),
    }

    hint_text, classifier = hints.get(key, ("", lambda v: "neutral"))
    sentiment = classifier(num)
    card_cls  = f"metric-{sentiment}"
    val_cls   = f"metric-value-{sentiment}"
    return card_cls, val_cls, raw_val, hint_text


def show_metrics(metrics):
    primary_keys = ["Sharpe Ratio", "Annual Return", "Max Drawdown", "Win Rate"]
    cols = st.columns(4)
    for i, key in enumerate(primary_keys):
        val = metrics.get(key, "N/A")
        card_cls, val_cls, display, hint = _metric_class_and_val(key, val)
        cols[i].markdown(
            f"""<div class="{card_cls}">
                    <div class="{val_cls}">{display}</div>
                    <div class="metric-label-style">{key}</div>
                    <div class="metric-hint">{hint}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("All Metrics"):
        remaining = {k: v for k, v in metrics.items() if k not in primary_keys}
        cols2 = st.columns(4)
        for i, (k, v) in enumerate(remaining.items()):
            card_cls, val_cls, display, hint = _metric_class_and_val(k, v)
            cols2[i % 4].markdown(
                f"""<div class="{card_cls}" style="margin-bottom:8px">
                        <div class="{val_cls}">{display}</div>
                        <div class="metric-label-style">{k}</div>
                        <div class="metric-hint">{hint}</div>
                    </div>""",
                unsafe_allow_html=True,
            )


# ── Helper: Charts ─────────────────────────────────────────────────

BG       = "#07181f"
BG_CARD  = "#0d2535"
GRID     = "#112f45"
ORANGE   = "#FE9321"
MINT     = "#6FE3CC"
TEAL     = "#185D7A"
LIME     = "#C8DB2A"
PINK     = "#EF4687"
TEXT     = "#e8f4f8"
MUTED    = "#7ec8d8"
COMPARE_COLORS = [ORANGE, MINT, LIME, PINK, TEAL, "#ffffff"]


def plot_equity_drawdown(result, title="Alpha Performance"):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    eq = result["equity"]
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=eq.values,
            name="Portfolio", line=dict(color=ORANGE, width=2.5),
            fill="tozeroy", fillcolor="rgba(254,147,33,0.12)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Value: $%{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=[1] * len(eq),
            name="$1 Baseline", line=dict(color=MUTED, width=1, dash="dash"),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    dd = result["drawdown"]
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values * 100,
            name="Drawdown", line=dict(color=PINK, width=1.5),
            fill="tozeroy", fillcolor="rgba(239,70,135,0.2)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Drawdown: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT)),
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center",
                    font=dict(color=MUTED)),
        paper_bgcolor=BG,
        plot_bgcolor=BG_CARD,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", title_font=dict(color=MUTED),
                     row=1, col=1, gridcolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(title_text="Drawdown (%)", title_font=dict(color=MUTED),
                     row=2, col=1, gridcolor=GRID, tickfont=dict(color=MUTED))
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED))
    return fig


def plot_monthly_heatmap(monthly_returns):
    if monthly_returns is None or len(monthly_returns) == 0:
        return None

    df = monthly_returns.copy()
    df.index = pd.to_datetime(df.index)
    pivot = pd.DataFrame({
        "year": df.index.year,
        "month": df.index.month,
        "return": df.values * 100,
    })
    table = pivot.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=table.values,
        x=table.columns,
        y=table.index.astype(str),
        colorscale=[[0, PINK], [0.5, BG_CARD], [1, MINT]],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in table.values],
        texttemplate="%{text}",
        textfont={"size": 11, "color": TEXT},
        hoverongaps=False,
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Monthly Returns (%)", font=dict(size=14, color=TEXT)),
        template="plotly_dark",
        height=max(200, 60 * len(table)),
        margin=dict(l=50, r=20, t=40, b=20),
        paper_bgcolor=BG,
        plot_bgcolor=BG_CARD,
        xaxis=dict(tickfont=dict(color=MUTED)),
        yaxis=dict(tickfont=dict(color=MUTED)),
    )
    return fig


def plot_return_distribution(daily_returns):
    mean_r = daily_returns.mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=daily_returns.values * 100,
        nbinsx=50,
        marker_color=TEAL,
        marker_line=dict(color=MINT, width=0.5),
        opacity=0.9,
        name="Daily Returns",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=mean_r, line_dash="dash", line_color=ORANGE,
        annotation_text=f"Mean: {mean_r:.2f}%",
        annotation_position="top right",
        annotation_font_color=ORANGE,
    )
    fig.update_layout(
        title=dict(text="Return Distribution", font=dict(size=14, color=TEXT)),
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=20, t=40, b=20),
        paper_bgcolor=BG,
        plot_bgcolor=BG_CARD,
        xaxis=dict(tickfont=dict(color=MUTED), gridcolor=GRID),
        yaxis=dict(tickfont=dict(color=MUTED), gridcolor=GRID),
    )
    return fig


def plot_correlation_matrix(results):
    """Correlation matrix of daily returns across selected alphas."""
    ret_df = pd.DataFrame({
        f"Alpha #{num}": res["daily_returns"]
        for num, res in results.items()
    })
    corr = ret_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, PINK], [0.5, BG_CARD], [1, MINT]],
        zmin=-1, zmax=1, zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont={"size": 11, "color": TEXT},
        hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=BG,
        plot_bgcolor=BG_CARD,
        xaxis=dict(tickfont=dict(color=MUTED)),
        yaxis=dict(tickfont=dict(color=MUTED)),
    )
    return fig


# ── Catalog Mode ───────────────────────────────────────────────────

if mode == "📋 Catalog":
    # Hero banner
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">📖 WorldQuant 101 Formulaic Alphas · Kakushadze (2015)</div>
        <h1>Alpha Catalog</h1>
        <p>
            Browse and explore quantitative trading signals derived from price and volume data.
            Select an alpha to read its logic, then run a backtest in <strong>Single Alpha</strong> mode
            to see real performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    categories = sorted(set(a["category"] for a in alpha_list))
    col1, col2, col3 = st.columns([1, 3, 1])
    cat_filter = col1.selectbox("Category", ["All"] + categories)
    search = col2.text_input("Search", placeholder="e.g. volume, reversal, correlation...")
    col3.markdown("<br>", unsafe_allow_html=True)

    filtered = alpha_list
    if cat_filter != "All":
        filtered = [a for a in filtered if a["category"] == cat_filter]
    if search:
        search_lower = search.lower()
        filtered = [a for a in filtered if
                    search_lower in a["description"].lower() or
                    search_lower in a["name"].lower() or
                    search_lower in a["formula"].lower()]

    # Result count
    st.markdown(
        f'<span class="result-count">Showing {len(filtered)} of {len(alpha_list)} alphas</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Empty state
    if not filtered:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text">No alphas match your search</div>
            <div class="empty-state-sub">Try a different keyword or clear the category filter</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Card grid
        for i in range(0, len(filtered), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(filtered):
                    a = filtered[i + j]
                    with col:
                        with st.container(border=True):
                            # Header row: number + pill
                            h_col, p_col = st.columns([1, 2])
                            h_col.caption(f"Alpha #{a['num']}")
                            p_col.markdown(category_pill(a["category"]), unsafe_allow_html=True)

                            # Name — native bold, theme-aware (dark on white / light on dark)
                            st.markdown(f"**{a['name']}**")
                            # Description — native text, theme-aware
                            st.markdown(
                                f"<span style='font-size:13px; line-height:1.5;'>{a['description']}</span>",
                                unsafe_allow_html=True,
                            )

                            # Formula — always-visible, forced-dark block, smaller font
                            formula_safe = (
                                a["formula"]
                                .replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                            )
                            st.markdown(
                                f'<div class="formula-block"><code>{formula_safe}</code></div>',
                                unsafe_allow_html=True,
                            )

                            if st.button("Backtest →", key=f"bt_{a['num']}", use_container_width=True):
                                st.session_state.jump_to_alpha = a["num"]
                                st.session_state.mode = "🔬 Single Alpha"
                                st.rerun()


# ── Single Alpha Mode ─────────────────────────────────────────────

elif mode == "🔬 Single Alpha":
    st.title("Single Alpha Backtest")
    st.caption("Select an alpha, configure the universe and date range in the sidebar, then run.")

    # Pre-select if jumping from catalog
    default_idx = 0
    if st.session_state.jump_to_alpha is not None:
        try:
            default_idx = alpha_nums.index(st.session_state.jump_to_alpha)
        except ValueError:
            default_idx = 0
        st.session_state.jump_to_alpha = None

    selected_num = st.selectbox(
        "Select Alpha",
        alpha_nums,
        index=default_idx,
        format_func=lambda x: alpha_labels[x],
    )

    info = ALPHA_CATALOG[selected_num]

    # Run button — placed prominently before details
    run = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    # Alpha detail card
    with st.expander("Alpha Details", expanded=True):
        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            st.markdown(f"**{info['description']}**")
            st.code(info["formula"], language="python")
        with d_col2:
            st.markdown(category_pill(info["category"]), unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="margin-top:12px; display:flex; flex-direction:column; gap:8px;">
                    <div style="background:#0d2535; border-radius:8px; padding:10px 14px; border:1px solid #185D7A;">
                        <div style="font-size:10px; color:#7ec8d8; text-transform:uppercase; letter-spacing:1px;">Universe</div>
                        <div style="font-size:15px; font-weight:600; color:#e8f4f8; margin-top:2px;">{len(tickers)} stocks</div>
                    </div>
                    <div style="background:#0d2535; border-radius:8px; padding:10px 14px; border:1px solid #185D7A;">
                        <div style="font-size:10px; color:#7ec8d8; text-transform:uppercase; letter-spacing:1px;">Period</div>
                        <div style="font-size:15px; font-weight:600; color:#e8f4f8; margin-top:2px;">{(end_date - start_date).days} days</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Run backtest
    if run:
        with st.spinner("Computing alpha signals and running backtest..."):
            result = run_single_alpha(selected_num, tuple(tickers), start_date, end_date)

        if result is None:
            st.error(
                "⚠️ **Not enough data to run this backtest.**\n\n"
                "**Try:**\n"
                "- Extending the date range (at least 2 years recommended)\n"
                "- Using a larger universe (S&P 100 Full instead of Top 30)\n"
                "- Some alphas require 250+ days of history for their lookback windows"
            )
        else:
            st.success(f"Backtest complete — {result['metrics']['Total Days']} trading days")

            # Metrics
            show_metrics(result["metrics"])
            st.divider()

            # Charts
            st.plotly_chart(
                plot_equity_drawdown(result, f"Alpha #{selected_num} — {info['name']}"),
                use_container_width=True,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                heatmap = plot_monthly_heatmap(result["monthly_returns"])
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
                else:
                    st.info("Not enough history for monthly heatmap.")
            with col_b:
                st.plotly_chart(
                    plot_return_distribution(result["daily_returns"]),
                    use_container_width=True,
                )

            # Holdings
            st.divider()
            st.subheader("Latest Holdings Snapshot")
            st.caption(f"As of {result['equity'].index[-1].strftime('%b %d, %Y')}")

            col_l, col_s = st.columns(2)
            with col_l:
                st.markdown('<span class="holdings-long">▲ Long — Top Decile</span>', unsafe_allow_html=True)
                long_df = result["top_holdings"].reset_index()
                long_df.columns = ["Ticker", "Rank Score"]
                long_df["Rank Score"] = long_df["Rank Score"].round(4)
                st.dataframe(long_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Download Long", long_df.to_csv(index=False),
                    file_name=f"alpha{selected_num}_long.csv", mime="text/csv",
                    key="dl_long",
                )
            with col_s:
                st.markdown('<span class="holdings-short">▼ Short — Bottom Decile</span>', unsafe_allow_html=True)
                short_df = result["bottom_holdings"].reset_index()
                short_df.columns = ["Ticker", "Rank Score"]
                short_df["Rank Score"] = short_df["Rank Score"].round(4)
                st.dataframe(short_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Download Short", short_df.to_csv(index=False),
                    file_name=f"alpha{selected_num}_short.csv", mime="text/csv",
                    key="dl_short",
                )


# ── Compare Mode ──────────────────────────────────────────────────

elif mode == "⚔️ Compare Alphas":
    st.title("Compare Alphas")
    st.caption("Select 2–6 alphas to compare their equity curves, metrics, and return correlation.")

    selected = st.multiselect(
        "Select Alphas to Compare",
        alpha_nums,
        default=alpha_nums[:3],
        format_func=lambda x: alpha_labels[x],
        max_selections=6,
    )

    if len(selected) < 2:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">⚔️</div>
            <div class="empty-state-text">Select at least 2 alphas to compare</div>
            <div class="empty-state-sub">You can compare up to 6 alphas side by side</div>
        </div>
        """, unsafe_allow_html=True)

    elif st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        results = {}
        failed = []
        progress = st.progress(0, text="Starting...")

        for i, num in enumerate(selected):
            progress.progress((i + 1) / len(selected), f"Computing Alpha #{num}...")
            result = run_single_alpha(num, tuple(tickers), start_date, end_date)
            if result:
                results[num] = result
            else:
                failed.append(num)

        progress.empty()

        if failed:
            st.warning(f"Alpha(s) {failed} returned no results (insufficient data) and were skipped.")

        if not results:
            st.error(
                "No alphas produced valid results.\n\n"
                "Try extending the date range or switching to a larger universe."
            )
        else:
            # Overlay equity curves
            colors = COMPARE_COLORS
            fig = go.Figure()
            last_eq = None
            for i, (num, res) in enumerate(results.items()):
                eq = res["equity"]
                last_eq = eq
                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values,
                    name=f"Alpha #{num}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>Alpha #{num}</b><br>%{{x|%b %d, %Y}}<br>$%{{y:.3f}}<extra></extra>",
                ))

            if last_eq is not None:
                fig.add_trace(go.Scatter(
                    x=last_eq.index, y=[1] * len(last_eq),
                    name="$1 Baseline",
                    line=dict(color=MUTED, width=1, dash="dash"),
                    hoverinfo="skip",
                ))

            fig.update_layout(
                title=dict(text="Equity Curves", font=dict(size=16, color=TEXT)),
                template="plotly_dark",
                height=500,
                margin=dict(l=50, r=20, t=50, b=20),
                paper_bgcolor=BG,
                plot_bgcolor=BG_CARD,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5,
                            xanchor="center", font=dict(color=MUTED)),
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
            )
            fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED))
            fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=MUTED))
            st.plotly_chart(fig, use_container_width=True)

            # Metrics comparison table
            st.subheader("Metrics Comparison")
            st.caption("Click any column header to sort.")
            comp_data = {}
            for num, res in results.items():
                comp_data[f"Alpha #{num}"] = res["metrics"]
            comp_df = pd.DataFrame(comp_data).T
            comp_df.index.name = "Alpha"

            # Color Annual Return: mint (green) if positive, pink (red) if negative
            def _color_return(val):
                try:
                    v = float(str(val).strip("%")) / 100 if "%" in str(val) else float(val)
                    return f"color: {MINT}; font-weight: 600" if v > 0 else f"color: {PINK}; font-weight: 600"
                except Exception:
                    return ""

            styled_comp = comp_df.style
            if "Annual Return" in comp_df.columns:
                styled_comp = styled_comp.map(_color_return, subset=["Annual Return"])
            st.dataframe(styled_comp, use_container_width=True)

            # Correlation matrix + radar side by side
            col_corr, col_radar = st.columns(2)

            with col_corr:
                # Header matches Factor Profile alignment
                st.subheader("Return Correlation Matrix")
                st.caption("Pearson correlation of daily returns — lower = better diversification.")
                st.plotly_chart(
                    plot_correlation_matrix(results),
                    use_container_width=True,
                )

            with col_radar:
                st.subheader("Factor Profile")
                st.caption("Values normalised 0–1 across selected alphas for shape comparison.")
                radar_metrics = ["Sharpe Ratio", "Win Rate", "Annual Return"]

                # Extract raw values for all alphas first, then min-max normalise per metric
                raw_vals = {}
                for num, res in results.items():
                    row = []
                    for m in radar_metrics:
                        v = res["metrics"][m]
                        v = float(v.strip("%")) / 100 if "%" in v else float(v)
                        row.append(v)
                    raw_vals[num] = row

                # Per-metric min/max for normalisation
                metric_min = [min(raw_vals[n][j] for n in raw_vals) for j in range(len(radar_metrics))]
                metric_max = [max(raw_vals[n][j] for n in raw_vals) for j in range(len(radar_metrics))]

                def normalise(v, lo, hi):
                    return (v - lo) / (hi - lo) if hi > lo else 0.5

                fig_radar = go.Figure()
                for i, (num, res) in enumerate(results.items()):
                    norm = [normalise(raw_vals[num][j], metric_min[j], metric_max[j])
                            for j in range(len(radar_metrics))]
                    norm.append(norm[0])  # close the polygon
                    # Build hover text showing the actual raw value
                    raw = raw_vals[num]
                    customdata = [
                        f"{radar_metrics[j]}: {raw[j]:.2f}" for j in range(len(radar_metrics))
                    ] + [f"{radar_metrics[0]}: {raw[0]:.2f}"]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=norm,
                        theta=radar_metrics + [radar_metrics[0]],
                        fill="toself",
                        name=f"Alpha #{num}",
                        line_color=colors[i % len(colors)],
                        opacity=0.65,
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=customdata,
                    ))

                fig_radar.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=30, r=30, t=20, b=60),
                    paper_bgcolor=BG,
                    plot_bgcolor=BG,
                    polar=dict(
                        bgcolor=BG_CARD,
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                            ticktext=["Low", "", "Mid", "", "High"],
                            gridcolor=GRID,
                            tickfont=dict(size=9, color=MUTED),
                            linecolor=TEAL,
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=12, color=TEXT),
                            linecolor=TEAL,
                            gridcolor=GRID,
                        ),
                    ),
                    legend=dict(orientation="h", y=-0.15, font=dict(size=11, color=MUTED)),
                )
                st.plotly_chart(fig_radar, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.caption(
    "Based on *101 Formulaic Alphas* by Zura Kakushadze (2015). "
    "Educational use only — not financial advice. "
    "VWAP approximated as (H+L+C)/3."
)
