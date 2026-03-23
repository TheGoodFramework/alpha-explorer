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
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .alpha-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        border-left: 4px solid #6366f1;
    }
    .category-momentum { border-left-color: #22c55e; }
    .category-mean-reversion { border-left-color: #f59e0b; }
    .category-volatility { border-left-color: #ef4444; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2a2a4a;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────

st.sidebar.title("📊 WQ101 Alpha Explorer")
st.sidebar.markdown("*WorldQuant 101 Formulaic Alphas*")
st.sidebar.divider()

# Mode selection
mode = st.sidebar.radio("Mode", ["📋 Catalog", "🔬 Single Alpha", "⚔️ Compare Alphas"], label_visibility="collapsed")

alpha_list = list_alphas()
alpha_nums = [a["num"] for a in alpha_list]
alpha_labels = {a["num"]: f"Alpha #{a['num']} — {a['name']}" for a in alpha_list}

# Common settings
st.sidebar.divider()
st.sidebar.subheader("Settings")

universe = st.sidebar.selectbox(
    "Universe",
    ["S&P 100 (Full)", "S&P 100 (Top 30)", "Custom"],
    index=1,
)

if universe == "Custom":
    custom_tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,NVDA,META,TSLA")
    tickers = [t.strip().upper() for t in custom_tickers.split(",")]
elif universe == "S&P 100 (Top 30)":
    tickers = SP100[:30]
else:
    tickers = SP100

col_start, col_end = st.sidebar.columns(2)
start_date = col_start.date_input("Start", datetime(2022, 1, 1))
end_date = col_end.date_input("End", datetime.now())


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


# ── Helper: Metric Cards ──────────────────────────────────────────

def show_metrics(metrics):
    cols = st.columns(4)
    key_metrics = ["Sharpe Ratio", "Annual Return", "Max Drawdown", "Win Rate"]
    colors = ["#6366f1", "#22c55e", "#ef4444", "#f59e0b"]
    for i, (key, color) in enumerate(zip(key_metrics, colors)):
        val = metrics.get(key, "N/A")
        cols[i].metric(key, val)

    with st.expander("All Metrics"):
        remaining = {k: v for k, v in metrics.items() if k not in key_metrics}
        cols2 = st.columns(4)
        for i, (k, v) in enumerate(remaining.items()):
            cols2[i % 4].metric(k, v)


# ── Helper: Charts ─────────────────────────────────────────────────

def plot_equity_drawdown(result, title="Alpha Performance"):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Equity curve
    eq = result["equity"]
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=eq.values,
            name="Portfolio", line=dict(color="#6366f1", width=2),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
        ),
        row=1, col=1,
    )

    # Benchmark (buy & hold equal weight)
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=[1] * len(eq),
            name="$1 Baseline", line=dict(color="#94a3b8", width=1, dash="dash"),
        ),
        row=1, col=1,
    )

    # Drawdown
    dd = result["drawdown"]
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values * 100,
            name="Drawdown", line=dict(color="#ef4444", width=1),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

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
        colorscale=[
            [0, "#ef4444"],
            [0.5, "#1a1a2e"],
            [1, "#22c55e"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in table.values],
        texttemplate="%{text}",
        textfont={"size": 11},
        hoverongaps=False,
    ))

    fig.update_layout(
        title="Monthly Returns (%)",
        template="plotly_dark",
        height=max(200, 60 * len(table)),
        margin=dict(l=50, r=20, t=40, b=20),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    return fig


def plot_return_distribution(daily_returns):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=daily_returns.values * 100,
        nbinsx=50,
        marker_color="#6366f1",
        opacity=0.8,
        name="Daily Returns",
    ))
    fig.update_layout(
        title="Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=20, t=40, b=20),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    return fig


# ── Catalog Mode ───────────────────────────────────────────────────

if mode == "📋 Catalog":
    st.title("Alpha Catalog")
    st.caption(f"{len(alpha_list)} alphas available — based on WorldQuant '101 Formulaic Alphas' (Kakushadze, 2015)")

    # Filters
    categories = sorted(set(a["category"] for a in alpha_list))
    col1, col2 = st.columns([1, 3])
    cat_filter = col1.selectbox("Category", ["All"] + categories)
    search = col2.text_input("Search", placeholder="e.g. volume, reversal, correlation...")

    filtered = alpha_list
    if cat_filter != "All":
        filtered = [a for a in filtered if a["category"] == cat_filter]
    if search:
        search_lower = search.lower()
        filtered = [a for a in filtered if
                    search_lower in a["description"].lower() or
                    search_lower in a["name"].lower() or
                    search_lower in a["formula"].lower()]

    # Card grid
    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(filtered):
                a = filtered[i + j]
                cat_class = f"category-{a['category']}"
                with col:
                    with st.container(border=True):
                        st.markdown(f"**Alpha #{a['num']}** — {a['name']}")
                        st.caption(f"🏷️ {a['category']}")
                        st.markdown(f"_{a['description']}_")
                        with st.expander("Formula"):
                            st.code(a["formula"], language="python")


# ── Single Alpha Mode ─────────────────────────────────────────────

elif mode == "🔬 Single Alpha":
    st.title("Single Alpha Backtest")

    selected_num = st.selectbox(
        "Select Alpha",
        alpha_nums,
        format_func=lambda x: alpha_labels[x],
    )

    info = ALPHA_CATALOG[selected_num]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### Alpha #{info['num']} — {info['name']}")
        st.markdown(f"_{info['description']}_")
        st.code(info["formula"], language="python")
    with col2:
        st.metric("Category", info["category"])
        st.metric("Universe", f"{len(tickers)} stocks")
        st.metric("Period", f"{start_date} → {end_date}")

    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Computing..."):
            result = run_single_alpha(selected_num, tuple(tickers), start_date, end_date)

        if result is None:
            st.error("Not enough data to backtest this alpha. Try a longer date range or larger universe.")
        else:
            # Metrics
            show_metrics(result["metrics"])

            # Charts
            st.plotly_chart(plot_equity_drawdown(result, f"Alpha #{selected_num} — {info['name']}"), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                heatmap = plot_monthly_heatmap(result["monthly_returns"])
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
            with col_b:
                st.plotly_chart(plot_return_distribution(result["daily_returns"]), use_container_width=True)

            # Holdings
            st.subheader("Current Holdings")
            col_l, col_s = st.columns(2)
            with col_l:
                st.markdown("**🟢 Long (Top Decile)**")
                st.dataframe(
                    result["top_holdings"].reset_index().rename(columns={"index": "Ticker", 0: "Rank Score"}),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_s:
                st.markdown("**🔴 Short (Bottom Decile)**")
                st.dataframe(
                    result["bottom_holdings"].reset_index().rename(columns={"index": "Ticker", 0: "Rank Score"}),
                    use_container_width=True,
                    hide_index=True,
                )


# ── Compare Mode ──────────────────────────────────────────────────

elif mode == "⚔️ Compare Alphas":
    st.title("Compare Alphas")

    selected = st.multiselect(
        "Select Alphas to Compare",
        alpha_nums,
        default=alpha_nums[:3],
        format_func=lambda x: alpha_labels[x],
        max_selections=6,
    )

    if len(selected) < 2:
        st.info("Select at least 2 alphas to compare.")
    elif st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        results = {}
        progress = st.progress(0)

        for i, num in enumerate(selected):
            progress.progress((i + 1) / len(selected), f"Computing Alpha #{num}...")
            result = run_single_alpha(num, tuple(tickers), start_date, end_date)
            if result:
                results[num] = result

        progress.empty()

        if not results:
            st.error("No alphas produced valid results.")
        else:
            # Overlay equity curves
            fig = go.Figure()
            colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#ec4899", "#06b6d4"]
            for i, (num, res) in enumerate(results.items()):
                eq = res["equity"]
                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values,
                    name=f"Alpha #{num}",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))

            fig.add_trace(go.Scatter(
                x=eq.index, y=[1] * len(eq),
                name="$1 Baseline",
                line=dict(color="#94a3b8", width=1, dash="dash"),
            ))

            fig.update_layout(
                title="Equity Curves Comparison",
                template="plotly_dark",
                height=500,
                margin=dict(l=50, r=20, t=40, b=20),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                yaxis_title="Portfolio Value ($)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics comparison table
            st.subheader("Metrics Comparison")
            comp_data = {}
            for num, res in results.items():
                comp_data[f"Alpha #{num}"] = res["metrics"]
            comp_df = pd.DataFrame(comp_data).T
            st.dataframe(comp_df, use_container_width=True)

            # Radar chart
            st.subheader("Factor Profile")
            radar_metrics = ["Sharpe Ratio", "Win Rate", "Annual Return"]
            fig_radar = go.Figure()
            for i, (num, res) in enumerate(results.items()):
                vals = []
                for m in radar_metrics:
                    v = res["metrics"][m]
                    v = float(v.strip("%")) / 100 if "%" in v else float(v)
                    vals.append(v)
                vals.append(vals[0])  # close the polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=f"Alpha #{num}",
                    line_color=colors[i % len(colors)],
                    opacity=0.6,
                ))

            fig_radar.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=50, r=50, t=20, b=20),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                polar=dict(
                    bgcolor="#0e1117",
                    radialaxis=dict(visible=True),
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.caption(
    "Based on *101 Formulaic Alphas* by Zura Kakushadze (2015). "
    "Educational use only — not financial advice. "
    "VWAP approximated as (H+L+C)/3."
)
