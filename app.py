import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ===============================
# CONFIG
# ===============================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1msc3DBtNx-xx04CdoG7-dCNkz4wA2sxwKBGsaJHTaI4/export?format=csv&gid=1459153692"
POSITIVE_MARK = 1
NEGATIVE_MARK = 0.25

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")
st.title("Performance Dashboard")

# ===============================
# TOP CONTROLS
# ===============================
col1, col2 = st.columns([1, 4])

with col1:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with col2:
    auto_refresh = st.checkbox("Auto Refresh (every 5 min)")

if auto_refresh:
    st_autorefresh(interval=300000, key="datarefresh")

# ===============================
# DATA LOADING
# ===============================
@st.cache_data(ttl=300)
def load_data():
    try:
        df = pd.read_csv(SHEET_URL, header=[0, 1])
    except Exception as e:
        st.error(f"Failed to load Google Sheet: {e}")
        return pd.DataFrame(), []

    if df.empty:
        return pd.DataFrame(), []

    # Clean MultiIndex levels aggressively
    level0 = df.columns.get_level_values(0).astype(str).str.strip().str.lower()
    level1 = df.columns.get_level_values(1).astype(str).str.strip().str.lower()

    df.columns = pd.MultiIndex.from_arrays([level0, level1])

    # Identify date column
    if "date" not in level0:
        st.error("No 'Date' column found.")
        return pd.DataFrame(), []

    # Extract subjects from level0 excluding date
    subjects = sorted(set(level0) - {"date"})

    # Flatten properly
    flat_cols = []
    for l0, l1 in df.columns:
        if l0 == "date":
            flat_cols.append("test_number")
        else:
            flat_cols.append(f"{l0}_{l1}")

    df.columns = flat_cols

    # Convert numeric columns
    for col in df.columns:
        if col != "test_number":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["test_number"])
    df = df.sort_values("test_number").reset_index(drop=True)

    return df, subjects


# ===============================
# SAFE DIVISION
# ===============================
def safe_divide(n, d):
    return np.where(d == 0, 0, n / d)


# ===============================
# CALCULATIONS
# ===============================
def compute_metrics(df, subjects):

    df = df.copy()

    for s in subjects:

        attempted = df.get(f"{s}_attempted", 0)
        wrong = df.get(f"{s}_wrong", 0)
        unattempt = df.get(f"{s}_unattempt", 0)

        df[f"{s}_total"] = attempted + unattempt
        df[f"{s}_correct"] = attempted - wrong

        df[f"{s}_accuracy"] = safe_divide(df[f"{s}_correct"], attempted)
        df[f"{s}_attempt_ratio"] = safe_divide(attempted, df[f"{s}_total"])

        net = df[f"{s}_correct"] * POSITIVE_MARK - wrong * NEGATIVE_MARK
        df[f"{s}_net_score"] = net
        df[f"{s}_normalized_score"] = safe_divide(net, df[f"{s}_total"])

    # Overall
    attempted_cols = [f"{s}_attempted" for s in subjects]
    wrong_cols = [f"{s}_wrong" for s in subjects]
    unattempt_cols = [f"{s}_unattempt" for s in subjects]

    df["total_attempted"] = df[attempted_cols].sum(axis=1)
    df["total_wrong"] = df[wrong_cols].sum(axis=1)
    df["total_unattempt"] = df[unattempt_cols].sum(axis=1)

    df["total_correct"] = df["total_attempted"] - df["total_wrong"]
    df["total_questions"] = df["total_attempted"] + df["total_unattempt"]

    df["overall_accuracy"] = safe_divide(
        df["total_correct"], df["total_attempted"]
    )

    df["overall_attempt_ratio"] = safe_divide(
        df["total_attempted"], df["total_questions"]
    )

    overall_net = df["total_correct"] * POSITIVE_MARK - df["total_wrong"] * NEGATIVE_MARK
    df["overall_net_score"] = overall_net
    df["overall_normalized_score"] = safe_divide(
        overall_net, df["total_questions"]
    )

    return df


# ===============================
# LINE CHARTS
# ===============================
def create_line_charts(df, subjects):

    metrics = [
        ("accuracy", "Accuracy"),
        ("attempt_ratio", "Attempt Ratio"),
        ("normalized_score", "Normalized Score"),
    ]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[m[1] for m in metrics])

    for idx, (key, title) in enumerate(metrics):
        col = idx + 1
        show_legend = idx == 0

        for s in subjects:
            fig.add_trace(
                go.Scatter(
                    x=df["test_number"],
                    y=df[f"{s}_{key}"],
                    mode="lines+markers",
                    name=s.capitalize(),
                    showlegend=show_legend,
                ),
                row=1, col=col
            )

        fig.add_trace(
            go.Scatter(
                x=df["test_number"],
                y=df[f"overall_{key}"],
                mode="lines+markers",
                name="Overall",
                showlegend=show_legend,
                line=dict(dash="dash")
            ),
            row=1, col=col
        )

    fig.update_layout(height=400)
    return fig


# ===============================
# PIE CHARTS
# ===============================
def create_pie_charts(df, subjects):

    latest = df.iloc[-1]

    fig = make_subplots(
        rows=1,
        cols=len(subjects),
        specs=[[{"type": "domain"}]*len(subjects)],
        subplot_titles=[f"{s.capitalize()} — Latest Test" for s in subjects]
    )

    for i, s in enumerate(subjects):
        fig.add_trace(
            go.Pie(
                labels=["Correct", "Wrong", "Unattempted"],
                values=[
                    latest[f"{s}_correct"],
                    latest[f"{s}_wrong"],
                    latest[f"{s}_unattempt"],
                ],
                hole=0.35,
                showlegend=(i == 0)
            ),
            row=1, col=i+1
        )

    fig.update_layout(height=350)
    return fig


# ===============================
# RUN APP
# ===============================
df, subjects = load_data()

if df.empty:
    st.warning("No data available.")
    st.stop()

df = compute_metrics(df, subjects)

st.caption(f"Last refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.subheader("Trends Over Tests")
st.plotly_chart(create_line_charts(df, subjects), use_container_width=True)

st.subheader("Latest Test Breakdown")
st.plotly_chart(create_pie_charts(df, subjects), use_container_width=True)
