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

    # Flatten multi-index columns safely
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            if c[0].strip().lower() == "date":
                new_cols.append("test_number")
            else:
                new_cols.append(f"{c[0].strip().lower()}_{c[1].strip().lower()}")
        else:
            new_cols.append(str(c).strip().lower())

    df.columns = new_cols

    # Convert numeric columns only (exclude test_number)
    for col in df.columns:
        if col != "test_number":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing test_number
    df = df.dropna(subset=["test_number"])

    # Sort properly
    df = df.sort_values("test_number").reset_index(drop=True)

    subjects = sorted({
        col.split("_")[0]
        for col in df.columns
        if "_" in col and col.endswith("_attempted")
    })

    return df, subjects


# ===============================
# CALCULATIONS
# ===============================
def safe_divide(numerator, denominator):
    return np.where(denominator == 0, 0, numerator / denominator)


def compute_metrics(df, subjects):

    if df.empty or not subjects:
        return df

    df = df.copy()

    for s in subjects:
        required_cols = [f"{s}_attempted", f"{s}_wrong", f"{s}_unattempt"]
        if not all(col in df.columns for col in required_cols):
            continue

        df[f"{s}_total"] = df[f"{s}_attempted"] + df[f"{s}_unattempt"]
        df[f"{s}_correct"] = df[f"{s}_attempted"] - df[f"{s}_wrong"]

        df[f"{s}_accuracy"] = safe_divide(
            df[f"{s}_correct"], df[f"{s}_attempted"]
        )

        df[f"{s}_attempt_ratio"] = safe_divide(
            df[f"{s}_attempted"], df[f"{s}_total"]
        )

        net = df[f"{s}_correct"] * POSITIVE_MARK - df[f"{s}_wrong"] * NEGATIVE_MARK
        df[f"{s}_net_score"] = net
        df[f"{s}_normalized_score"] = safe_divide(net, df[f"{s}_total"])

    # Overall calculations
    attempted_cols = [f"{s}_attempted" for s in subjects if f"{s}_attempted" in df]
    wrong_cols = [f"{s}_wrong" for s in subjects if f"{s}_wrong" in df]
    unattempt_cols = [f"{s}_unattempt" for s in subjects if f"{s}_unattempt" in df]

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
# FIGURE 1: Line Charts
# ===============================
def create_line_charts(df, subjects):

    metrics = [
        ("accuracy", "Accuracy"),
        ("attempt_ratio", "Attempt Ratio"),
        ("normalized_score", "Normalized Score"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[m[1] for m in metrics],
        horizontal_spacing=0.08,
    )

    for col_idx, (key, ylabel) in enumerate(metrics):
        col = col_idx + 1
        first_chart = col_idx == 0

        for s in subjects:
            fig.add_trace(
                go.Scatter(
                    x=df["test_number"],
                    y=df[f"{s}_{key}"],
                    mode="lines+markers",
                    name=s.capitalize(),
                    legendgroup=s,
                    showlegend=first_chart,
                ),
                row=1,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=df["test_number"],
                y=df[f"overall_{key}"],
                mode="lines+markers",
                name="Overall",
                legendgroup="overall",
                showlegend=first_chart,
                line=dict(dash="dash"),
            ),
            row=1,
            col=col,
        )

        fig.update_yaxes(title_text=ylabel, row=1, col=col)
        fig.update_xaxes(title_text="Test #", row=1, col=col)

    fig.update_layout(height=400, margin=dict(t=60, b=40, l=40, r=40))
    return fig


# ===============================
# FIGURE 2: Pie Charts
# ===============================
def create_pie_charts(df, subjects):

    if df.empty:
        return go.Figure()

    latest = df.iloc[-1]

    fig = make_subplots(
        rows=1,
        cols=len(subjects),
        specs=[[{"type": "domain"}] * len(subjects)],
        subplot_titles=[f"{s.capitalize()} — Latest Test" for s in subjects],
    )

    for i, s in enumerate(subjects):
        fig.add_trace(
            go.Pie(
                labels=["Correct", "Wrong", "Unattempted"],
                values=[
                    latest.get(f"{s}_correct", 0),
                    latest.get(f"{s}_wrong", 0),
                    latest.get(f"{s}_unattempt", 0),
                ],
                hole=0.35,
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(height=350, margin=dict(t=60, b=20, l=20, r=20))
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
