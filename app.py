import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
    time.sleep(300)
    st.rerun()

# ===============================
# DATA LOADING
# ===============================
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL, header=[0, 1])

    df.columns = [
        "test_number" if c[0].strip().lower() == "date"
        else f"{c[0].strip().lower()}_{c[1].strip().lower()}"
        for c in df.columns
    ]

    df = df.apply(pd.to_numeric)

    subjects = sorted({
        col.split("_")[0]
        for col in df.columns if "_" in col
    } - {"test", "total"})

    return df, subjects


# ===============================
# CALCULATIONS
# ===============================
def compute_metrics(df, subjects):

    for s in subjects:
        df[f"{s}_total"] = df[f"{s}_attempted"] + df[f"{s}_unattempt"]
        df[f"{s}_correct"] = df[f"{s}_attempted"] - df[f"{s}_wrong"]

        df[f"{s}_accuracy"] = df[f"{s}_correct"] / df[f"{s}_attempted"].replace(0, 1)
        df[f"{s}_attempt_ratio"] = df[f"{s}_attempted"] / df[f"{s}_total"].replace(0, 1)

        net = df[f"{s}_correct"] * POSITIVE_MARK - df[f"{s}_wrong"] * NEGATIVE_MARK
        df[f"{s}_net_score"] = net
        df[f"{s}_normalized_score"] = net / df[f"{s}_total"].replace(0, 1)

    df["total_attempted"] = df[[f"{s}_attempted" for s in subjects]].sum(axis=1)
    df["total_unattempt"] = df[[f"{s}_unattempt" for s in subjects]].sum(axis=1)
    df["total_wrong"] = df[[f"{s}_wrong" for s in subjects]].sum(axis=1)

    df["total_correct"] = df["total_attempted"] - df["total_wrong"]
    df["total_questions"] = df["total_attempted"] + df["total_unattempt"]

    df["overall_accuracy"] = df["total_correct"] / df["total_attempted"].replace(0, 1)
    df["overall_attempt_ratio"] = df["total_attempted"] / df["total_questions"].replace(0, 1)

    overall_net = df["total_correct"] * POSITIVE_MARK - df["total_wrong"] * NEGATIVE_MARK
    df["overall_net_score"] = overall_net
    df["overall_normalized_score"] = overall_net / df["total_questions"].replace(0, 1)

    return df


# ===============================
# FIGURE 1: Line Charts
# ===============================
def create_line_charts(df, subjects):
    COLORS = ["#4C9BE8", "#E8704C", "#4CE8A0", "#C44CE8", "#E8C44C"]
    OVERALL_COLOR = "#f0f0f0"

    metrics = [
        ("accuracy",         "Accuracy"),
        ("attempt_ratio",    "Attempt Ratio"),
        ("normalized_score", "Normalized Score"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Accuracy", "Attempt Ratio", "Normalized Score"],
        horizontal_spacing=0.08,
    )

    for col_idx, (key, ylabel) in enumerate(metrics):
        col = col_idx + 1
        first_chart = col_idx == 0

        for j, s in enumerate(subjects):
            fig.add_trace(
                go.Scatter(
                    x=df["test_number"],
                    y=df[f"{s}_{key}"],
                    mode="lines+markers",
                    name=s.capitalize(),
                    legendgroup=s,
                    showlegend=first_chart,
                    line=dict(color=COLORS[j % len(COLORS)], width=2),
                    marker=dict(size=6),
                ),
                row=1, col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=df["test_number"],
                y=df[f"overall_{key}"],
                mode="lines+markers",
                name="Overall",
                legendgroup="overall",
                showlegend=first_chart,
                line=dict(color=OVERALL_COLOR, width=3, dash="dash"),
                marker=dict(size=7, symbol="diamond"),
            ),
            row=1, col=col,
        )

        fig.update_yaxes(title_text=ylabel, row=1, col=col)
        fig.update_xaxes(title_text="Test #", row=1, col=col)

    fig.update_layout(
        height=380,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="#444",
            borderwidth=1,
        ),
        margin=dict(t=60, b=40, l=40, r=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#2a2a4a", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a4a", zeroline=False)

    return fig


# ===============================
# FIGURE 2: Pie Charts
# ===============================
def create_pie_charts(df, subjects):
    PIE_COLORS = ["#4CE8A0", "#E8704C", "#888888"]
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
                    latest[f"{s}_correct"],
                    latest[f"{s}_wrong"],
                    latest[f"{s}_unattempt"],
                ],
                name=s.capitalize(),
                marker=dict(colors=PIE_COLORS),
                hole=0.35,
                textinfo="label+percent",
                showlegend=(i == 0),
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        height=350,
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="#444",
            borderwidth=1,
        ),
        margin=dict(t=60, b=20, l=20, r=20),
    )

    return fig


# ===============================
# RUN APP
# ===============================
df, subjects = load_data()
df = compute_metrics(df, subjects)

st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.subheader("Trends Over Tests")
fig_lines = create_line_charts(df, subjects)
st.plotly_chart(fig_lines, use_container_width=True)

st.subheader("Latest Test Breakdown")
fig_pies = create_pie_charts(df, subjects)
st.plotly_chart(fig_pies, use_container_width=True)
