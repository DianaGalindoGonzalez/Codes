# --- Imports ---
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re
import glob
from pathlib import Path

# -----------------------------
# Page config & styles
# -----------------------------
st.set_page_config(layout="wide", page_title="ACB Temperature Anomaly Detection")

def styled_header(text: str, color: str = "#00126E", size: str = "24px"):
    st.markdown(
        f"<h4 style='color:{color}; text-align: center; font-size:{size}; margin: 0 0 0.5rem 0;'>{text}</h4>",
        unsafe_allow_html=True
    )

def styled_subheader(text: str, color: str = "#0B5ED7", size: str = "18px"):
    st.markdown(
        f"<h3 style='color:{color}; text-align: center; font-size:{size}; margin: 0.5rem 0;'>{text}</h3>",
        unsafe_allow_html=True
    )

def styled_text(text: str, color: str = "#808080", size: str = "16px"):
    st.markdown(
        f"<p style='color:{color}; font-size:{size}; margin: 0.25rem 0;'>{text}</p>",
        unsafe_allow_html=True
    )

def header(text):
    st.markdown(
        f"<h2 style='text-align:center; color:#00126E; margin-top: 0.5rem;'>{text}</h2>",
        unsafe_allow_html=True
    )

st.markdown("""
    <style>
        .zero-gap { margin-top: -25px !important; padding-top: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# ---- Inputs (hardcoded) ----
# -----------------------------
FOLDER_PATTERN = r"C:/Users/320303731/Downloads/*_Temperature.csv"
EHA_PATH = r"C:/Users/320303731/OneDrive - Philips/Documents/Models/260317_EHAEventsSystemsWAlerts2.txt"

MIN_TEMP = 0.0
MAX_TEMP = 93.0
ROLLING_WINDOW = 8
ZSCORE_THRESHOLD = 5
ZSCORE_MOD_THRESHOLD = 5
JUMP_THRESHOLD = 10
MAX_EXPECTED_GAP_MIN = 2881   # minutes

# -----------------------------
# Helpers
# -----------------------------
def extract_sysid_from_name(name: str):
    """
    Extracts SysID from filenames like "['795200--US918B1315']_Temperature.csv".
    Returns None if pattern not found.
    """
    m = re.search(r"\['(.+?)'\]_Temperature\.csv$", name)
    if m:
        return m.group(1)
    m2 = re.search(r"\[([^\]]+)\]", name)
    return m2.group(1) if m2 else None

@st.cache_data(show_spinner=False)
def load_eha(eha_file) -> pd.DataFrame:
    if not Path(eha_file).exists():
        return pd.DataFrame()
    df = pd.read_csv(eha_file, sep = ',')
    if "EventTimestamp" in df.columns:
        df["EventTimestamp"] = pd.to_datetime(df["EventTimestamp"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_csvs_from_pattern(pattern: str):
    """
    Returns list of (sysid, df) for all files matching pattern.
    """
    paths = glob.glob(pattern)
    out = []
    for p in paths:
        name = Path(p).name
        sysid = extract_sysid_from_name(name)
        if sysid is None:
            continue
        df = pd.read_csv(p, sep=',')
        df["SysID"] = sysid
        out.append((sysid, df))
    return out

def compute_anomaly_dataframe(
    df: pd.DataFrame,
    min_temp: float,
    max_temp: float,
    rolling_window: int,
    zscore_threshold: float,
    modz_threshold: float,
    jump_threshold: float,
    max_expected_gap_min: float
) -> pd.DataFrame:
    """
    df must contain: SamplingTimestamp, SampleValue (numeric), optionally SamplingID, for a single SysID.
    Returns a new DataFrame with anomaly columns.
    """
    df = df.copy()

    # Filter to ACB if column exists
    if "SamplingID" in df.columns:
        df = df[df["SamplingID"] == "ACB"].copy()

    # Timestamps
    df["SamplingTimestamp"] = pd.to_datetime(df["SamplingTimestamp"], errors="coerce")
    df = df.dropna(subset=["SamplingTimestamp"]).sort_values("SamplingTimestamp").reset_index(drop=True)

    if df.empty:
        # Add expected columns to avoid downstream errors
        for col in ["time_diff_min", "new_streak", "streak_id", "threshold_alert",
                    "rolling_mean", "rolling_median", "rolling_std", "z_score",
                    "zscore_alert", "modified_z_score", "modified_zscore_alert",
                    "delta_temp", "jump_alert", "anomaly", "reason"]:
            df[col] = pd.Series(dtype="float64") if col not in ["new_streak", "reason"] else (False if col=="new_streak" else "")
        return df

    # Compute intra-streak time gap
    df["time_diff_min"] = df["SamplingTimestamp"].diff().dt.total_seconds() / 60
    df["new_streak"] = (df["time_diff_min"].isna()) | (df["time_diff_min"] > max_expected_gap_min)
    df["streak_id"] = df["new_streak"].cumsum()

    # Threshold alerts
    df["threshold_alert"] = (df["SampleValue"] < min_temp) | (df["SampleValue"] > max_temp)

    # Rolling statistics (shifted so current point compares to prior history)
    df["rolling_mean"] = (
        df.groupby("streak_id")["SampleValue"]
          .transform(lambda s: s.shift(1).rolling(window=rolling_window, min_periods=rolling_window).mean())
    )
    df["rolling_median"] = (
        df.groupby("streak_id")["SampleValue"]
          .transform(lambda s: s.shift(1).rolling(window=rolling_window, min_periods=rolling_window).median())
    )
    df["rolling_std"] = (
        df.groupby("streak_id")["SampleValue"]
          .transform(lambda s: s.shift(1).rolling(window=rolling_window, min_periods=rolling_window).std())
    )
    df["rolling_std"] = df["rolling_std"].replace(0, np.nan)

    # Z-score alerts
    df["z_score"] = (df["SampleValue"] - df["rolling_mean"]) / df["rolling_std"]
    df["zscore_alert"] = df["z_score"].abs() > zscore_threshold

    # Modified z-score (robust)
    const_modz = 0.6745
    global_median = df["SampleValue"].median()
    mad = (np.abs(df["SampleValue"] - global_median)).median()
    mad_safe = mad if mad and not np.isclose(mad, 0.0) else np.nan
    df["modified_z_score"] = const_modz * (df["SampleValue"] - df["rolling_median"]) / mad_safe
    df["modified_zscore_alert"] = df["modified_z_score"].abs() > modz_threshold

    # Sudden jump detection (per streak)
    df["delta_temp"] = df.groupby("streak_id")["SampleValue"].diff()
    df["jump_alert"] = df["delta_temp"].abs() > jump_threshold

    # Final anomaly flag
    df["anomaly"] = df[["threshold_alert", "zscore_alert", "modified_zscore_alert", "jump_alert"]].any(axis=1)

    # Reason label
    def build_reason(row):
        reasons = []
        if row.get("threshold_alert", False): reasons.append("threshold")
        if row.get("zscore_alert", False): reasons.append("z-score")
        if row.get("modified_zscore_alert", False): reasons.append("modified z-score")
        if row.get("jump_alert", False): reasons.append("jump")
        return ", ".join(reasons)

    df["reason"] = df.apply(build_reason, axis=1)
    return df

def build_plot(df: pd.DataFrame, eha_df: pd.DataFrame, min_temp: float, max_temp: float, title: str) -> go.Figure:
    palette = px.colors.qualitative.Dark24

    # Map each reason to a color
    unique_reasons_ordered = [r for r in df["reason"].unique() if r]  # skip empty
    reason_color_map = {reason: palette[i % len(palette)] for i, reason in enumerate(unique_reasons_ordered)}

    fig = go.Figure()

    # EHA vertical lines
    if not eha_df.empty and "EventTimestamp" in eha_df.columns:
        ts_series = pd.to_datetime(eha_df["EventTimestamp"], errors="coerce").dropna().dt.floor("s")
        for t in ts_series:
            fig.add_vline(
                x=t,
                line_dash="solid",
                line_color="red",
                line_width=3,
                opacity=1
            )

    # Main series
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df["SamplingTimestamp"],
            y=df["SampleValue"],
            mode="markers",
            name="ACB Temperature",
            marker=dict(size=6, color="#2a6fdb")
        ))

    # Threshold lines
    fig.add_hline(y=max_temp, line_dash="dash",
                  annotation_text=f"Max threshold ({max_temp}°C)",
                  annotation_position="top left")
    fig.add_hline(y=min_temp, line_dash="dash",
                  annotation_text=f"Min threshold ({min_temp}°C)",
                  annotation_position="bottom left")
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="red", width=3),
        name="EHA Event [001]"
    ))

    # Anomaly markers grouped by reason
    anoms = df[df["anomaly"]] if not df.empty else pd.DataFrame()
    if not anoms.empty:
        for reason, subset in anoms.groupby("reason"):
            color = reason_color_map.get(reason, "black")
            fig.add_trace(go.Scatter(
                x=subset["SamplingTimestamp"],
                y=subset["SampleValue"],
                mode="markers",
                marker=dict(size=11, symbol="x", color=color, line=dict(width=1, color=color)),
                name=f"Anomaly: {reason if reason else 'unknown'}"
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Temperature (°C)",
        height=750,
        margin=dict(t=00, r=20, b=40, l=60),
        legend=dict(orientation="h", yanchor="bottom",xanchor="center",x=0.5,y=-0.5,font=dict(size=13),itemwidth=30)
    )
    fig.update_xaxes(tickangle=-90)

    return fig

# -----------------------------
# App layout & logic (single device)
# -----------------------------
header("ACB Temperature sensor anomaly detection (Single device)")

# Load
csv_items = load_csvs_from_pattern(FOLDER_PATTERN)
eha_df = load_eha(EHA_PATH)

if not csv_items:
    styled_text("No CSV files found. Please check your folder pattern.", color="#a33")
    st.stop()

# Device picker (single selection)
devices = sorted([sid for sid, _ in csv_items])
selected_device = st.selectbox("Choose a device (SysID):", devices, index=0)

# Prepare data for the selected device
dev_df = None
for sid, df in csv_items:
    if sid == selected_device:
        dev_df = df.copy()
        break

if dev_df is None or dev_df.empty:
    styled_text("No data for the selected device.", color="#a33")
    st.stop()

# Compute anomalies
df = compute_anomaly_dataframe(
    dev_df,
    min_temp=MIN_TEMP,
    max_temp=MAX_TEMP,
    rolling_window=ROLLING_WINDOW,
    zscore_threshold=ZSCORE_THRESHOLD,
    modz_threshold=ZSCORE_MOD_THRESHOLD,
    jump_threshold=JUMP_THRESHOLD,
    max_expected_gap_min=MAX_EXPECTED_GAP_MIN
)

# EHA for this device
eha_for_dev = pd.DataFrame()
if not eha_df.empty and "ms_sysid" in eha_df.columns:
    eha_for_dev = eha_df[eha_df["ms_sysid"] == selected_device].copy()

styled_header(f"ACB Temperature sensor anomaly detection for system {selected_device}")
styled_subheader("1. Univariate (based on its own historic data)")

# Metrics
col1, col2, col3 = st.columns(3)
num_streaks = int(df["streak_id"].max()) if not df.empty else 0
median_freq = df["time_diff_min"].median() if "time_diff_min" in df.columns else np.nan
num_eha = 0 if eha_for_dev.empty else eha_for_dev.shape[0]
col1.metric("Number of streaks", f"{num_streaks}", border=True)
col2.metric("Timestamp frequency median", f"{median_freq:.0f} min" if pd.notna(median_freq) else "—",border = True)
col3.metric("Number of EHA [001] Events", f"{num_eha}", border=True)

# Plot
fig = build_plot(
    df=df,
    eha_df=eha_for_dev if not eha_for_dev.empty else pd.DataFrame(),
    min_temp=MIN_TEMP,
    max_temp=MAX_TEMP,
    title=f"{selected_device} — ACB Temperature"
)
st.markdown('<div class="zero-gap">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_device}")
st.markdown('</div>', unsafe_allow_html=True)

# Anomaly table
with st.expander("Show anomaly and EHA events rows"):
    cols = ["SamplingTimestamp", "SampleValue", "reason", "streak_id", "delta_temp", "z_score", "modified_z_score"]
    existing_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[df["anomaly"]][existing_cols], use_container_width=True)
    st.dataframe(eha_for_dev)
