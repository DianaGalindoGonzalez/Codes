# AlertExplore_HardDisk2025.py
# Streamlit app to explore disk space alerts per system.

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="centered", page_title="Hard Disk Alerts Explorer")

DATA_FILE = "./data/Result_45.txt"  
# To obtain Result_45.txt:
# SQL Query = 

DATA_SEP = ","                      # original separator

# ---------------------------
# Helpers
# ---------------------------
def _parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse SamplingTimestamp with a strict format first; if it fails, retry with
    pandas flexible parser to avoid breaking the app.
    """
    try:
        return pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S.%f", errors="raise")
    except Exception:
        return pd.to_datetime(series, errors="coerce")


@st.cache_data(show_spinner=False)
def load_data(path: str, sep: str = ",") -> pd.DataFrame:
    """Load the source file and apply basic normalization."""
    df = pd.read_csv(path, sep=sep)
    # Build unique system identifier (CatalogNumber_SerialNumber)
    df["uniqueID"] = df["CatalogNumber"].astype(str) + "_" + df["SerialNumber"].astype(str)
    # Drop exact duplicates early
    df = df.drop_duplicates()
    return df


def check_alert(temp_df: pd.DataFrame) -> bool:
    """
    Factory-default alert rule (Tab 1):
      - Treat values <= 5 as alert hits
      - Rolling 48h window
      - If any 48h window has at least 2 hits → system flagged
    """
    local = temp_df.drop_duplicates().copy()
    local["SamplingTimestamp"] = _parse_timestamp(local["SamplingTimestamp"])
    local = local.sort_values("SamplingTimestamp").set_index("SamplingTimestamp")

    mask = (local["SampleValue"] <= 5).astype(int)
    rolling_counts = mask.rolling("48h").sum()
    return (rolling_counts >= 2).any()


def check_alert_rec(group: pd.DataFrame, *, t: float, w: int, l: int) -> bool:
    """
    Recommended-values alert rule (Tab 2):
      - threshold t (value <= t is a hit)
      - rolling window w hours
      - need l hits in any window to flag
    """
    local = group.drop_duplicates().copy()
    local["SamplingTimestamp"] = _parse_timestamp(local["SamplingTimestamp"])
    local = local.sort_values("SamplingTimestamp").set_index("SamplingTimestamp")

    mask = (local["SampleValue"] <= t).astype(int)
    rolling_counts = mask.rolling(f"{w}h").sum()
    return (rolling_counts >= l).any()


# ---------------------------
# Load data
# ---------------------------
try:
    df = load_data(DATA_FILE, DATA_SEP)
except FileNotFoundError:
    st.error(f"Data file not found: {DATA_FILE}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["Alerts factory defaults", "Alerts recommended values"])

# =============================================================================
# TAB 1: Factory defaults (<= 5, 48h window, need >= 2 hits)
# =============================================================================
with tab1:
    st.text("Number of systems per partition reporting 5% or less of available free space")

    # Find systems that meet the factory-default alert condition
    alerts = df.groupby("uniqueID").apply(check_alert, include_groups=False)
    alerts = alerts[alerts]  # keep only True

    filtered_dfFD = df.loc[
        (df["uniqueID"].isin(alerts.index)) & (df["SampleValue"] <= 5)
    ].drop_duplicates()

    # Count systems per partition (SamplingID)
    color_map = ["#02ABB1", "#0B5ED7", "#00126E", "#008800", "#00666F"]
    count_df = (
        filtered_dfFD.groupby("SamplingID")["uniqueID"]
        .nunique()
        .reset_index(name="Count")
        .sort_values("SamplingID")
    )
    fig_counts = px.bar(
        count_df,
        x="SamplingID",
        y="Count",
        color="SamplingID",
        color_discrete_sequence=color_map,
        text_auto=True,
        title="Systems with alerts per partition (factory defaults)",
    )
    fig_counts.update_layout(showlegend=False)
    st.plotly_chart(fig_counts, key="countSystems_gen")

    # Partition selector (only among partitions that have alerts)
    if filtered_dfFD.empty:
        st.info("No systems triggered factory-default alerts.")
        st.stop()

    selected_partition_tab2 = st.selectbox(
        "Select partition:", filtered_dfFD["SamplingID"].sort_values().unique(), key="key_tab2"
    )

    filtered_df = filtered_dfFD[filtered_dfFD["SamplingID"] == selected_partition_tab2].copy()
    filtered_df["SamplingTimestamp"] = _parse_timestamp(filtered_df["SamplingTimestamp"])

    # List systems table
    list_devices = (
        filtered_df[["SamplingID", "uniqueID"]]
        .drop_duplicates()
        .sort_values(by="uniqueID")
        .copy()
    )
    list_devices["#"] = range(1, len(list_devices) + 1)
    st.dataframe(list_devices[["#", "SamplingID", "uniqueID"]], hide_index=True, use_container_width=True)

    # Per-system charts
    st.text("Timeline per system")
    for uid in list_devices["uniqueID"]:
        subset = df[df["uniqueID"] == uid].copy()
        subset["SamplingTimestamp"] = _parse_timestamp(subset["SamplingTimestamp"])

        fig = px.scatter(
            subset,
            x="SamplingTimestamp",
            y="SampleValue",
            title=f"Percentage of available free space timeline for {uid}",
        )
        fig.update_traces(marker=dict(size=3, symbol="diamond", color="#00126E"))
        fig.update_layout(xaxis_autorange=True, yaxis_range=[0, 8])
        fig.update_xaxes(ticks="inside", tickangle=270)

        # Threshold line (factory default)
        fig.add_hline(
            y=5,
            line_color="#F85569",
            annotation_text="Alert threshold: 5",
            annotation_position="top left",
            layer="below",
            line_width=4,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download filtered rows for that system (<= 5)
        alert_rows = (
            subset.loc[subset["SampleValue"] <= 5, ["uniqueID", "SamplingTimestamp", "SampleValue", "SampleUnits", "CatalogNumber", "SerialNumber"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        csv = alert_rows.to_csv(index=False)
        key_dfd = f"download_{uid}"
        file_name = f"{uid}_data.csv"

        with st.expander("System alert data"):
            st.dataframe(alert_rows, use_container_width=True)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
                icon=":material/download:",
                key=key_dfd,
            )

# =============================================================================
# TAB 2: Recommended values (per partition thresholds & windows)
# =============================================================================
with tab2:
    # Per-partition configs
    value_thresholds = {"D:": 5, "E:": 15, "J:": 5, "K:": 20, "R:": 5, "S:": 5}
    window_hours = {"D:": 48, "E:": 6, "J:": 48, "K:": 6, "R:": 48, "S:": 48}
    limit = {"D:": 2, "E:": 3, "J:": 2, "K:": 3, "R:": 2, "S:": 2}

    selected_partition_tab3 = st.selectbox(
        "Select partition:", df["SamplingID"].sort_values().unique(), key="key_tab3"
    )
    filtered_df = df[df["SamplingID"] == selected_partition_tab3].copy()
    filtered_df["SamplingTimestamp"] = _parse_timestamp(filtered_df["SamplingTimestamp"])

    # Fetch parameters for selected partition
    w = int(window_hours.get(selected_partition_tab3, 48))
    l = int(limit.get(selected_partition_tab3, 2))
    t = float(value_thresholds.get(selected_partition_tab3, 5.0))

    # Compute alerts per system with recommended config
    alerts = df.groupby("uniqueID").apply(
        lambda g: check_alert_rec(g, t=t, w=w, l=l),
        include_groups=False,
    )
    alerts = alerts[alerts]

    filtered_dfR = filtered_df.loc[
        (filtered_df["uniqueID"].isin(alerts.index)) & (filtered_df["SampleValue"] <= t)
    ].drop_duplicates()

    list_devices_recommended = (
        filtered_dfR[["SamplingID", "uniqueID"]]
        .drop_duplicates()
        .sort_values(by="uniqueID")
        .copy()
    )
    list_devices_recommended["#"] = range(1, len(list_devices_recommended) + 1)

    st.text("Recommended values per partition selected:")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Threshold (%)", t, border=True)
    m2.metric("Time window (hours)", w, border=True)
    m3.metric("Limit", l, border=True)
    m4.metric("Number of systems", len(list_devices_recommended), border=True)

    st.dataframe(
        list_devices_recommended[["#", "SamplingID", "uniqueID"]],
        hide_index=True,
        use_container_width=True,
    )

    st.text("Timeline per system")
    for uid in list_devices_recommended["uniqueID"]:
        subset_rec = df[df["uniqueID"] == uid].copy()
        subset_rec["SamplingTimestamp"] = _parse_timestamp(subset_rec["SamplingTimestamp"])

        fig_rec = px.scatter(
            subset_rec,
            x="SamplingTimestamp",
            y="SampleValue",
            title=f"Percentage of available free space timeline for {uid}",
        )
        fig_rec.update_traces(marker=dict(size=3, symbol="diamond", color="#A80DF2"))
        fig_rec.update_xaxes(ticks="inside", tickangle=270)
        fig_rec.update_layout(xaxis_autorange=True, yaxis_autorange=True)

        fig_rec.add_hline(
            y=t,
            line_color="#F62FBF",
            annotation_text=f"Alert threshold: {t}",
            annotation_position="top left",
            layer="below",
            line_width=4,
        )
        st.plotly_chart(fig_rec, use_container_width=True)

        # Download filtered rows for that system (<= t)
        alert_rows_rec = (
            subset_rec.loc[
                subset_rec["SampleValue"] <= t,
                ["uniqueID", "SamplingTimestamp", "SampleValue", "SampleUnits", "CatalogNumber", "SerialNumber"],
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        csv_rec = alert_rows_rec.to_csv(index=False)
        key_download_rec = f"download_r{uid}"
        file_name = f"{uid}_data.csv"

        with st.expander("System alert data"):
            st.dataframe(alert_rows_rec, use_container_width=True)
            st.download_button(
                label="Download CSV",
                data=csv_rec,
                file_name=file_name,
                mime="text/csv",
                icon=":material/download:",
                key=key_download_rec,
            )
