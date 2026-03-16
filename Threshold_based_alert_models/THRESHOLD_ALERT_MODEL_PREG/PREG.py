# AlertExplorePREG_251216.py
# Streamlit app to explore PREG (power regulator) fan speed alerts.
# Preserves original three tabs and behavior, with light polish and comments.

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="PREG Alerts Explorer")

# ---------------------------
# Data config (relative path)
# ---------------------------

DATA_FILE = "C:/Users/320303731/OneDrive - Philips/Documents/PREG_MD.txt"
##DATA_FILE = "./data/PREG_MD.txt"  # was absolute: C:/.../PREG_MD.txt
DATA_SEP = ","


# ---------------------------
# Style helpers (kept simple)
# ---------------------------
def styled_header(text: str):
    st.markdown(f"<h3 style='color:#00126E;margin-bottom:0.25rem'>{text}</h3>", unsafe_allow_html=True)

def styled_subheader(text: str):
    st.markdown(f"<h4 style='color:#0B5ED7;margin-top:0.75rem'>{text}</h4>", unsafe_allow_html=True)


# ---------------------------
# Helpers
# ---------------------------
def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse SamplingTimestamp using a strict format first; fallback to flexible."""
    try:
        return pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S.%f", errors="raise")
    except Exception:
        return pd.to_datetime(series, errors="coerce")


@st.cache_data(show_spinner=False)
def load_data(path: str, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    # Normalize datetime fields
    df["SamplingDate"] = pd.to_datetime(df["SamplingDate"], errors="coerce")
    # Unique ID combines SampleID + device identifiers
    df["uniqueID"] = df["SampleID"].astype(str) + "_" + df["CatalogNumber"].astype(str) + "_" + df["SerialNumber"].astype(str)
    df = df.drop_duplicates()
    return df


def check_alert(group: pd.DataFrame, *, t: float, w: int, l: int) -> bool:
    """
    Generic alert function:
      - value <= t is a hit
      - rolling window of w hours
      - alert if any window has >= l hits
    """
    local = group.drop_duplicates().copy()
    local["SamplingTimestamp"] = _parse_timestamp(local["SamplingTimestamp"])
    local = local.sort_values("SamplingTimestamp").set_index("SamplingTimestamp")
    mask = (local["SampleValue"] <= t).astype(int)
    rolling_counts = mask.rolling(f"{w}h").sum()
    return (rolling_counts >= l).any()


# ---------------------------
# Title
# ---------------------------
styled_header("Sensor data analysis model: Power regulators use case")

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
tab1, tab2, tab3 = st.tabs(["Alerts by factory defaults", "Alerts by recommended values", "SampleValue distribution"])


# =============================================================================
# TAB 1: Factory default rule (t=1026 rpm, w=2h, l=2)
# =============================================================================
with tab1:
    t = 1026
    w = 2
    l = 2

    color_map = ["#02ABB1", "#0B5ED7", "#00126E", "#008800", "#00666F"]

    # Compute alerts per system (no partition filter for the bar count)
    systems = df.groupby("uniqueID").apply(lambda g: check_alert(g, t=t, w=w, l=l), include_groups=False)
    alerts = systems[systems == True]

    count_df = (
        df.loc[df["uniqueID"].isin(alerts.index)]
        .drop_duplicates()
        .groupby("SampleID")["uniqueID"]
        .nunique()
        .reset_index(name="Count")
    )

    fig_counts = px.bar(
        count_df,
        x="SampleID",
        y="Count",
        color="SampleID",
        color_discrete_sequence=color_map,
        text_auto=True,
        title="Number of systems reporting fan speed at threshold or less",
    )
    fig_counts.update_layout(showlegend=False)
    st.plotly_chart(fig_counts, key="countSystems_gen")

    # Detail for selected PREG/SampleID
    selected_partition_tab1 = st.selectbox(
        "Select PREG to see detail:", df["SampleID"].sort_values().unique(), key="key_tab1"
    )
    DF_df = df[df["SampleID"] == selected_partition_tab1].copy()

    alerts_calc = DF_df.groupby("uniqueID").apply(lambda g: check_alert(g, t=t, w=w, l=l), include_groups=False)
    alerts_calc = alerts_calc[alerts_calc == True]

    alerts_DF = DF_df.loc[(DF_df["uniqueID"].isin(alerts_calc.index)) & (DF_df["SampleValue"] <= t)].drop_duplicates()

    list_devices = (
        alerts_DF[["SampleID", "CatalogNumber", "SerialNumber", "uniqueID"]]
        .drop_duplicates()
        .sort_values(by="uniqueID")
        .copy()
    )
    list_devices["#"] = range(1, len(list_devices) + 1)

    styled_subheader("Systems reporting alert using factory default values:")
    st.dataframe(
        list_devices[["#", "SampleID", "CatalogNumber", "SerialNumber", "uniqueID"]],
        hide_index=True,
        use_container_width=True,
    )

    # Per-system timeline
    styled_subheader("Timeline per system")
    for uid in list_devices["uniqueID"]:
        subset = df[df["uniqueID"] == uid].copy()
        subset["SamplingTimestamp"] = _parse_timestamp(subset["SamplingTimestamp"])

        fig = px.scatter(
            subset,
            x="SamplingTimestamp",
            y="SampleValue",
            title=f"Fan speed timeline for Power Regulator in system {uid}",
        )
        fig.update_traces(marker=dict(size=3, symbol="diamond", color="#00126E"))
        fig.update_layout(xaxis_autorange=True, yaxis_autorange=True)
        fig.update_xaxes(ticks="inside", tickangle=270)

        # Threshold line
        fig.add_hline(
            y=t,
            line_color="#F85569",
            annotation_text=f"Alert threshold: {t}",
            annotation_position="top left",
            layer="below",
            line_width=4,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download rows for that system (<= t)
        alert_rows = (
            subset.loc[
                subset["SampleValue"] <= t,
                ["uniqueID", "SamplingTimestamp", "SampleValue", "SampleUnits", "CatalogNumber", "SerialNumber"],
            ]
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
# TAB 2: Recommended values (per PREG)
# =============================================================================
with tab2:
    # Per-PREG thresholds/config
    value_thresholds = {"PREG_FAN_1": 1026, "PREG_FAN_2": 1053, "PREG_FAN_3": 1053}
    window_hours = {"PREG_FAN_1": 2, "PREG_FAN_2": 2, "PREG_FAN_3": 2}
    limit = {"PREG_FAN_1": 3, "PREG_FAN_2": 3, "PREG_FAN_3": 3}

    selected_partition_tab2 = st.selectbox(
        "Select PREG:", df["SampleID"].sort_values().unique(), key="key_tab2"
    )
    w = int(window_hours.get(selected_partition_tab2, 2))
    l = int(limit.get(selected_partition_tab2, 3))
    t = float(value_thresholds.get(selected_partition_tab2, 1026))

    REC_df = df[df["SampleID"] == selected_partition_tab2].copy()

    alerts_calcR = REC_df.groupby("uniqueID").apply(lambda g: check_alert(g, t=t, w=w, l=l), include_groups=False)
    alerts_calcR = alerts_calcR[alerts_calcR == True]

    alerts_REC = REC_df.loc[
        (REC_df["uniqueID"].isin(alerts_calcR.index)) & (REC_df["SampleValue"] <= t)
    ].drop_duplicates()

    list_devicesREC = (
        alerts_REC[["SampleID", "CatalogNumber", "SerialNumber", "uniqueID"]]
        .drop_duplicates()
        .sort_values(by="uniqueID")
        .copy()
    )
    list_devicesREC["#"] = range(1, len(list_devicesREC) + 1)

    st.text("Values per PREG selected:")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Threshold (RPM)", t, border=True)
    m2.metric("Time window (hours)", w, border=True)
    m3.metric("Limit (hits)", l, border=True)
    m4.metric("Number of systems", len(list_devicesREC), border=True)

    styled_subheader("Systems reporting alert using recommended values:")
    st.dataframe(
        list_devicesREC[["#", "SampleID", "CatalogNumber", "SerialNumber", "uniqueID"]],
        hide_index=True,
        use_container_width=True,
    )

    styled_subheader("Timeline per system")
    for uid in list_devicesREC["uniqueID"]:
        subset_rec = df[df["uniqueID"] == uid].copy()
        subset_rec["SamplingTimestamp"] = _parse_timestamp(subset_rec["SamplingTimestamp"])

        fig_rec = px.scatter(
            subset_rec,
            x="SamplingTimestamp",
            y="SampleValue",
            title=f"Fan speed timeline for Power Regulator in system {uid}",
        )
        fig_rec.update_layout(
            title_font=dict(size=24),
            font=dict(size=16),
            legend=dict(font=dict(size=14)),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18),
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

        # Download rows for that system (<= t)
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

# =============================================================================
# TAB 3: Distribution
# =============================================================================
with tab3:
    value_thresholds = {"PREG_FAN_1": 1026, "PREG_FAN_2": 1053, "PREG_FAN_3": 1053}

    selected_partition_tab3 = st.selectbox(
        "Select PREG:", df["SampleID"].sort_values().unique(), key="key_tab3"
    )
    t = float(value_thresholds.get(selected_partition_tab3, 1026))

    df_distro = df[df["SampleID"] == selected_partition_tab3].copy()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Threshold (RPM)", t, border=True)
    m2.metric("Min value (RPM)", df_distro["SampleValue"].min(skipna=True, numeric_only=False), border=True)
    m3.metric("Max value (RPM)", df_distro["SampleValue"].max(skipna=True, numeric_only=False), border=True)
    m4.metric("Median value (RPM)", df_distro["SampleValue"].median(skipna=True, numeric_only=False), border=True)

    fig_SV = px.histogram(
        df_distro,
        x="SampleValue",
        title=f"Distribution of SampleValue for PREG: {selected_partition_tab3}",
    )
    fig_SV.add_vline(
        x=t,
        line_color="red",
        line_width=2,
        line_dash="dash",
        annotation_text=str(t),
        annotation_position="top right",
    )
    fig_SV.update_layout(margin=dict(t=80))
    st.plotly_chart(fig_SV, use_container_width=True)
    st.warning("Assure model is to set an alert when the FanSpeed values are too low instead of too high")
