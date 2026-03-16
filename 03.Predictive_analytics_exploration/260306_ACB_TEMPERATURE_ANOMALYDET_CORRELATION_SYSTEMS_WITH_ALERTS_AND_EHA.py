import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")

# =========================================================
# STYLING HELPERS
# =========================================================
def styled_header(text: str, color: str = "#00126E", size: str = "24px"):
    st.markdown(
        f"<h4 style='color:{color}; text-align: center; font-size:{size};'>{text}</h4>",
        unsafe_allow_html=True
    )

def styled_subheader(text: str, color: str = "#0B5ED7", size: str = "18px"):
    st.markdown(
        f"<h3 style='color:{color}; text-align: center; font-size:{size};'>{text}</h3>",
        unsafe_allow_html=True
    )

def styled_text(text: str, color: str = "#808080", size: str = "18px"):
    st.markdown(
        f"<p style='color:{color}; font-size:{size};'>{text}</p>",
        unsafe_allow_html=True
    )

def header(text):
    st.markdown(
        f"<h2 style='text-align:center; color:#00126E'>{text}</h2>",
        unsafe_allow_html=True
    )

# =========================================================
# MAPPINGS
# =========================================================

# Map exact (SamplingID, SampleID) -> parameter group
PARAM_GROUP_MAP = {
    # ---------------- Temperature ----------------
    ("ACB", "FPGA_FUSION"): "Temperature",
    ("ACB", "FPGA_XENON"): "Temperature",
    ("ACB", "PROTEGO_FPGA_FUSION"): "Temperature",
    ("ACB", "TEMP_ACB_FPGA_0"): "Temperature",
    ("CB", "CB_TEMP"): "Temperature",
    ("CB", "FPGA_KRYPTONA"): "Temperature",
    ("CB", "FPGA_NEON0"): "Temperature",
    ("CB", "FPGA_NEON1"): "Temperature",
    ("CB", "FPGA_PHILAE"): "Temperature",
    ("CB", "FPGA_KRYPTONB"): "Temperature",
    ("CB", "FPGA_TESLA"): "Temperature",
    ("CB", "TEMP_CB_FPGA_0"): "Temperature",
    ("MotherBoard", "CPU1_TEMP"): "Temperature",
    ("MotherBoard", "CPU2_TEMP"): "Temperature",
    ("MotherBoard", "CPU_TEMP"): "Temperature",
    ("MotherBoard", "SYS1_TEMP"): "Temperature",
    ("MotherBoard", "SYS2_TEMP"): "Temperature",
    ("MotherBoard", "SYS_TEMP"): "Temperature",
    ("MotherBoard", "System"): "Temperature",
    ("PRB", "TEMP_PREG_BOARD_0"): "Temperature",
    ("PREG", "PREG_TEMP"): "Temperature",

    # ---------------- Fan Speed ----------------
    ("IMB", "PREG_FAN_1"): "Fan Speed",
    ("IMB", "PREG_FAN_2"): "Fan Speed",
    ("IMB", "PREG_FAN_3"): "Fan Speed",
    ("MotherBoard", "CPU_FAN"): "Fan Speed",
    ("PREG", "PREG_FAN_1"): "Fan Speed",
    ("PREG", "PREG_FAN_2"): "Fan Speed",
    ("PREG", "PREG_FAN_3"): "Fan Speed",

    # ---------------- Voltage ----------------
    ("MotherBoard", "+1.5V"): "Voltage",
    ("MotherBoard", "+12V"): "Voltage",
    ("MotherBoard", "+5V"): "Voltage",
    ("MotherBoard", "+3.3V"): "Voltage",
    ("MotherBoard", "+3.3VSB"): "Voltage",
    ("MotherBoard", "+3VSB"): "Voltage",
    ("MotherBoard", "+5V"): "Voltage",
    ("MotherBoard", "+5VSB"): "Voltage",
    ("MotherBoard", "Battery"): "Voltage",
    ("MotherBoard", "CPU"): "Voltage",
    ("MotherBoard", "AVCC"): "Voltage",
    ("MotherBoard", "CPU1_DMM"): "Voltage",
    ("MotherBoard", "CPU1"): "Voltage",
    ("MotherBoard", "CPU2"): "Voltage",
    ("MotherBoard", "CPU2_DMM"): "Voltage",
    ("MotherBoard", "VBAT"): "Voltage",

    # ---------------- Hard Disk Space ----------------
    ("D:", "PercentAvailableFreeSpace"): "PercentAvailableFreeSpace",
    ("E:","PercentAvailableFreeSpace"): "PercentAvailableFreeSpace",
    ("J:","PercentAvailableFreeSpace"): "PercentAvailableFreeSpace",
    ("K:","PercentAvailableFreeSpace"): "PercentAvailableFreeSpace",
    ("R:","PercentAvailableFreeSpace"): "PercentAvailableFreeSpace",
}

# Map PARAMETER -> sensor bucket
# Anything not listed becomes MB by default
SENSOR_GROUP_MAP = {
    # ACB
    "FPGA_ARGON": "ACB",
    "FPGA_FUSION": "ACB",
    "FPGA_XENON": "ACB",
    "PROTEGO_FPGA_FUSION": "ACB",
    "TEMP_ACB_FPGA_0": "ACB",
    # CB
    "CB_TEMP": "CB",
    "FPGA_KRYPTONA": "CB",
    "FPGA_KRYPTONB": "CB",
    "FPGA_NEON0": "CB",
    "FPGA_NEON1": "CB",
    "FPGA_PHILAE": "CB",
    "FPGA_TESLA": "CB",
    "PROTEGO_FPGA_PHILAE": "CB",
    "TEMP_CB_FPGA_0": "CB",
    # IMB
    "PREG_FAN_1": "IMB",
    "PREG_FAN_2": "IMB",
    "PREG_FAN_3": "IMB",
    # HD
    "D:": "PercentAvailableFreeSpace",
    "E:": "PercentAvailableFreeSpace",
    "J:": "PercentAvailableFreeSpace",
    "K:": "PercentAvailableFreeSpace",
    "R:": "PercentAvailableFreeSpace",
    #MotherBoard
    "+1.5V": "MotherBoard",
    "+12V": "MotherBoard",
    "+3.3V": "MotherBoard",
    "+3.3VSB": "MotherBoard",
    "+3VSB": "MotherBoard",
    "+5V": "MotherBoard",
    "+5VSB": "MotherBoard",
    "AVCC": "MotherBoard",
    "Battery": "MotherBoard",
    "CPU": "MotherBoard",
    "CPU1": "MotherBoard",
    "CPU1_DMM": "MotherBoard",
    "CPU1_TEMP": "MotherBoard",
    "CPU2": "MotherBoard",
    "CPU2_DMM": "MotherBoard",
    "CPU2_TEMP": "MotherBoard",
    "CPU_FAN": "MotherBoard",
    "CPU_TEMP": "MotherBoard",
    "HUMIDITY": "MotherBoard",
    "SYS1_TEMP": "MotherBoard",
    "SYS2_TEMP": "MotherBoard",
    "SYS_TEMP": "MotherBoard",
    "System": "MotherBoard",
    "VBAT": "MotherBoard"

}

# Optional: order parameters in legends / heatmaps
# Smaller number = earlier
PARAMETER_SORT_ORDER = {
    "FPGA_FUSION":1,
    "FPGA_XENON":2,
    "PROTEGO_FPGA_FUSION":3,
    "TEMP_ACB_FPGA_0":4,
    "CB_TEMP": 5,
    "FPGA_KRYPTONA": 6,
    "FPGA_NEON0": 7,
    "FPGA_NEON1": 8,
    "FPGA_PHILAE": 9,
    "FPGA_KRYPTONB": 10,
    "FPGA_TESLA": 11,
    "TEMP_CB_FPGA_0": 12,
    "CPU1_TEMP": 13,
    "CPU2_TEMP": 14,
    "CPU_TEMP": 15,
    "SYS1_TEMP": 16,
    "SYS2_TEMP": 17,
    "SYS_TEMP": 18,
    "System": 19,
    "TEMP_PREG_BOARD_0": 20,
    "PREG_TEMP": 21,
    "PREG_FAN_1":22,
    "PREG_FAN_2":23,
    "PREG_FAN_3":24,
    "CPU_FAN":25,
    "PREG_FAN_1":26,
    "PREG_FAN_2":27,
    "PREG_FAN_3":28,
    "+1.5V":29,
    "+12V":30,
    "+5V":31,
    "+3.3V":32,
    "+3.3VSB":33,
    "+3VSB":34,
    "+5V":35,
    "+5VSB":36,
    "Battery":37,
    "CPU":38,
    "AVCC":39,
    "CPU1_DMM":40,
    "CPU1":41,
    "CPU2":42,
    "CPU2_DMM":43,
    "VBAT":44,
    "D:":45,
    "E:":46,
    "J:":47,
    "K:":48,
    "R:":49
}

PARAM_GROUP_ORDER = ["Temperature", "Fan Speed", "Voltage", "PercentAvailableFreeSpace"]
SENSOR_GROUP_ORDER = ["ACB", "CB", "IMB", "PercentAvailableFreeSpace", "MotherBoard"]

# =========================================================
# CORE PIPELINE
# =========================================================
@st.cache_data
def load_data(uploaded_file, ehafile) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=",")
    df["SamplingTimestamp"] = pd.to_datetime(df["SamplingTimestamp"])
    df = df[df["SysID"].isin(["795201--USO16B0454","795200--US418B0134","795200--US918B1315","795231--US322B0813","795231--US822B2381","795231--US722B1669","795231--USD21B1076","795231--USO22B1377"])]
    EHA = pd.read_csv(ehafile, sep=",")
    EHA["EventTimestamp"] = pd.to_datetime(EHA["EventTimestamp"])
    return df, EHA

def derive_sensor_parameter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Default derivation
    df["SENSOR"] = df["SamplingID"]
    df["PARAMETER"] = df["SampleID"]

    # Hard disk override
    hd_ids = ["D:", "E:", "J:", "K:", "R:"]
    mask_hd = df["SamplingID"].isin(hd_ids)
    df.loc[mask_hd, "SENSOR"] = "PercentAvailableFreeSpace"
    df.loc[mask_hd, "PARAMETER"] = df.loc[mask_hd, "SamplingID"]

    return df

def make_clean_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "SystemName",
        "SysID",
        "SamplingID",
        "SampleID",
        "SENSOR",
        "PARAMETER",
        "SamplingDate",
        "SamplingTimestamp",
        "SampleValue",
    ]

    clean = (
        df[cols]
        .sort_values(cols)
        .drop_duplicates()
    )
    return clean

def classify_param_group(row) -> str:
    key = (str(row["SamplingID"]), str(row["SampleID"]))
    return PARAM_GROUP_MAP.get(key, "Other")

def classify_sensor_group(row) -> str:
    parameter = str(row["PARAMETER"])
    return SENSOR_GROUP_MAP.get(parameter, "MB")

def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PARAM_GROUP"] = df.apply(classify_param_group, axis=1)
    df["SENSOR_GROUP"] = df.apply(classify_sensor_group, axis=1)
    return df

def sort_key_parameter(param_name: str):
    return (PARAMETER_SORT_ORDER.get(str(param_name), 999999), str(param_name))

def make_series_label(row) -> str:
    return f"{row['SENSOR']} | {row['PARAMETER']}"

def make_series_label_with_group(row) -> str:
    return f"{row['PARAM_GROUP']} | {row['SENSOR']} | {row['PARAMETER']}"

def to_wide_filtered(clean: pd.DataFrame, series_col: str = "SERIES") -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()

    wide = (
        clean.pivot_table(
            index=["SysID", "SamplingDate", "SamplingTimestamp"],
            columns=series_col,
            values="SampleValue",
            aggfunc="first"
        )
        .reset_index()
    )

    return wide

def reorder_wide_columns(wide: pd.DataFrame, ordered_series: list[str]) -> pd.DataFrame:
    if wide.empty:
        return wide

    fixed_cols = ["SysID", "SamplingDate", "SamplingTimestamp"]
    existing_series = [c for c in ordered_series if c in wide.columns]
    remaining = [c for c in wide.columns if c not in fixed_cols and c not in existing_series]

    return wide[fixed_cols + existing_series + remaining]

def corr_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()

    numeric = wide.select_dtypes(include="number")
    if numeric.empty or numeric.shape[1] < 2:
        return pd.DataFrame()

    return numeric.corr(method="pearson")

def plot_corr(corr: pd.DataFrame):
    if corr.empty:
        styled_header("AQUIIII")
        fig = px.imshow(
            [[0]],
            text_auto=False,
            color_continuous_scale="RdBu_r",
            labels=dict(x="No data", y="No data")
        )
        return fig

    fig = px.imshow(
        corr,
        text_auto=False,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(tickangle=-90)
    return fig

# =========================================================
# APP
# =========================================================


# Option 1: file uploader
uploaded = "C:/Users/320303731/OneDrive - Philips/Documents/Models/260305_ALLSENSORS_ACB_WITH_ALERTS.txt"
ehafile = "C:/Users/320303731/OneDrive - Philips/Documents/Models/20260312_EHACodesACB.txt"

df, EHA = load_data(uploaded, ehafile)
raw = df.copy()

# =========================================================
# FILTER BY SYSTEM
# =========================================================
system_names = sorted(df["SystemName"].dropna().unique())
if not system_names:
    st.warning("No SystemName values found in the dataset.")
    st.stop()

system_name = st.sidebar.selectbox("SystemName", system_names)

raw_sys = raw[raw["SystemName"] == system_name].copy()
if raw_sys.empty:
    st.warning("No rows for the selected SystemName.")
    st.stop()

# =========================================================
# PIPELINE
# =========================================================
df = derive_sensor_parameter(raw_sys)
clean_sys = make_clean_long(df)
clean_sys = add_groups(clean_sys)

# Optional SysID filter if multiple SysIDs exist under same SystemName
sysids_for_system = sorted(clean_sys["SysID"].dropna().unique())
selected_sysid = None

if len(sysids_for_system) > 1:
    selected_sysid = st.sidebar.selectbox(
        "SysID (subset within selected SystemName)",
        sysids_for_system
    )
    clean_sys = clean_sys[clean_sys["SysID"] == selected_sysid].copy()

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Timeline", "Correlation", "Availability","Univariate graphs for systems with peaks and alerts"])

# =========================================================
# TAB 1 - OVERVIEW
# =========================================================
with tab1:
    st.write("SystemName:", system_name)
    if selected_sysid is not None:
        st.write("SysID:", selected_sysid)

    st.write("Rows:", len(clean_sys))
    st.write(
        "Sensors:", clean_sys["SENSOR"].nunique(),
        "Parameters:", clean_sys["PARAMETER"].nunique()
    )

    if not clean_sys.empty:
        st.write(
            "Time range:",
            clean_sys["SamplingTimestamp"].min(),
            "→",
            clean_sys["SamplingTimestamp"].max()
        )
        st.dataframe(clean_sys.head(50), width='stretch')
    else:
        st.info("No data after filtering.")

    with st.expander("Check unique combinations"):
        combos = (
            clean_sys[["SamplingID", "SampleID", "SENSOR", "PARAMETER"]]
            .drop_duplicates()
            .sort_values(["SamplingID", "SampleID"])
        )
        st.dataframe(combos, width='stretch')

    with st.expander("Check group assignment"):
        check_df = (
            clean_sys[
                ["SamplingID", "SampleID", "SENSOR", "PARAMETER", "PARAM_GROUP", "SENSOR_GROUP"]
            ]
            .drop_duplicates()
            .sort_values(["PARAM_GROUP", "SENSOR_GROUP", "SamplingID", "SampleID"])
        )
        st.dataframe(check_df, width='stretch')

# =========================================================
# TAB 2 - TIMELINE (4 charts in 2x2 grid)
# =========================================================
with tab2:
    styled_subheader("Timeline by parameter group")

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    grid_cols = [c1, c2, c3, c4]

    PARAM_GROUP_COLORS = {
        "Temperature": px.colors.qualitative.Set1,
        "Fan Speed": px.colors.qualitative.Set2,
        "Voltage": px.colors.qualitative.Set3,
        "PercentAvailableFreeSpace": px.colors.qualitative.Dark24,
    }

    for col, param_group in zip(grid_cols, PARAM_GROUP_ORDER):
        plotdf = clean_sys[clean_sys["PARAM_GROUP"] == param_group].copy()

        with col:
            styled_header(f"{param_group}")

            if plotdf.empty:
                st.info(f"No data for {param_group}.")
            else:
                plotdf["PARAMETER_SORT"] = plotdf["PARAMETER"].map(
                    lambda x: PARAMETER_SORT_ORDER.get(str(x), 999999)
                )
                plotdf = plotdf.sort_values(
                    ["PARAMETER_SORT", "PARAMETER", "SamplingTimestamp"]
                )

                # Calculate time differences per parameter
                plotdf["time_diff"] = plotdf.groupby("PARAMETER")["SamplingTimestamp"].diff()
                sysid = plotdf["SysID"].dropna().unique().tolist()
                syslabel = ",".join(map(str, sysid))
                EHAE = EHA[EHA["ms_sysid"].isin(sysid)]

                # Insert None for large gaps (e.g., > 1 day)
                threshold = pd.Timedelta(days=1)
                plotdf.loc[plotdf["time_diff"] > threshold, "SampleValue"] = None

                color_sequence = PARAM_GROUP_COLORS.get(param_group, px.colors.qualitative.Plotly)

                fig = px.scatter(
                    plotdf,
                    x="SamplingTimestamp",
                    y="SampleValue",
                    color="PARAMETER",
                    #line_group="PARAMETER",
                    color_discrete_sequence=color_sequence,
                    hover_data=["SENSOR", "PARAMETER", "SamplingID", "SampleID"]
                )
                legend = dict(orientation="h", yanchor="bottom",xanchor="center",x=0.5,y=-0.5,font=dict(size=13),itemwidth=30)
                fig.update_layout(
                    #height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    # Title: center + bigger font
                    title=dict(
                        text=f"<b>{system_name}</b><br><sup>ID: {sysid}</sup>",
                        x=0.5, xanchor="center",
                        font=dict(size=22)  # title font size
                    ),
                    legend=legend,xaxis_title = None, yaxis_title = None,
                    # Global base font (legend, hover, etc.)
                    font=dict(size=13),
                )
                # Axis title font sizes + tick label font sizes
                fig.update_xaxes(
                    title_text="SamplingTimestamp",  # optional: override label
                    title_font=dict(size=16),  # x-axis title size
                    tickfont=dict(size=12),  # x-axis tick size
                    tickangle=-90
                )
                fig.update_yaxes(
                    title_text="SampleValue",  # optional: override label
                    title_font=dict(size=16),  # y-axis title size
                    tickfont=dict(size=12)  # y-axis tick size
                )

                st.plotly_chart(fig, width='stretch')

                with st.expander("Download data"):
                    st.write("Download the dataset used to generate the graph.")

                    # Convert dataframe to CSV
                    csv = plotdf.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{sysid}_{param_group}.csv",
                        mime="text/csv", key=f"download_{param_group}_{str(sysid)}"
                    )

# =========================================================
# TAB 3 - CORRELATION
# =========================================================
with tab3:
    st.write("Correlation analysis")

    time_window_mode = st.radio("Time window", ["All data", "Last 24h", "Last 3d", "Last 7d", "Last 30d", "Custom range"], horizontal=True)
    min_ts = clean_sys["SamplingTimestamp"].min()
    max_ts = clean_sys["SamplingTimestamp"].max()

    if time_window_mode == "Custom range":
        start_date = st.date_input("Start date", min_ts.date())
        end_date = st.date_input("End date", max_ts.date())

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    elif time_window_mode == "Last 24h":
        end_ts = max_ts
        start_ts = max_ts - pd.Timedelta(hours=24)

    elif time_window_mode == "Last 3d":
        end_ts = max_ts
        start_ts = max_ts - pd.Timedelta(days=3)

    elif time_window_mode == "Last 7d":
        end_ts = max_ts
        start_ts = max_ts - pd.Timedelta(days=7)

    elif time_window_mode == "Last 30d":
        end_ts = max_ts
        start_ts = max_ts - pd.Timedelta(days=30)

    else:
        start_ts = min_ts
        end_ts = max_ts

    clean_sys_window = clean_sys[
        (clean_sys["SamplingTimestamp"] >= start_ts) &
        (clean_sys["SamplingTimestamp"] <= end_ts)
        ].copy()

    sub1, sub2, sub3 = st.tabs([
        "Per parameter",
        "Per sensor",
        "All sensor-parameters by device"
    ])

    st.caption(f"Using {len(clean_sys_window)}  rows from{start_ts} -> {end_ts}")

    # -----------------------------------------------------
    # 3.1 Per parameter
    # -----------------------------------------------------
    with sub1:
        styled_header("Correlation by parameter group")

        c1, c2, c3, c4 = st.columns(4)
        grid_cols = [c1, c2, c3, c4]

        for col, param_group in zip(grid_cols, PARAM_GROUP_ORDER):
            subset = clean_sys_window[clean_sys_window["PARAM_GROUP"] == param_group].copy()

            with col:
                styled_subheader(f"{param_group}")

                if subset.empty:
                    st.info(f"No data for {param_group} in selected window.")
                else:
                    subset["SERIES"] = subset.apply(make_series_label, axis=1)

                    unique_series = (
                        subset[["SERIES", "PARAMETER"]]
                        .drop_duplicates()
                        .sort_values(
                            by="PARAMETER",
                            key=lambda s: s.map(lambda x: sort_key_parameter(str(x)))
                        )
                    )
                    ordered_series = unique_series["SERIES"].tolist()

                    wide_group = to_wide_filtered(subset, series_col="SERIES")
                    wide_group = reorder_wide_columns(wide_group, ordered_series)

                    corr = corr_matrix(wide_group)

                    if corr.empty or corr.shape[0] < 2:
                        st.info("Not enough numeric data.")
                    else:
                        fig = plot_corr(corr)
                        ##fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                        legend = dict(orientation="h", yanchor="bottom",xanchor="center",x=0.5,y=-0.1,font=dict(size=13),itemwidth=30)
                        fig.update_layout(legend=legend, xaxis_title=None, yaxis_title=None)
                        fig.update_xaxes(tickangle=-90)
                        st.plotly_chart(fig, width='stretch')

    # -----------------------------------------------------
    # 3.2 Per sensor
    # -----------------------------------------------------
    with sub2:
        styled_header(f"Correlation by sensor group - {sysid}")

        s1, s2,s3, s4, s5 = st.columns(5)
        sensor_cols = [s1, s2, s3, s4, s5]

        for col, sensor_group in zip(sensor_cols, SENSOR_GROUP_ORDER):
            subset = clean_sys_window[clean_sys_window["SENSOR_GROUP"] == sensor_group].copy()

            with col:
                styled_subheader(f"{sensor_group}")

                if subset.empty:
                    st.info(f"No data for {sensor_group} in selected window.")
                else:
                    subset["SERIES"] = subset.apply(make_series_label, axis=1)

                    unique_series = (
                        subset[["SERIES", "PARAMETER"]]
                        .drop_duplicates()
                        .sort_values(
                            by="PARAMETER",
                            key=lambda s: s.map(lambda x: sort_key_parameter(str(x)))
                        )
                    )
                    ordered_series = unique_series["SERIES"].tolist()

                    wide_group = to_wide_filtered(subset, series_col="SERIES")
                    wide_group = reorder_wide_columns(wide_group, ordered_series)

                    corr = corr_matrix(wide_group)

                    if corr.empty or corr.shape[0] < 1:
                        st.info("Not enough numeric data.")
                    else:
                        fig = plot_corr(corr)
                        legend = dict(orientation="h", yanchor="bottom",xanchor="center",x=0.5,y=-0.1,font=dict(size=13),itemwidth=30)
                        fig.update_layout(legend=legend,xaxis_title=None, yaxis_title=None)
                        fig.update_xaxes(tickangle=-90, title_text=None)
                        fig.update_yaxes(title_text=None)

                        st.plotly_chart(fig, width='stretch',key=f"plCN_{sensor_group}")

    # -----------------------------------------------------
    # 3.3 All sensor-parameters by device
    # -----------------------------------------------------
    with sub3:
        styled_header("All sensor-parameters by device")
        st.caption("Ordered Temperature → Fan Speed → Voltage, excluding Hard Disk Space")

        subset = clean_sys_window[
            clean_sys_window["PARAM_GROUP"].isin(["Temperature", "Fan Speed", "Voltage"])
        ].copy()

        if subset.empty:
            st.info("No data for Temperature / Fan Speed / Voltage in selected window.")
        else:
            group_rank = {
                "Temperature": 1,
                "Fan Speed": 2,
                "Voltage": 3,
            }

            subset["GROUP_RANK"] = subset["PARAM_GROUP"].map(group_rank)
            subset["SERIES"] = subset.apply(make_series_label_with_group, axis=1)

            unique_series = (
                subset[["SERIES", "GROUP_RANK", "SENSOR", "PARAMETER"]]
                .drop_duplicates()
            )

            unique_series["PARAMETER_ORDER"] = unique_series["PARAMETER"].map(
                lambda x: PARAMETER_SORT_ORDER.get(str(x), 999999)
            )

            unique_series = unique_series.sort_values(
                ["GROUP_RANK", "SENSOR", "PARAMETER_ORDER", "PARAMETER"]
            )

            ordered_series = unique_series["SERIES"].tolist()

            wide_all = to_wide_filtered(subset, series_col="SERIES")
            wide_all = reorder_wide_columns(wide_all, ordered_series)

            corr = corr_matrix(wide_all)

            if corr.empty or corr.shape[0] < 2:
                st.info("Not enough numeric data.")
            else:
                fig = plot_corr(corr)
                fig.update_xaxes(tickangle=-90)
                legend = dict(orientation="h", yanchor="bottom",xanchor="center",x=0.5, y=-0.1,font=dict(size=13),itemwidth=30)
                fig.update_layout(legend=legend, xaxis_title=None, yaxis_title=None, height=850)
                st.plotly_chart(fig, width='stretch')

# =========================================================
# TAB 4 - AVAILABILITY
# =========================================================
with tab4:
    base_df = df.copy()

    if selected_sysid is not None:
        base_df = base_df[base_df["SysID"] == selected_sysid].copy()

        sensor_matrix = (
        base_df.assign(HasData=base_df["SampleValue"].notnull())
        .pivot_table(
            index="SystemName",
            columns="SampleID",
            values="HasData",
            aggfunc="any",
            fill_value=False
        )
        )

        st.write("Sensor availability (True means at least one data point present):")
        st.dataframe(sensor_matrix.astype(bool), width='stretch')

    else:
        st.info("System not selected.")

    with tab5:

        uploaded_file = "C:/Users/320303731/Downloads/['795231--US822B2381']_Temperature.csv"
