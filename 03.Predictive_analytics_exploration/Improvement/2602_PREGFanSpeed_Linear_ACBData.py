import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import statistics

from pyexpat import features

st.set_page_config(layout="wide")

def styled_header(text: str, color: str = "#00126E", size: str = "24px"):
        st.markdown(f"<h4 style='color:{color}; text-align: center; font-size:{size};'>{text}</h4>", unsafe_allow_html=True)

def styled_subheader(text: str, color: str = "#0B5ED7", size: str = "18px"):
    st.markdown(f"<h3 style='color:{color}; text-align: center; font-size:{size};'>{text}</h3>", unsafe_allow_html=True)


value_thresholds = {
        'ARGON': 74,
        'XENON': 67,
        'OTHER': 90
    }

type_colors = {
    "ARGON": "#00126E",  #Dark blue - #00126E  -
    "XENON": "#00666F",  #DarkCyan - #00666F -
    "OTHER": "#2c3e50"  # Green - #008800 - Fucsia #DB0383 - Red #D43F44
}

def check_alert_ACBT(group):
    w = 2
    l = 2
    t = value_thresholds.get(statistics.mode(group["Type"]), "N/A")
    group = group[
        ['SampleID', 'SamplingID', 'SampleCategory', 'SampleUnits', 'SamplingDate',
         'SamplingTimestamp', 'SampleValue', 'Type']]
    group = group.drop_duplicates()
    group['SamplingTimestamp'] = pd.to_datetime(group['SamplingTimestamp'])
    group = group.sort_values('SamplingTimestamp').set_index('SamplingTimestamp')

    # Boolean mask for SampleValue <= threshold
    mask = (group['SampleValue'] >= t).astype(int)

    # Rolling count over 48-hour window
    rolling_counts = mask.rolling(f'{w}h').sum()

    # If any rolling window has at least 2 occurrences, flag alert
    return (rolling_counts >= l).any()


styled_header("Sensor data analysis models")

tab1, tab2 = st.tabs(["PREG Fan speed", "ACB Temperature"])

with tab1:
    data_preg = pd.read_csv("C:/Users/320303731/OneDrive - Philips/Documents/PREG_MD.txt", sep=",")
    data_preg['SamplingDate'] = pd.to_datetime(data_preg['SamplingDate'])
    data_preg['uniqueID'] = data_preg['SampleID'].astype(str) + "_" + data_preg['CatalogNumber'].astype(str) + "_" + \
                            data_preg[
                                'SerialNumber'].astype(str)
    df_preg = data_preg.drop_duplicates()

    # Dictionaries for dynamic thresholds
    value_thresholds = {
        'PREG_FAN_1': 1026,
        'PREG_FAN_2': 1053,
        'PREG_FAN_3': 1053
    }
    selected_partition_tab3 = st.selectbox("Select PREG:", df_preg['SampleID'].sort_values().unique(), key='key_tab3')
    t = value_thresholds.get(selected_partition_tab3, "N/A")

    df_distro = df_preg[df_preg['SampleID'] == selected_partition_tab3].copy()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric('Threshold (RPM)', t, border=True)
    m2.metric('Min value (RPM)', df_distro['SampleValue'].min(skipna=True, numeric_only=False), border=True)
    m3.metric('Max value (RPM)', df_distro['SampleValue'].max(skipna=True, numeric_only=False), border=True)
    m4.metric('Median value (RPM)', df_distro['SampleValue'].median(skipna=True, numeric_only=False), border=True)

    fig_SV = px.histogram(
        df_distro,
        x='SampleValue',
        title=f"Distribution of SampleValue for PREG: {selected_partition_tab3}",
    )

    # Add vertical lines at 1023 and 1056
    fig_SV.add_vline(
        x=t,
        line_color="red",
        line_width=2,
        line_dash="dash",
        annotation_text=t,
        annotation_position="top right"
    )
    fig_SV.update_layout(
        # Optional: make sure the annotations don't overlap the title
        margin=dict(t=80)
    )
    st.plotly_chart(fig_SV)


with tab2:
    data_acbT = pd.read_csv("C:/Users/320303731/OneDrive - Philips/Documents/260204_ACB_T.txt", sep=",")
    data_acbT['SamplingDate'] = pd.to_datetime(data_acbT['SamplingDate'])
    data_acbT = data_acbT.drop_duplicates()

    # Dictionaries for dynamic thresholds
    value_thresholds = {
        'ARGON': 74,
        'XENON': 67,
        'OTHER': 90
    }

    selected_type = st.selectbox("Select TYPE:", data_acbT['Type'].sort_values().unique(), key='key_tab3a')
    df_ACB_T = data_acbT[data_acbT['Type'] == selected_type].copy()
    t_acbt = value_thresholds.get(selected_type, "N/A")


    #for i, TYPE in enumerate(data_acbT['Type'].sort_values().unique()):
    #    subset_g = data_acbT[data_acbT['Type'] == TYPE].copy()

    df_ACB_T['SamplingDate'] = pd.to_datetime(df_ACB_T['SamplingDate'])

    m1, m2, m3, m4 = st.columns(4)

    m1.metric('Threshold (ºC)', t_acbt, border=True)
    m2.metric('Min value (ºC)', round(df_ACB_T['SampleValue'].min(skipna=True, numeric_only=False),2), border=True)
    m3.metric('Max value (ºC)', round(df_ACB_T['SampleValue'].max(skipna=True, numeric_only=False),2), border=True)
    m4.metric('Median value (ºC)', round(df_ACB_T['SampleValue'].median(skipna=True, numeric_only=False),2), border=True)

    fig_SV = px.histogram(
        df_ACB_T,
        x='SampleValue',
        title=f"Distribution of Temperature (C) for ACB Type: {selected_type} (Jun 2025 - Jan 2026)",
    )

    fig_SV.add_vline(
        x=t_acbt,
        line_color="red",
        line_width=2,
        line_dash="dash",
        annotation_text=t_acbt,
        annotation_position="top right"
    )
    fig_SV.update_layout(
        margin=dict(t=80)
    )
    st.plotly_chart(fig_SV)

    def check_alert_ACBT(group):
        w = 2
        l = 2
        t = t_acbt
        group = group[
            ['SampleID', 'SamplingID', 'SampleCategory', 'SampleUnits', 'SamplingDate',
             'SamplingTimestamp', 'SampleValue','Type']]
        group = group.drop_duplicates()
        group['SamplingTimestamp'] = pd.to_datetime(group['SamplingTimestamp'])
        group = group.sort_values('SamplingTimestamp').set_index('SamplingTimestamp')

        # Boolean mask for SampleValue <= threshold
        mask = (group['SampleValue'] >= t).astype(int)

        # Rolling count over 48-hour window
        rolling_counts = mask.rolling(f'{w}h').sum()

        # If any rolling window has at least 2 occurrences, flag alert
        return (rolling_counts >= l).any()


    systems_acbT = df_ACB_T.groupby('SysId').apply(check_alert_ACBT, include_groups=False)
    alerts_acbT = systems_acbT[systems_acbT == True]
    #count_df_ACB_T = df_ACB_T.loc[df['SysId'].isin(alerts_acbT.index)].drop_duplicates().groupby('SamplingID')[
    #    'SysId'].nunique().reset_index(name='Count')

    #alerts_calc = DF_df.groupby('uniqueID').apply(check_alert, include_groups=False)
    #alerts_calc = alerts_calc[alerts_calc == True]
    alerts_DF_ACB_T = df_ACB_T.loc[(df_ACB_T['SysId'].isin(alerts_acbT.index)) & (df_ACB_T['SampleValue'] >= t_acbt)].drop_duplicates()

    list_devices = alerts_DF_ACB_T[['SamplingID', 'SysId','Type']].drop_duplicates().sort_values(
        by='Type')
    list_devices['#'] = range(1, len(list_devices) + 1)

    styled_subheader(f"Timeline for ACB {selected_type} systems with alerts (Jun 2025 - Jan 2026)")

    palette = px.colors.qualitative.Safe  # or 'D3', 'Set2', 'Bold', 'Pastel', 'Dark24', etc.

    for i, CN in enumerate(list_devices['SysId'].unique()):
        subset_rec = df_ACB_T[df_ACB_T['SysId'] == CN].copy()
        subset_rec['SamplingTimestamp'] = pd.to_datetime(subset_rec['SamplingTimestamp'])
        # summary = subset_rec.groupby("SysID").agg(days_count=("SamplingDate", "nunique"),min_value=("MIN", "min"),max_value=("MAX", "max")).reset_index()

        fig_all = px.scatter(
            subset_rec,
            x="SamplingDate",
            y="SampleValue",
            color="SysId",
            trendline='ols',
            trendline_color_override="#d455f8",
            #symbol="SysId",
            #markers=True,
            title=f"System {CN} – Temperature (Jun 2025 - Jan 2026)"
        )
        fig_all.update_layout(hovermode="x unified", showlegend=False)

        fig_all.add_hline(
            y=t_acbt,
            line_color="DarkRed",
            annotation_text=f"Alert threshold: {t_acbt}",
            annotation_position="top left",
            layer="below",
            line_width=4
        )

        st.plotly_chart(fig_all, key=f"plCN_{CN}_{i}")
        st.dataframe(subset_rec)
