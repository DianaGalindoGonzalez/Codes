import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go


st.set_page_config(layout="wide")

def styled_header(text: str, color: str = "#00126E", size: str = "24px"):
        st.markdown(f"<h4 style='color:{color}; font-size:{size};'>{text}</h4>", unsafe_allow_html=True)

def styled_subheader(text: str, color: str = "#0B5ED7", size: str = "18px"):
    st.markdown(f"<h3 style='color:{color}; font-size:{size};'>{text}</h3>", unsafe_allow_html=True)


# Example usage
styled_header("Sensor data analysis models: Hard Disks")

data = pd.read_csv("C:/Users/320303731/OneDrive - Philips/Documents/Result_25.txt", sep = ",")
data['SamplingDate'] = pd.to_datetime(data['SamplingDate'])
df = data.drop_duplicates()

# Dictionaries for dynamic thresholds
value_thresholds = {
'D:': 5,
'E:': 15,
'J:': 5,
'K:': 20,
'R:': 5,
'S:': 5
}
window_hours = {
'D:': 48,
'E:': 6,
'J:': 48,
'K:': 6,
'R:': 48,
'S:': 48
}
limit = {
'D:': 1,
'E:': 1,
'J:': 1,
'K:': 1,
'R:': 1,
'S:': 1
}

def check_alert(group):
    group = group[['CatalogNumber','SampleID','SamplingId','SampleCategory','SampleUnits','SamplingDate','MIN']]
    group = group.drop_duplicates()
    group['SamplingDate'] = pd.to_datetime(group['SamplingDate'])
    group = group.sort_values('SamplingDate').set_index('SamplingDate')

    # Boolean mask for SampleValue <= threshold
    mask = (group['MIN'] <= t).astype(int)

    # Rolling count over 48-hour window
    rolling_counts = mask.rolling(f'{w}h').sum()

    # If any rolling window has at least 2 occurrences, flag alert
    return (rolling_counts >= l).any()

tab1, tab2, tab3 = st.tabs(["K: Individual", "K: Per catalog number", "Number of Days"])

with tab1:
    # Select partition
    #selected_partition_tab2 = st.selectbox("Select partition:", df['SamplingID'].sort_values().unique(), key='key_tab3')

    w = window_hours.get('K:', "N/A")
    l = limit.get('K:', "N/A")
    t = value_thresholds.get('K:', "N/A")

    REC_df = df[df['SamplingId'] == 'K:'].copy()
    # REC_df = df[df['SamplingID'] == 'R:'].copy()

    alerts_calcR = REC_df.groupby('SysID').apply(check_alert, include_groups=False)
    alerts_calcR = alerts_calcR[alerts_calcR == True]
    alerts_REC = REC_df.loc[
        (REC_df['SysID'].isin(alerts_calcR.index)) & (REC_df['MIN'] <= t)].drop_duplicates()

    list_devicesREC = alerts_REC[['SamplingId', 'CatalogNumber', 'SysID']].drop_duplicates().sort_values(by='SysID')
    list_devicesREC['#'] = range(1, len(list_devicesREC) + 1)

    #st.text("Values per partition selected:")
    #m1, m2, m3, m4 = st.columns(4)

    #m1.metric('Threshold (%)', t, border=True)
    #m2.metric('Time window (hours)', w, border=True)
    #m3.metric('Limit', l, border=True)
    #m4.metric('Number of systems', len(list_devicesREC), border=True)

    #styled_subheader("Systems reporting alert using recommended values:")
    st.dataframe(list_devicesREC[['#', 'SamplingId', 'CatalogNumber', 'SysID']], hide_index=True)

    styled_subheader("Timeline per system")

    for uid in list_devicesREC['SysID'].unique():
        subset_rec = alerts_REC[alerts_REC['SysID'] == uid].copy()
        subset_rec['SamplingDate'] = pd.to_datetime(subset_rec['SamplingDate'])

        fig_rec = px.scatter(
            subset_rec,
            x='SamplingDate',
            y='MIN',
            trendline='ols',
            trendline_color_override="#d455f8",
            title=f'Percentage of available free space timeline for {uid}'
        )

        fig_rec.update_layout(
            title_font=dict(size=24),
            font=dict(size=16),
            legend=dict(font=dict(size=14)),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18)
        )

        fig_rec.update_traces(marker=dict(size=3, symbol="diamond", color='#A80DF2'))
        fig_rec.update_xaxes(ticks="inside", tickangle=270)

        fig_rec.add_hline(
            y=t,
            line_color="#4a0336",
            annotation_text=f"Alert threshold: {t}",
            annotation_position="top left",
            layer="below",
            line_width=4
        )

        st.plotly_chart(fig_rec, key=f"pl_sys_{uid}")

with tab2:
    palette = px.colors.qualitative.Safe  # or 'D3', 'Set2', 'Bold', 'Pastel', 'Dark24', etc.

    for i, CN in enumerate(list_devicesREC['CatalogNumber'].unique()):
        subset_rec = alerts_REC[alerts_REC['CatalogNumber'] == CN].copy()
        subset_rec['SamplingDate'] = pd.to_datetime(subset_rec['SamplingDate'])
        summary = subset_rec.groupby("SysID").agg(days_count=("SamplingDate", "nunique"),min_value=("MIN", "min"),max_value=("MAX", "max")).reset_index()

        # Display in Streamlit


        fig_all = px.line(
            subset_rec,
            x="SamplingDate",
            y="MIN",
            color="SysID",
            symbol="SysID",
            markers=True,
            title=f"CatalogNumber {CN} – Min Sample Value per SysID"
        )

        fig_all.update_layout(hovermode="x unified")

        fig_all.add_hline(
            y=t,
            line_color="#4a0336",
            annotation_text=f"Alert threshold: {t}",
            annotation_position="top left",
            layer="below",
            line_width=4
        )

        st.plotly_chart(fig_all, key=f"plCN_{CN}_{i}")
        st.table(summary)

with tab3:
    summary = alerts_REC.groupby("SysID").agg(days_count=("SamplingDate", "nunique"), min_value=("MIN", "min"),
                                              max_value=("MAX", "max")).reset_index()

    idx_min = alerts_REC.groupby("SysID")["MIN"].idxmin()
    idx_max = alerts_REC.groupby("SysID")["MAX"].idxmax()

    min_day = (
        alerts_REC.loc[idx_min, ["SysID", "SamplingDate"]]
        .rename(columns={"SamplingDate": "min_day"})
    )

    max_day = (
        alerts_REC.loc[idx_max, ["SysID", "SamplingDate"]]
        .rename(columns={"SamplingDate": "max_day"})
    )

    min_rows = (alerts_REC.merge(
        summary[["SysID", "min_value"]],
        on="SysID",
        how="inner"
    ).query("MIN == min_value")[["SysID", "SamplingDate"]]
                .groupby("SysID")["SamplingDate"].apply(lambda s: sorted(s.unique()))
                .rename("min_days").reset_index())

    max_rows = (alerts_REC.merge(
        summary[["SysID", "max_value"]],
        on="SysID",
        how="inner"
    ).query("MAX == max_value")[["SysID", "SamplingDate"]]
                .groupby("SysID")["SamplingDate"].apply(lambda s: sorted(s.unique()))
                .rename("max_days").reset_index())

    summary_with_days = (
        summary
        .merge(min_rows, on="SysID", how="left")
        .merge(max_rows, on="SysID", how="left")
    )

    filtered = summary_with_days[(summary_with_days["min_value"] <= 5) & (summary["max_value"] >= 20) ]

    for col in ["min_days", "max_days"]:
        filtered[col] = filtered[col].apply(
            lambda lst: ", ".join(d.strftime("%Y-%m-%d") for d in lst)
        )

    st.dataframe(filtered)

    if not filtered.empty:
        fig = px.histogram(
            filtered,
            x="days_count",
            title="Distribution of days Count",
            labels={"days_count": "Days Count"},
            opacity=0.75
        )
        fig.update_layout(
            bargap=0.1,
            xaxis_title="Days Count",
            yaxis_title="Frequency",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data meets the filter criteria.")








