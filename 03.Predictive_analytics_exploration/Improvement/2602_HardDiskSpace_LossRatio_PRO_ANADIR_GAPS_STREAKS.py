import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from statistics import median
import pickle

import gc
gc.collect()

st.set_page_config(layout="wide")

def styled_header(text: str, color: str = "#00126E", size: str = "24px"):
    st.markdown(f"<h4 style='color:{color}; font-size:{size}; text-align:center;'>{text}</h4>", unsafe_allow_html=True)

def styled_subheader(text: str, color: str = "#0B5ED7", size: str = "18px"):
    st.markdown(f"<h3 style='color:{color}; font-size:{size}; text-align:center;'>{text}</h3>", unsafe_allow_html=True)

def styled_text(text: str, color: str = "#808080", size: str = "18px"):
    st.markdown(f"<p style='color:{color}; font-size:{size};'>{text}</p>", unsafe_allow_html=True)



@st.cache_data(show_spinner="Loading data...", persist="disk")
def load_hd(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)
    # Do your “once-only” cleanup here:
    df["SamplingTimestamp"] = pd.to_datetime(df["SamplingTimestamp"], errors="coerce")
    df["SamplingDate"] = pd.to_datetime(df["SamplingDate"], errors="coerce")
    return df

@st.cache_data(show_spinner="Preparing partition data...", persist="disk")
def prepare_partition(df: pd.DataFrame, sampling_id: str) -> pd.DataFrame:
    out = df[df["SamplingID"] == sampling_id].drop_duplicates()
    return out


df = load_hd('DataHDName.pkl')

#@st.cache_data(show_spinner="Loading file...", persist="disk", max_entries=1, hash_funcs={str: lambda _: None})

#@st.cache_data(show_spinner="Loading data…", max_entries=1)
#def load_csv(path: str):
    #    return pd.read_csv(
    #    path,
    #    sep=",",
    #    engine="pyarrow",
    #    dtype_backend="pyarrow",   # most memory‑efficient
    #)


#data = load_csv("C:/Users/320303731/OneDrive - Philips/Documents/Models/260302_HD.txt")
#data = pd.read_csv("C:/Users/320303731/OneDrive - Philips/Documents/Models/260302_HD.txt", sep=",")
#data['SamplingDate'] = pd.to_datetime(data['SamplingDate'])
#df = data.drop_duplicates()

#df_HD = df[df['SampleCategory'] == 'HardDisk']

#with open("DataHDName.pkl", "wb") as f:
#  pickle.dump(data, f)

#print("Object saved to data.pkl")

#with open("data_sensors.pkl", "wb") as f:
#  pickle.dump(data, f)

#print("Object saved to data.pkl")

#with open("DataHDName.pkl", "rb") as f:
   #loaded_object = pickle.load(f)

#print("Loaded object:", loaded_object)

#df_HD = loaded_object
#df = df_HD.drop_duplicates()

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
'D:': 2,
'E:': 3,
'J:': 2,
'K:': 3,
'R:': 2,
'S:': 2
}

def check_alert(group):
    #group = group[['SystemName','SamplingId','SamplingTimestamp','SampleValue']]
    group = group[['SamplingID', 'SamplingTimestamp', 'SampleValue']]
    group = group.drop_duplicates()
    group['SamplingTimestamp'] = pd.to_datetime(group['SamplingTimestamp'])
    group = group.sort_values('SamplingTimestamp').set_index('SamplingTimestamp')

    # Boolean mask for SampleValue <= threshold
    mask = (group['SampleValue'] <= t).astype(int)

    # Rolling count over 48-hour window
    rolling_counts = mask.rolling(f'{w}h').sum()

    # If any rolling window has at least 2 occurrences, flag alert
    return (rolling_counts >= l).any()

# Select partition
#selected_partition_tab2 = st.selectbox("Select partition:", df['SamplingID'].sort_values().unique(), key='key_tab3')


w = window_hours.get('K:', "N/A")
l = limit.get('K:', "N/A")
t = value_thresholds.get('K:', "N/A")

REC_df = prepare_partition(df,'K:')
#REC_df = df[df['SamplingID'] == 'K:'].copy()
# REC_df = df[df['SamplingID'] == 'R:'].copy()

alerts_calcR = REC_df.groupby('SysID').apply(check_alert, include_groups=False)
alerts_calcR = alerts_calcR[alerts_calcR == True]
#alerts_REC = REC_df.loc[(REC_df['SysID'].isin(alerts_calcR.index)) & (REC_df['SampleValue'] <= t)].drop_duplicates()
alerts_REC = REC_df.loc[(REC_df['SysID'].isin(alerts_calcR.index))].drop_duplicates()

list_devicesREC = alerts_REC[['SamplingID', 'SysID']].drop_duplicates()
list_devicesREC['#'] = range(1, len(list_devicesREC) + 1)

alerts_REC['SamplingTimestamp'] = pd.to_datetime(alerts_REC['SamplingTimestamp'])

df_alerts = alerts_REC.sort_values(['SysID','SamplingTimestamp'])
df_alerts['SamplingTimestamp']  = pd.to_datetime(df_alerts['SamplingTimestamp'])
df_alerts['day'] = df_alerts['SamplingTimestamp'].dt.normalize()

idx=df_alerts.groupby(['SysID','day'])['SampleValue'].idxmin()
df_min = df_alerts.loc[idx,['SysID','day','SampleValue','SystemName']].copy()
df_min = df_min.sort_values(['SysID','day']).reset_index(drop=True)

prev_day = df_min.groupby('SysID')['day'].shift()
new_streak = prev_day.isna()|(df_min['day'].sub(prev_day).dt.days.ne(1))

df_min['streak_id'] = new_streak.groupby(df_min['SysID']).cumsum().astype(int)
df_min['streak_row'] = df_min.groupby(['SysID','streak_id']).cumcount() + 1

df_min = df_min.sort_values(['SysID','day']).copy()
df_min['delta_day'] = (df_min.groupby('SysID')['SampleValue'].diff())
df_min['delta_day'] = df_min['delta_day'].round()

#df_min[df_min['SysID'] == '795117 -- US216B0267'][['day','SampleValue','streak_id','streak_row','delta_day']]

streak_change = (df_min.groupby(['SysID','streak_id'])['SampleValue']
                 .agg(lambda x: x.iloc[-1] - x.iloc[0])
                 .reset_index (name = "delta_streak"))

streak_stats=(df_min.groupby(['SysID','streak_id'])['day']
              .count()
              .reset_index(name='streak_len'))

n_streaks = streak_stats.groupby('SysID')['streak_id'].nunique()
avg_days = streak_stats.groupby('SysID')['streak_len'].median()
avg_delta_day=df_min.groupby('SysID')['delta_day'].median()
avg_delta_streak= streak_change.groupby('SysID')['delta_streak'].median()

summary = pd.concat([n_streaks,avg_days,avg_delta_day,avg_delta_streak], axis=1)
summary.columns=['n_streaks','avg_days','avg_delta_day','avg_delta_streak']

type_table = (
    df_min[['SysID','SystemName']].drop_duplicates()
)

summary = summary.merge(type_table, how='left', on='SysID')

styled_header("Sensor data analysis models - HardDisk K: Partition proactive approach")
styled_subheader("Objective")
st.text(
    """
Given that hard‑disk space consumption is primarily driven by usage patterns rather than a steady daily decline, the goal of this analysis is to offer a proactive monitoring approach. Specifically for HardDisk space, instead of attempting to predict a fixed daily loss, we aim to:

1. Enhance the current alerting logic used in the deployed model with additional data‑driven insights.  
2. Leverage hard‑disk usage data current in Vertica, using the last 6 months as reference (scalable to more data when available).  
3. Use the percentage of available disk space as the main observation unit.
"""
)

st.markdown(
        """
- A **streak** represents a sequence of days where data is available continuously for a particular system.  
  A system may have **multiple streaks**, depending on data availability.
- Hard‑disk space usage tends to be **activity‑driven** rather than linear over time.  
  Therefore, analyzing patterns within each streak helps us understand **how quickly a system consumes disk space** based on actual usage.
"""
 )


colA, colB, colC = st.columns(3)

colA.metric("Total Systems", REC_df['SysID'].nunique(), border=True)
colB.metric("Systems with alerts (01 jul 25 - 02 mar 26)",list_devicesREC.shape[0],border=True)
colC.metric("median of streaks / System", summary.n_streaks.median(),border=True)
#colD.metric("Global Median Decline (%)", round(SummaryHD.global_median_ratio.median()*100, 2),border=True)
#colE.metric("Global Median Decline (units)", round(SummaryHD.global_median_abs.median(), 2),border=True)

styled_subheader("Current data")
fig_all = px.line(
    df_min[df_min['SysID'].isin(list_devicesREC['SysID'].unique())],
    x="day",
    y="SampleValue",
    color="SystemName",    # <-- one color per SystemName
    line_group="SysID",    # <-- different line inside each SystemName group
    symbol="SysID",
    markers=True,
    title="Min SampleValue per SysID and SystemName"
)
fig_all.update_traces(connectgaps=False)
fig_all.update_xaxes(range =["2025-07-01", "2026-02-27"])
st.plotly_chart(fig_all)

daily = df_min.drop('SampleValue',axis=1).copy()
rec = df_alerts.copy()
rec['SamplingDate'] = pd.to_datetime(rec['SamplingDate'])
rec['day'] = rec['SamplingDate'].dt.normalize()

rec = rec.merge(daily, how='left', on=['SysID','day'])


styled_subheader("Data per system - used for proactive approach")

palette = px.colors.qualitative.Safe  # or 'D3', 'Set2', 'Bold', 'Pastel', 'Dark24', etc.

#list_devices_show = ['System 795200--US319B1322','System 795117--US418B0121','System 795210--BZD24F0982']

sys_options= list_devicesREC['SysID'].unique()
selected= st.multiselect("Systems to display",sys_options,default=sys_options[:3])

for i, CN in enumerate(selected):
    #subset_rec = df_alerts[df_alerts['SysID'] == CN].copy()
    #subset_rec['SamplingDate'] = pd.to_datetime(subset_rec['SamplingDate'])
    # summary = subset_rec.groupby("SysID").agg(days_count=("SamplingDate", "nunique"),min_value=("MIN", "min"),max_value=("MAX", "max")).reset_index()

    subset_rec_total = rec[rec["SysID"] == CN].copy()
    subset_rec_total = subset_rec_total.sort_values("SamplingDate")

    # Insert None between streaks
    subset_rec_total["y_plot"] = subset_rec_total["SampleValue"]
    subset_rec_total.loc[subset_rec_total["streak_id"].diff().ne(0), "y_plot"] = None

    ms = summary[summary["SysID"] == CN].iloc[0]
    mind = df_min[(df_min['SysID'] == CN) & (df_min['delta_day'] < 0)]['delta_day'].median()
    a_when = (df_min[df_min['SysID']== CN][['day','SampleValue']]
    .sort_values(by='day', ascending=False)
    ['SampleValue']
    .iloc[0])
    uses_left_int = int(a_when // mind)

    styled_text(f"System {CN}")

    fig = px.line(
        subset_rec_total,
        x="SamplingDate",
        y="y_plot",
        color="SysID",
        markers=True,
        title=f"Min Sample Value per day for SysID-{CN} (All Streaks)"
    )
    fig.update_layout(
        showlegend=False,  # removes the legend
        title_x=0.5,  # centers the title
        title_xanchor="center",  # ensures proper centering across layouts
        hovermode="x unified",
        yaxis_title = "K: Available space (%)"
    )
    st.plotly_chart(fig)

    c7, c8,c9 = st.columns(3)
    c7.metric("Total streaks", ms.n_streaks, border=True)
    c8.metric("K: median delta", mind, border=True)
    c9.metric("Usages of K: to reach 0%",uses_left_int, border=True)

    streaks = subset_rec_total['streak_id'].unique()
    with st.expander("Streak data"):
        for s in sorted(streaks):
            streak_df = subset_rec_total[subset_rec_total['streak_id'] == s]
            st.dataframe(streak_df)

styled_subheader("Improvement proposal")

st.markdown("""
The current alert displays:
""")
st.code("Partition < SEC_Partition> has reached a low percentage of available free space of <SEC_LowPercValue> at <SEC_EventDate>."
        "The threshold of <SEC_ThresholdPerc> has been reached <SEC_NrFailuresBound> times in a time window of <SEC_Window_Width_Hrs> hours in the past <historical_period> days.")

st.markdown("""
We propose to add:
""")

st.code("... According to the log, available space in < SEC_Partition> will reach 100% after <SEC_usage_pro> uses")




