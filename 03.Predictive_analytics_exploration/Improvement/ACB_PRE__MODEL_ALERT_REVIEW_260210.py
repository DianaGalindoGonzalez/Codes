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
    "ARGON": "#00126E",
    "XENON": "#00666F",
    "OTHER": "#2c3e50"
}

# Function to check alerts (equivalent result to US-Sensor data models)
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

    # Rolling count over w-hour window
    rolling_counts = mask.rolling(f'{w}h').sum()

    # If any rolling window has at least 2 occurrences, flag alert
    return (rolling_counts >= l).any()

styled_header("Sensor data analysis models")

styled_subheader("Sensor data modeling exploration: 7-day high ACB temperature alert prediction")

st.markdown(
    """
Objective: Predict whether a device will generate *one or more* high ACB temperature **alerts within the next 7 days**. We explored before training a model:
1. We have enough genuine alerts (not random noise).
2. Alerts show learnable patterns (changes in temp/variance/near-threshold behavior) prior to the alert.
3. Devices have enough historical coverage to build lag/rolling features safely (without leakage).
"""
)

with st.expander("Assumptions we are testing"):
    st.markdown(
        """
- The dataset contains enough history per device to capture behavior (related to the expectations)
- There is a detectable change prior to alert events (trend, near-threshold time).
- Alerts are not purely random; they correlate with measurable temperature dynamics.
"""
    )

def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

tab1, tab2, tab3 = st.tabs(["ACB Temperature - data preview", "EDA","Model"])

with (tab1):
    # ACB temperature data in the last six months
    data_acbT = pd.read_csv("C:/Users/320303731/OneDrive - Philips/Documents/260204_ACB_T.txt", sep=",")
    data_acbT['SamplingDate'] = pd.to_datetime(data_acbT['SamplingDate'])
    data_acbT = data_acbT.drop_duplicates()

    systems_acbT = data_acbT.groupby('SysId').apply(check_alert_ACBT, include_groups=False)
    alerts_acbT = systems_acbT[systems_acbT == True]

    list_devices = pd.DataFrame({'SysId': alerts_acbT.index.drop_duplicates().sort_values()})
    list_devices['#'] = range(1, len(list_devices) + 1)

    data_acbT["Has_alert"] = data_acbT['SysId'].isin(alerts_acbT.index)

    n_sys = data_acbT["SysId"].nunique()
    n_sys_alerts = data_acbT.loc[data_acbT["Has_alert"] == True, "SysId"].nunique()
    percentage = n_sys_alerts / n_sys

    styled_subheader("General")

    c1, c2, c3 = st.columns(3,gap="large")
    c1.metric("Systems with data last semester", n_sys, border=True)
    c2.metric("Systems which have reported alerts", n_sys_alerts,border=True)
    c3.metric("% of systems with an alert", "{:.2%}".format(percentage),border=True)

    styled_subheader("ACB Temperature data distribution per type (Argon, Xenon, Other)")
    c1, c2, c3 = st.columns(3,gap="large")
    colors = px.colors.qualitative.Plotly

    with c1:
        figA = px.histogram(
            data_acbT[data_acbT["Type"] == "ARGON"],
            x="SampleValue",
            color="Has_alert",
            histnorm="probability density",
            barmode="overlay",
            opacity=0.5,
            title="ARGON: Value distribution (alert vs non-alert systems)"
        )
        figA.add_vline(x=74, line_dash="dash",annotation_text=f"Alert threshold: 74")
        st.plotly_chart(figA)

        nsysA = data_acbT.loc[data_acbT['Type'] == "ARGON", ['SysId']].nunique()
        nsysaA = data_acbT.query("Type == 'ARGON' and Has_alert")['SysId'].nunique()
        pA = round(nsysaA / nsysA,2) * 100

        c4, c5, c6 = st.columns(3)
        c4.metric("Systems with data last semester", nsysA, border=True)
        c5.metric("Systems which have reported alerts", nsysaA, border=True)
        c6.metric("% of systems with an alert", pA, border=True)

    with c2:
        figX = px.histogram(
            data_acbT[data_acbT["Type"] == "XENON"],
            x="SampleValue",
            color="Has_alert",
            histnorm="probability density",
            barmode="overlay",
            opacity=0.5,
            title="XENON: Value distribution (alert vs non-alert systems)"
        )
        figX.add_vline(x=67, line_dash="dash",annotation_text=f"Alert threshold: 67")
        st.plotly_chart(figX)

        nsysX = data_acbT.loc[data_acbT['Type'] == "XENON", ['SysId']].nunique()
        nsysaX = data_acbT.query("Type == 'XENON' and Has_alert")['SysId'].nunique()
        pX = round(nsysaX / nsysX,2) * 100

        c7, c8, c9 = st.columns(3)
        c7.metric("Systems with data last semester", nsysX, border=True)
        c8.metric("Systems which have reported alerts", nsysaX, border=True)
        c9.metric("% of systems with an alert", pX, border=True)

    with c3:
        figO = px.histogram(
            data_acbT[data_acbT["Type"] == "OTHER"],
            x="SampleValue",
            color="Has_alert",
            histnorm="probability density",
            barmode="overlay",
            opacity=0.5,
            title="OTHER: Value distribution (alert vs non-alert systems)"
        )

        figO.add_vline(x=90, line_dash="dash",annotation_text=f"Alert threshold: 90")
        figO.for_each_trace(lambda t: t.update(
            marker=dict(color=t.marker.color, opacity=0.5)
        ))

        st.plotly_chart(figO)

        nsysO = data_acbT.loc[data_acbT['Type'] == "OTHER", ['SysId']].nunique()
        nsysaO = data_acbT.query("Type == 'OTHER' and Has_alert")['SysId'].nunique()
        pO = round(nsysaO / nsysO,2) * 100

        c10, c11, c12 = st.columns(3)
        c10.metric("Systems with data last semester", nsysO, border=True)
        c11.metric("Systems which have reported alerts", nsysaO, border=True)
        c12.metric("% of systems with an alert", pO, border=True)

    st.success("ACB Temperature data distributions show alerts are caused not by isolated outliers, but by sustained time near or above the threshold, which supports the hypotheses that alert can be predicted based on the data")
    st.metric("Median alerts per alerting devices", systems_acbT[systems_acbT].shape[0]/ n_sys_alerts)
    st.caption("Alerts are concentrated in a subset of devices indicating non-randon behavior")

    styled_subheader(f"Timeline for ACB systems with alerts (Jun 2025 - Jan 2026)")

    if st.button("Generate timeline per ARGON system (with alerts)"):
        dataA = data_acbT.query("Type == 'ARGON' and Has_alert")
        for i, CN in enumerate(dataA['SysId'].unique()):
            subset_rec = dataA[dataA['SysId'] == CN].copy()
            subset_rec['SamplingTimestamp'] = pd.to_datetime(subset_rec['SamplingTimestamp'])
            # summary = subset_rec.groupby("SysID").agg(days_count=("SamplingDate", "nunique"),min_value=("MIN", "min"),max_value=("MAX", "max")).reset_index()
            fig_all = px.line(
                subset_rec,
                x="SamplingTimestamp",
                y="SampleValue",
                color="SysId",
                trendline='ols',
                trendline_color_override="#d455f8",
                title=f"System {CN} – Temperature (Jun 2025 - Jan 2026)"
            )
            fig_all.update_layout(hovermode="x unified", showlegend=False)
            fig_all.add_hline(
                y=67,
                annotation_text=f"Alert threshold: 67",
                annotation_position="top left",
                layer="below",
                line_width=4
            )
            fig_all.add_vrect(
                x0=subset_rec["SamplingDate"].max() - pd.Timedelta(days=7),
                x1=subset_rec["SamplingDate"].max(),
                fillcolor="orange",
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text="7-day prediction window"
            )
            st.plotly_chart(fig_all, key=f"plCN_{CN}_{i}")
            st.dataframe(subset_rec)

    if st.button("Generate timeline per XENON system (with alerts)"):
        dataX = data_acbT.query("Type == 'XENON' and Has_alert")
        for i, CN in enumerate(dataX['SysId'].unique()):
            subset_rec = dataX[dataX['SysId'] == CN].copy()
            subset_rec['SamplingTimestamp'] = pd.to_datetime(subset_rec['SamplingTimestamp'])
            # summary = subset_rec.groupby("SysID").agg(days_count=("SamplingDate", "nunique"),min_value=("MIN", "min"),max_value=("MAX", "max")).reset_index()
            fig_all = px.scatter(
                subset_rec,
                x="SamplingTimestamp",
                y="SampleValue",
                color="SysId",
                trendline='ols',
                trendline_color_override="#d455f8",
                title=f"System {CN} – Temperature (Jun 2025 - Jan 2026)"
            )
            fig_all.update_layout(hovermode="x unified", showlegend=False)
            fig_all.add_hline(
                y=67,
                annotation_text=f"Alert threshold: 67",
                annotation_position="top left",
                layer="below",
                line_width=4
            )
            fig_all.add_vrect(
                x0=subset_rec["SamplingDate"].max() - pd.Timedelta(days=7),
                x1=subset_rec["SamplingDate"].max(),
                fillcolor="orange",
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text="7-day prediction window"
            )
            st.plotly_chart(fig_all, key=f"plCN_{CN}_{i}")
            st.dataframe(subset_rec)

    if st.button("Generate timeline per OTHER types system (with alerts)"):
        dataO = data_acbT.query("Type == 'OTHER' and Has_alert")
        for i, CN in enumerate(dataO['SysId'].unique()):
            subset_rec = dataO[dataO['SysId'] == CN].copy()
            subset_rec['SamplingTimestamp'] = pd.to_datetime(subset_rec['SamplingTimestamp'])
            # summary = subset_rec.groupby("SysID").agg(days_count=("SamplingDate", "nunique"),min_value=("MIN", "min"),max_value=("MAX", "max")).reset_index()
            fig_all = px.scatter(
                subset_rec,
                x="SamplingTimestamp",
                y="SampleValue",
                color="SysId",
                trendline='ols',
                trendline_color_override="#d455f8",
                title=f"System {CN} – Temperature (Jun 2025 - Jan 2026)"
            )
            fig_all.update_layout(hovermode="x unified", showlegend=False)
            fig_all.add_hline(
                y=67,
                annotation_text=f"Alert threshold: 67",
                annotation_position="top left",
                layer="below",
                line_width=4
            )
            fig_all.add_vrect(
                x0=subset_rec["SamplingDate"].max() - pd.Timedelta(days=7),
                x1=subset_rec["SamplingDate"].max(),
                fillcolor="orange",
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text="7-day prediction window"
            )
            st.plotly_chart(fig_all, key=f"plCN_{CN}_{i}")
            st.dataframe(subset_rec)

with (tab2):
    styled_subheader("Coverage per device (is 7-day forecasting feasible?)")

    st.caption(
        "Taking as a rule of thumb: for a 7-day horizon, devices need >14 days of history to compute stable features."
    )
    # --- Above threshold flag ---
    data_acbT['above_t'] = (
        ((data_acbT['Type'] == "ARGON") & (data_acbT['SampleValue'] >= 74)) |
        ((data_acbT['Type'] == "XENON") & (data_acbT['SampleValue'] >= 67)) |
        ((data_acbT['Type'] == "OTHER") & (data_acbT['SampleValue'] >= 90))
    ).map({True: 1, False: 0})

    days = data_acbT[["SysId","SamplingDate"]].drop_duplicates().sort_values(["SysId","SamplingDate"])
    prev = days.groupby("SysId")["SamplingDate"].shift()
    days["is_new_streak"] = (days["SamplingDate"] != prev + pd.Timedelta(days=1)) | prev.isna()
    days["streak_id"] = days.groupby("SysId")["is_new_streak"].cumsum()

    streaks = (days.groupby(["SysId","streak_id"])
               .agg(start=("SamplingDate", "min"), end=("SamplingDate", "max"), n_days=("SamplingDate", "size"))
               .reset_index()
               )

    max_streak_per_sys = (
        streaks.sort_values(["SysId","n_days","end"], ascending=[True, False, False])
        .groupby("SysId", as_index=False)
        .head(1)
        .drop(columns=["streak_id"])
    )

    data_acbT_maxStreak = data_acbT.merge(max_streak_per_sys,on="SysId", how="left")
    data_acbT_maxStreak["in_max_streak"] = (
        (data_acbT_maxStreak["SamplingDate"] >= data_acbT_maxStreak["start"]) &
        (data_acbT_maxStreak["SamplingDate"] <= data_acbT_maxStreak["end"])
    )

    #data_acbT_maxStreak[data_acbT_maxStreak["in_max_streak"] == True].filter(items=["SysId"]).nunique()

    dfp = data_acbT_maxStreak[(data_acbT_maxStreak['Has_alert'] == True) & (data_acbT_maxStreak['in_max_streak'] == True) & (data_acbT_maxStreak['n_days'] >= 14) ].copy()
    dfp["SamplingDate"] = pd.to_datetime(dfp["SamplingDate"]).dt.normalize()

    presence = (
        dfp.groupby(["SamplingDate", "SysId","Type"])
        .size()
        .reset_index(name="count")
    )
    presence["has_data"] = presence["count"] > 0

    totals = (
        presence[presence["has_data"]]
        .groupby("SysId")["SamplingDate"]
        .nunique()
        .reset_index(name="total_days")
    )

    fig = px.scatter(
        presence[presence["has_data"]],
        x="SamplingDate",
        y="SysId",
        color="Type",
        color_discrete_sequence=["Blue"],
    )
    fig.update_traces(marker=dict(size=8, symbol="circle"))
    fig.update_layout(
        title="Daily data per SysId with alerts and ≥14 days of continuous sampling",
        xaxis_title="Day",
        yaxis_title="SysId",
        showlegend=False,
        xaxis_tickangle=-90,
        height = 800,
    )
    # --- Add total-days labels on the right ---
    last_pos = (
        presence[presence["has_data"]]
        .sort_values(["SysId", "SamplingDate"])
        .groupby("SysId")
        .tail(1)
        .merge(totals, on="SysId", how="left")
    )
    fig.add_scatter(
        x=last_pos["SamplingDate"],
        y=last_pos["SysId"],
        mode="text",
        text=last_pos["total_days"].astype(int).astype(str),
        textposition="bottom right",
        textfont=dict(size=18),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dfp['SysId'].unique())

    def to_dt(s):
        return pd.to_datetime(s, errors="coerce")

    dfp.loc[:, "days_to_alert"] = (
            dfp.groupby("SysId")["SamplingDate"].transform("max")
            - dfp["SamplingDate"]
    ).dt.days

    pre_alert = dfp.query("days_to_alert >= 14 and Has_alert")
    #pre_alert = dfp.query("Has_alert")
    baseline = data_acbT_maxStreak

    figbp = px.box(
        pd.concat([
            pre_alert.assign(window="Pre-alert (≤7d)"),
            baseline.assign(window="Baseline")
        ]),
        x="window",
        y="SampleValue",
        color="Type",
        title="Temperature behavior before alerts vs baseline"
    )
    figbp.update_layout(height = 500)
    st.plotly_chart(figbp)

    for i, s in enumerate(dfp['SysId'].unique()):
        subset_rec = dfp[dfp['SysId'] == s].copy()
        subset_rec['SamplingTimestamp'] = pd.to_datetime(subset_rec['SamplingTimestamp'])
        fig_t14 = px.scatter(
            subset_rec,
            x="SamplingTimestamp",
            y="SampleValue",
            color="SysId",
            #trendline='ols',
            #trendline_color_override="#d455f8",
            title=f"System {s} – Temperature (Jun 2025 - Jan 2026)"
        )
        fig_t14.update_layout(hovermode="x unified", showlegend=False)
        fig_t14.add_vrect(
            x0=subset_rec["SamplingDate"].max() - pd.Timedelta(days=7),
            x1=subset_rec["SamplingDate"].max(),
            fillcolor="orange",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="7-day prediction window"
        )
        st.plotly_chart(fig_t14, key=f"plCN_{s}_{i}")

    st.success("Pre-alert windows show higher temperature levels and variance (ARGON - OTHER) compared to baseline, indicating learnable degradation patterns")

    min_days_needed = 14

    asn = data_acbT_maxStreak[data_acbT_maxStreak["n_days"] >= min_days_needed].filter(items=['SysId']).nunique()
    ahn = data_acbT_maxStreak[data_acbT_maxStreak["Has_alert"] == True].filter(items=['SysId']).nunique()

    pct_ok = asn / ahn
    #(data_acbT_maxStreak["n_days"] >= min_days_needed).mean() * 100
    #st.info(f"*{pct_ok:.1f}%* of devices which alerts have ≥ *{min_days_needed} days* of data.")


with tab3:
    WINDOW = "24h"
    ALERT_THRESHOLD = 67
    PROB_THRESHOLD = 0.5

    # Ensure timestamp is datetime and sort properly
    data_acbT["SamplingTimestamp"] = pd.to_datetime(data_acbT["SamplingTimestamp"])
    data_acbT = data_acbT.sort_values(["SysId", "SamplingTimestamp"])
                                                                                    

    g = data_acbT.groupby("SysId", sort=False)
                                          
                                         

    def rolling_feature(df, col, func):
        return g.apply(
            lambda x: x.assign(
                tmp=x.set_index("SamplingTimestamp")[col]
                .rolling(WINDOW)
                .apply(func, raw=False)
                .values
            )
        )["tmp"].reset_index(drop=True)
         
                                                                                     
                             

                                                                                
                                                                                    
                                          

    data_acbT["temperature_mean"] = rolling_feature(data_acbT, "SampleValue", lambda x: x.mean())
    data_acbT["temperature_std"] = rolling_feature(data_acbT, "SampleValue", lambda x: x.std())
    data_acbT["temperature_min"] = rolling_feature(data_acbT, "SampleValue", lambda x: x.min())
    data_acbT["temperature_max"] = rolling_feature(data_acbT, "SampleValue", lambda x: x.max())

    data_acbT["temp_diff"] = g["SampleValue"].diff()
                            
    data_acbT["temp_slope"] = g.apply(
        lambda x: x.assign(
            tmp=x.set_index("SamplingTimestamp")["SampleValue"]
            .diff()
            .rolling(WINDOW)
            .mean()
            .values
        )
    )["tmp"].reset_index(drop=True)
                             

    data_acbT["dist_to_threshold"] = ALERT_THRESHOLD - data_acbT["SampleValue"]
    data_acbT["pct_to_threshold"] = data_acbT["SampleValue"] / ALERT_THRESHOLD
    data_acbT["above_t"] = data_acbT["SampleValue"] > ALERT_THRESHOLD

    data_acbT["time_above_t"] = g.apply(
        lambda x: x.assign(
            tmp=x.set_index("SamplingTimestamp")["above_t"]
            .rolling(WINDOW)
            .sum()
            .values
        )
    )["tmp"].reset_index(drop=True)

    data_acbT["future_alert"] = g.apply(
        lambda x: x.assign(
            tmp=x.set_index("SamplingTimestamp")["above_t"]
            .rolling(WINDOW)
            .max()
            .shift(-1)
            .fillna(0)
            .astype(int)
            .values
        )
    )["tmp"].reset_index(drop=True)

    features = [
        "SampleValue",
        "temperature_mean", "temperature_std", "temperature_max", "temperature_min",
        "dist_to_threshold", "pct_to_threshold",
        "time_above_t", "temp_diff", "temp_slope"
    ]

    df_model = data_acbT.dropna(subset=features + ["future_alert"])
    X = df_model[features].copy()
    y = df_model["future_alert"]

    timestamps = df_model["SamplingTimestamp"]
    split_time = timestamps.quantile(train_ratio)

    train_mask = timestamps <= split_time
    val_mask = timestamps > split_time

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
                                          

    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]
                                                                             
                                                                 

    # Logistic regression
    from sklearn.metrics import (precision_score, recall_score, average_precision_score)
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if st.button("Train logistic regression"):
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train, y_train)

        prob_val = lr.predict_proba(X_val)[:, 1]
        pred_val = (prob_val > PROB_THRESHOLD).astype(int)
                                                        
                                                           
                                                                                             
                                                                                                                                                                                                                                                       
                              
                           
                                      
                                
                              
                                
                                                   
                                     
             
                                                                          
                              
                     
                                                       
                                               
                              
                            
             
                              
                                                                           
                                                    
                                   
                             
                              
                             
                                                         
             
                                                          
                                    

        precision = precision_score(y_val, pred_val)
        recall = recall_score(y_val, pred_val)
        pr_auc = average_precision_score(y_val, prob_val)
                                                           
                                                                                             
                                                                                                                                                                                                                                                       
                                 
                           
                                      
                                
                              
                                
                                                   
                                     
             
                                                                          
                              
                     
                                                       
                                               
                              
                            
             
                              
                                                                           
                                                    
                                   
                             
                              
                             
                                                         
             
                                                          
                                    

        st.success("Model trained")
                                                                
                                                        
                                                           
                                                                                             
                                                                                                                                                                                                                                                       
                                 
                           
                                      
                                
                              
                                
                                                   
                                     
             
                                                                          
                              
                     
                                                       
                                               
                              
                            
             
                              
                                                                           
                                                    
                                   
                             
                              
                             
                                                         
             
                                                          
                                    

        # ---------------- Plot ----------------
        sysid = st.selectbox("Select SysId:", sorted(df_model["SysId"].unique()))

        sys_df = df_model.loc[val_mask & (df_model["SysId"] == sysid)].copy()
        sys_df["risk_score"] = lr.predict_proba(sys_df[features])[:, 1]
     
                                  
                            
                                                                             
                                                                             
                                                                           
                              

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sys_df["SamplingTimestamp"], sys_df["risk_score"], label="Risk score")
        ax.axhline(PROB_THRESHOLD, linestyle="--", color="r", label="Alert threshold")
                                                                       

        ax.set_ylim(0, 1)
        ax.set_ylabel("Alert probability")
        ax.set_xlabel("Time")
        ax.legend()
        st.pyplot(fig)

        prob_val = lr_prob
        pred_val = (prob_val > ALERT_THRESHOLD).astype(int)
                                         
                
                                    
     

        precision = precision_score(y_val, pred_val)
        recall = recall_score(y_val, pred_val)
        pr_auc = average_precision_score(y_val, prob_val)
                                                                           
     

        c1,c2,c3 = st.columns(3)
        c1.metric("Precision",f"{precision:.3f}")
        c2.metric("Recall", f"{recall:.3f}")
        c3.metric("PR_AUC", f"{pr_auc:.3f}")

        styled_subheader("Prediction timeline")
                                                                            

        sysid = st.selectbox("Select SysId:", data_acbT['SysId'].sort_values().unique(), key='key_tab_sysid')
                                                     
               
                                  
     
                                                

        val_df = data_acbT[val_mask & data_acbT["SysId"] == sysid].copy()
        val_df["risk_score"] = lr.predict_proba(val_df[features])[:,1]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(val_df["SamplingTimestamp"], val_df["risk_score"], label="risk_score")
        ax.axline(alert_threshold, linestyle="dashed", label="Alert threshold")
     

        ax.set_ylim(0,1)
        ax.set_ylabel("Alert probability")
        ax.set_xlabel("Time")
        ax.legend()
                     
                                         
     
                                                           
                      
                                                                                       
                          
                            
                         
                            
                     
     
                                                
                
                                      
                                               
                         
                
                                              
     
                    
                                   
                            
                    
                                                            
                                    
                               
                         
     
                                                  
                                       

        st.pyplot(fig)
                                                 

        styled_subheader("Alert outcomes")
        summary_lr = pd.DataFrame({
            "Predict alert": pred_val,
            "Actual alert": y_val
        })

        st.bar_chart(summary_lr.value_counts().unstack(fill_value=0))
                                       
                                  

    #random forest
    from sklearn.ensemble import RandomForestClassifier
                                                         
                                              
           
                   
                        
                     
                                                              
     
                                     
                          

    if st.button("Train random forest"):
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight='balanced',## Revisar
            random_state=1983
                            
                          
                             
                                                
                                                                     
        )

        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        rf_prob = rf.predict_proba(X_val)[:,1]

        rf_results = {
            'model' : 'Random Forest',
            'precision': precision_score(y_val, rf_pred),
            'recall': recall_score(y_val, rf_pred),
            'pr_auc': average_precision_score(y_val, rf_prob),
        }

        prob_val = rf.predict_proba(X_val)[:, 1]
        pred_val = (prob_val > ALERT_THRESHOLD).astype(int)

        precision = precision_score(y_val, pred_val)
        recall = recall_score(y_val, pred_val)
        pr_auc = average_precision_score(y_val, prob_val)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{precision:.3f}")
        c2.metric("Recall", f"{recall:.3f}")
        c3.metric("PR_AUC", f"{pr_auc:.3f}")

        styled_subheader("Feature importance")

        imp = pd.Series(
            rf.feature_importances_,
            index=features
        ).sort_values()

        fig, ax = plt.subplots()
        imp.plot.barh(ax=ax)
        ax.set_title("Random forest feature importance")
        st.pyplot(fig)

        styled_subheader("Prediction timeline")

        sysid = st.selectbox("Select SysId:", data_acbT['Sysid'].sort_values().unique(), key='key_tab_sysid')

        val_df = df_model[val_mask & df_model["SysId"] == sysid].copy()
        val_df["risk_score"] = rf.predict_proba(val_df[features])[:, 1]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(val_df["SamplingTimestamp"], val_df["risk_score"], label="risk_score")
        ax.axline(alert_threshold, linestyle="dashed", label="Alert threshold")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Alert probability")
        ax.set_xlabel("Time")
        ax.legend()

        st.pyplot(fig)

        styled_subheader("Alert outcomes")
        summary_rf = pd.DataFrame({
            "Predict alert": pred_val,
            "Actual alert": y_val
        })

        st.bar_chart(summary_rf.value_counts().unstack(fill_value=0))



    #st.text(lr_results)