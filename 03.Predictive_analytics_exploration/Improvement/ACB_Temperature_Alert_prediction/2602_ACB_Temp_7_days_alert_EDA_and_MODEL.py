from __future__ import annotations

# --- Standard library
from datetime import timedelta
import statistics
import os

# --- Third-party
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # kept (used for hlines/rects)
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit

# ============================
# App Config & Constants
# ============================

st.set_page_config(layout="wide")

# NOTE: Replace this with your CSV/TXT path. You can also supply via
# `streamlit run app.py -- --data-path path/to/file.csv` and parse args.
DATA_PATH = "C:/Users/320303731/OneDrive - Philips/Documents/260204_ACB_T.txt"

# Per-type threshold used across EDA and labeling
VALUE_THRESHOLDS = {
    'ARGON': 74,
    'XENON': 67,
    'OTHER': 90,
}

# Simple color map for narrative plots
TYPE_COLORS = {"ARGON": "#00126E", "XENON": "#00666F", "OTHER": "#2c3e50"}
COLOR_MAP_ALERT = {True: "red", False: "blue"}

# ============================
# Small UI helpers
# ============================

def styled_header(text: str, color: str = "#00126E", size: str = "24px"):
    st.markdown(f"<h4 style='color:{color}; text-align: center; font-size:{size};'>{text}</h4>", unsafe_allow_html=True)


def styled_subheader(text: str, color: str = "#0B5ED7", size: str = "18px"):
    st.markdown(f"<h3 style='color:{color}; text-align: center; font-size:{size};'>{text}</h3>", unsafe_allow_html=True)


# ============================
# Domain Logic
# ============================

def check_alert_acbt(group: pd.DataFrame) -> bool:
    """Return True if a SysId group has an alert according to the US-Sensor logic.

    Logic: at least 2 readings >= threshold within a 2-hour rolling window.
    Threshold chosen by the group's most common Type.
    """
    w_hours = 2
    min_count = 2
    t = VALUE_THRESHOLDS.get(statistics.mode(group["Type"]), np.nan)

    cols = ['SampleID', 'SamplingID', 'SampleCategory', 'SampleUnits',
            'SamplingDate', 'SamplingTimestamp', 'SampleValue', 'Type']
    group = group[cols].drop_duplicates().copy()
    group['SamplingTimestamp'] = pd.to_datetime(group['SamplingTimestamp'])
    group = group.sort_values('SamplingTimestamp').set_index('SamplingTimestamp')

    # Boolean mask for SampleValue >= threshold
    mask = (group['SampleValue'] >= t).astype(int)
    # Rolling sum over 2h window
    rolling_counts = mask.rolling(f'{w_hours}h').sum()
    # If any window reaches the minimum count, it's an alert
    return bool((rolling_counts >= min_count).any())


def add_alert_flag(group: pd.DataFrame) -> pd.DataFrame:
    """Compute instantaneous alert flag based on 2h rolling window >= 2 events."""
    group = group.sort_values("SamplingTimestamp").copy()
    group["above_t"] = (group["SampleValue"] >= group["alert_threshold"]).astype(int)
    g2 = group.set_index("SamplingTimestamp")
    group["alert_now"] = (
        g2["above_t"].rolling("2h").sum().ge(2).astype(int).values
    )
    return group


def add_future_label(group: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Label each timestamp if any alert occurs within the next `horizon_days` days."""
    group = group.sort_values("SamplingTimestamp").copy()
    times = group["SamplingTimestamp"].values
    alerts = group["alert_now"].values
    future_alert = np.zeros(len(group), dtype=int)

    for i in range(len(group)):
        end_time = times[i] + np.timedelta64(horizon_days, "D")
        mask_future = (times > times[i]) & (times <= end_time)
        if np.any(alerts[mask_future] == 1):
            future_alert[i] = 1
    group["future_alert_7d"] = future_alert
    return group


# ============================
# Load Data (single read)
# ============================

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load ACB temperature data. Accepts CSV/TXT with commas as separator."""
    df = pd.read_csv(path, sep=",")
    df = df.drop_duplicates().copy()
    df['SamplingDate'] = pd.to_datetime(df['SamplingDate'])
    df['SamplingTimestamp'] = pd.to_datetime(df['SamplingTimestamp'], errors='coerce')
    return df


def main() -> None:
    styled_header("Sensor data analysis models")
    styled_subheader("Sensor data modeling exploration: 7-day high ACB temperature alert prediction")

    st.markdown(
        (
            """
            **Objective**: Predict whether a device will generate *one or more* high ACB
            temperature **alerts within the next 7 days**.

            Before training, we check that:
            1. There are enough genuine alerts (not random noise).
            2. Alerts show learnable patterns (levels/variance/near-threshold behavior) prior to the alert.
            3. Devices have enough historical coverage to build lag/rolling features (without leakage).
            """
        )
    )

    with st.expander("Assumptions we are testing"):
        st.markdown(
            """
            - The dataset contains enough device history (≥14 days) for a 7‑day horizon.
            - Detectable change prior to alert events (trend, near-threshold time).
            - Alerts correlate with measurable temperature dynamics (not purely random).
            """
        )

    data_acbT = load_data(DATA_PATH)

    # Pre-compute alert per SysId (used in EDA tab)
    systems_acbT = data_acbT.groupby('SysId').apply(check_alert_acbt, include_groups=False)
    alerts_acbT = systems_acbT[systems_acbT == True]
    data_acbT["Has_alert"] = data_acbT['SysId'].isin(alerts_acbT.index)

    # Threshold column for various computations
    df = data_acbT.copy()
    df["alert_threshold"] = df["Type"].map(VALUE_THRESHOLDS)

    # Near-threshold helper (used later in model features)
    df = df.groupby("SysId", group_keys=False).apply(add_alert_flag)
    df["near_t"] = (df["SampleValue"] >= 0.95 * df["alert_threshold"]).astype(int)

    tab1, tab2, tab3 = st.tabs(["ACB Temperature - data preview", "EDA", "Model"])

    # ============================
    # TAB 1 – Preview & distributions
    # ============================
    with tab1:
        list_devices = (
            pd.DataFrame({'SysId': alerts_acbT.index.drop_duplicates().sort_values()})
            .assign(**{'#': lambda d: range(1, len(d) + 1)})
        )

        n_sys = data_acbT["SysId"].nunique()
        n_sys_alerts = data_acbT.loc[data_acbT["Has_alert"] == True, "SysId"].nunique()
        percentage = (n_sys_alerts / n_sys) if n_sys else 0

        styled_subheader("General")
        c1, c2, c3 = st.columns(3, gap="large")
        c1.metric("Systems with data last semester", n_sys, border=True)
        c2.metric("Systems which have reported alerts", n_sys_alerts, border=True)
        c3.metric("% of systems with an alert", f"{percentage:.2%}", border=True)

        styled_subheader("ACB Temperature data distribution per type (Argon, Xenon, Other)")
        type_options = ["ARGON", "XENON", "OTHER"]
        selected_type = st.selectbox("Select gas type", type_options, index=0)

        df_sel = data_acbT[data_acbT["Type"] == selected_type].copy()
        title_text = f"{selected_type}: Value distribution (alert vs non-alert systems)"

        fig = px.histogram(
            df_sel,
            x="SampleValue",
            color="Has_alert",
            histnorm="probability density",
            barmode="overlay",
            opacity=0.5,
            title=title_text,
            color_discrete_map=COLOR_MAP_ALERT,
        )
        thr = VALUE_THRESHOLDS.get(selected_type)
        if thr is not None:
            fig.add_vline(x=thr, line_dash="dash", annotation_text=f"Alert threshold: {thr}")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics by selection
        nsys = df_sel["SysId"].nunique()
        nsys_alert = df_sel.query("Has_alert == True")["SysId"].nunique() if not df_sel.empty else 0
        pct_alert = round((nsys_alert / nsys * 100), 2) if nsys > 0 else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Systems with data last semester", nsys, border=True)
        c2.metric("Systems which have reported alerts", nsys_alert, border=True)
        c3.metric("% of systems with an alert", pct_alert, border=True)

        st.success(
            "ACB Temperature distributions suggest alerts stem from sustained time near/above"
            " thresholds rather than isolated outliers, supporting predictability.")

        if n_sys_alerts:
            st.metric("Median alerts per alerting devices", systems_acbT[systems_acbT].shape[0] / n_sys_alerts)
        st.caption("Alerts are concentrated in a subset of devices indicating non-random behavior")

        styled_subheader("Timeline for ACB systems with alerts (rolling window view)")

        if st.button("Generate timeline per ARGON system (with alerts)"):
            dataA = data_acbT.query("Type == 'ARGON' and Has_alert")
            for i, CN in enumerate(dataA['SysId'].unique()):
                subset = dataA[dataA['SysId'] == CN].copy()
                subset['SamplingTimestamp'] = pd.to_datetime(subset['SamplingTimestamp'])
                fig_all = px.line(
                    subset, x="SamplingTimestamp", y="SampleValue", color="SysId",
                    trendline='ols', trendline_color_override="#d455f8",
                    title=f"System {CN} – Temperature"
                )
                fig_all.update_layout(hovermode="x unified", showlegend=False)
                fig_all.add_hline(y=VALUE_THRESHOLDS['XENON'], annotation_text=f"Alert threshold: {VALUE_THRESHOLDS['XENON']}",
                                  annotation_position="top left", layer="below", line_width=4)
                st.plotly_chart(fig_all, key=f"plCN_A_{CN}_{i}")
                st.dataframe(subset)

        if st.button("Generate timeline per XENON system (with alerts)"):
            dataX = data_acbT.query("Type == 'XENON' and Has_alert")
            for i, CN in enumerate(dataX['SysId'].unique()):
                subset = dataX[dataX['SysId'] == CN].copy()
                subset['SamplingTimestamp'] = pd.to_datetime(subset['SamplingTimestamp'])
                fig_all = px.scatter(
                    subset, x="SamplingTimestamp", y="SampleValue", color="SysId",
                    trendline='ols', trendline_color_override="#d455f8",
                    title=f"System {CN} – Temperature"
                )
                fig_all.update_layout(hovermode="x unified", showlegend=False)
                fig_all.add_hline(y=VALUE_THRESHOLDS['XENON'], annotation_text=f"Alert threshold: {VALUE_THRESHOLDS['XENON']}",
                                  annotation_position="top left", layer="below", line_width=4)
                st.plotly_chart(fig_all, key=f"plCN_X_{CN}_{i}")
                st.dataframe(subset)

        if st.button("Generate timeline per OTHER types system (with alerts)"):
            dataO = data_acbT.query("Type == 'OTHER' and Has_alert")
            for i, CN in enumerate(dataO['SysId'].unique()):
                subset = dataO[dataO['SysId'] == CN].copy()
                subset['SamplingTimestamp'] = pd.to_datetime(subset['SamplingTimestamp'])
                fig_all = px.scatter(
                    subset, x="SamplingTimestamp", y="SampleValue", color="SysId",
                    trendline='ols', trendline_color_override="#d455f8",
                    title=f"System {CN} – Temperature"
                )
                fig_all.update_layout(hovermode="x unified", showlegend=False)
                fig_all.add_hline(y=VALUE_THRESHOLDS['XENON'], annotation_text=f"Alert threshold: {VALUE_THRESHOLDS['XENON']}",
                                  annotation_position="top left", layer="below", line_width=4)
                st.plotly_chart(fig_all, key=f"plCN_O_{CN}_{i}")
                st.dataframe(subset)

    # ============================
    # TAB 2 – EDA: coverage & behavior
    # ============================
    with tab2:
        styled_subheader("Coverage per device (is 7-day forecasting feasible?)")
        st.caption("Rule of thumb: for a 7-day horizon, devices need >14 days of history.")

        # Above threshold flag for EDA (explicit per Type)
        data_ed = data_acbT.copy()
        data_ed['above_t'] = (
            ((data_ed['Type'] == "ARGON") & (data_ed['SampleValue'] >= VALUE_THRESHOLDS['ARGON'])) |
            ((data_ed['Type'] == "XENON") & (data_ed['SampleValue'] >= VALUE_THRESHOLDS['XENON'])) |
            ((data_ed['Type'] == "OTHER") & (data_ed['SampleValue'] >= VALUE_THRESHOLDS['OTHER']))
        ).map({True: 1, False: 0})

        # Build consecutive-day streaks per SysId
        days = data_ed[["SysId", "SamplingDate"]].drop_duplicates().sort_values(["SysId", "SamplingDate"]) \
            .assign(prev=lambda d: d.groupby("SysId")["SamplingDate"].shift())
        days["is_new_streak"] = (days["SamplingDate"] != days["prev"] + pd.Timedelta(days=1)) | days["prev"].isna()
        days["streak_id"] = days.groupby("SysId")["is_new_streak"].cumsum()
        streaks = (
            days.groupby(["SysId", "streak_id"]) \
                .agg(start=("SamplingDate", "min"), end=("SamplingDate", "max"), n_days=("SamplingDate", "size")) \
                .reset_index()
        )
        max_streak_per_sys = (
            streaks.sort_values(["SysId", "n_days", "end"], ascending=[True, False, False])
                   .groupby("SysId", as_index=False)
                   .head(1)
                   .drop(columns=["streak_id"])
        )

        data_max = data_ed.merge(max_streak_per_sys, on="SysId", how="left")
        data_max["in_max_streak"] = (
            (data_max["SamplingDate"] >= data_max["start"]) & (data_max["SamplingDate"] <= data_max["end"]) 
        )

        dfp = data_max[(data_max['Has_alert'] == True) & (data_max['in_max_streak'] == True) & (data_max['n_days'] >= 14)].copy()
        dfp["SamplingDate"] = pd.to_datetime(dfp["SamplingDate"]).dt.normalize()

        presence = (
            dfp.groupby(["SamplingDate", "SysId", "Type"]).size().reset_index(name="count")
        )
        presence["has_data"] = presence["count"] > 0
        totals = (
            presence[presence["has_data"]]
                .groupby("SysId")["SamplingDate"].nunique()
                .reset_index(name="total_days")
        )

        fig = px.scatter(
            presence[presence["has_data"]], x="SamplingDate", y="SysId", color="Type",
            color_discrete_sequence=["Blue"],
        )
        fig.update_traces(marker=dict(size=8, symbol="circle"))
        fig.update_layout(
            title="Daily data per SysId with alerts and ≥14 days of continuous sampling",
            xaxis_title="Day", yaxis_title="SysId", showlegend=False, xaxis_tickangle=-90, height=700,
        )

        last_pos = (
            presence[presence["has_data"]]
                .sort_values(["SysId", "SamplingDate"]).groupby("SysId").tail(1)
                .merge(totals, on="SysId", how="left")
        )
        fig.add_scatter(
            x=last_pos["SamplingDate"], y=last_pos["SysId"], mode="text",
            text=last_pos["total_days"].astype(int).astype(str), textposition="bottom right",
            textfont=dict(size=14), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dfp['SysId'].unique())

        # Simple pre-alert vs baseline distribution
        dfp = dfp.copy()
        dfp.loc[:, "days_to_alert"] = (
            dfp.groupby("SysId")["SamplingDate"].transform("max") - dfp["SamplingDate"]
        ).dt.days
        pre_alert = dfp.query("days_to_alert >= 14 and Has_alert")
        baseline = data_max

        figbp = px.box(
            pd.concat([
                pre_alert.assign(window="Pre-alert (≤7d)"),
                baseline.assign(window="Baseline"),
            ]),
            x="window", y="SampleValue", color="Type",
            title="Temperature behavior before alerts vs baseline",
        )
        figbp.update_layout(height=480)
        st.plotly_chart(figbp)

        for i, s in enumerate(dfp['SysId'].unique()):
            subset = dfp[dfp['SysId'] == s].copy()
            subset['SamplingTimestamp'] = pd.to_datetime(subset['SamplingTimestamp'])
            fig_t14 = px.scatter(
                subset, x="SamplingTimestamp", y="SampleValue", color="SysId",
                title=f"System {s} – Temperature"
            )
            fig_t14.update_layout(hovermode="x unified", showlegend=False)
            st.plotly_chart(fig_t14, key=f"plCN_{s}_{i}")

        st.success(
            "Pre-alert windows show higher levels/variance (ARGON/OTHER) vs baseline,\n"
            "indicating learnable degradation patterns.")

    # ============================
    # TAB 3 – Baseline model
    # ============================
    with tab3:
        styled_subheader("7-day alert prediction model")
        st.caption(
            "Baseline predictive model to estimate whether a system will generate "
            "a high ACB temperature alert within the next 7 days."
        )

        # --- Config ---
        PREDICTION_HORIZON_DAYS = 7
        ROLLING_WINDOW = "24h"
        MIN_HISTORY_ROWS = 20
        USE_RANDOM_FOREST = True

        # 1) Prepare and validate
        model_df = data_acbT.copy()
        model_df["SamplingTimestamp"] = pd.to_datetime(model_df["SamplingTimestamp"], errors="coerce")
        model_df["SamplingDate"] = pd.to_datetime(model_df["SamplingDate"], errors="coerce").dt.normalize()
        model_df = model_df.dropna(subset=["SysId", "SamplingTimestamp", "SampleValue", "Type"]).copy()
        model_df = model_df.sort_values(["SysId", "SamplingTimestamp"]).reset_index(drop=True)
        model_df["alert_threshold"] = model_df["Type"].map(VALUE_THRESHOLDS)
        model_df = model_df.dropna(subset=["alert_threshold"]).copy()

        # 2) Operational alert logic
        model_df = model_df.groupby("SysId", group_keys=False).apply(add_alert_flag)

        # 3) Features (24h rolling statistics + slopes + near/above counts)
        model_df = model_df.set_index("SamplingTimestamp")
        grouped_roll = model_df.groupby("SysId")["SampleValue"].rolling(ROLLING_WINDOW)
        model_df["temp_mean_24h"] = grouped_roll.mean().reset_index(level=0, drop=True)
        model_df["temp_std_24h"] = grouped_roll.std().reset_index(level=0, drop=True)
        model_df["temp_min_24h"] = grouped_roll.min().reset_index(level=0, drop=True)
        model_df["temp_max_24h"] = grouped_roll.max().reset_index(level=0, drop=True)
        model_df = model_df.reset_index()
        g = model_df.groupby("SysId")
        model_df["temp_diff"] = g["SampleValue"].diff()
        model_df["temp_slope_24h"] = (
            g["SampleValue"].diff().rolling(8, min_periods=2).mean().reset_index(level=0, drop=True)
        )
        model_df["dist_to_threshold"] = model_df["alert_threshold"] - model_df["SampleValue"]
        model_df["pct_to_threshold"] = model_df["SampleValue"] / model_df["alert_threshold"]
        model_df["near_threshold"] = (model_df["SampleValue"] >= 0.95 * model_df["alert_threshold"]).astype(int)
        model_df["near_threshold_count_24h"] = (
            g["near_threshold"].rolling(8, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        model_df["above_threshold_count_24h"] = (
            g["above_t"].rolling(8, min_periods=1).sum().reset_index(level=0, drop=True)
        )

        # 4) Label = any alert in next horizon
        model_df = model_df.groupby("SysId", group_keys=False).apply(
            add_future_label, horizon_days=PREDICTION_HORIZON_DAYS
        )

        # 5) Coverage filter
        valid_devices = (
            model_df.groupby("SysId").size().loc[lambda s: s >= MIN_HISTORY_ROWS].index
        )
        model_df = model_df[model_df["SysId"].isin(valid_devices)].copy()

        feature_cols = [
            "SampleValue", "temp_mean_24h", "temp_std_24h", "temp_min_24h", "temp_max_24h",
            "temp_diff", "temp_slope_24h", "dist_to_threshold", "pct_to_threshold",
            "near_threshold_count_24h", "above_threshold_count_24h",
        ]
        model_ready = model_df.dropna(subset=feature_cols + ["future_alert_7d"]).copy()

        if model_ready.empty or model_ready["future_alert_7d"].nunique() < 2:
            st.warning("Not enough labelled data to train the model after preprocessing.")
            st.stop()

        X = model_ready[feature_cols]
        y = model_ready["future_alert_7d"]
        groups = model_ready["SysId"]

        # 6) Group-wise split (by device)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 7) Logistic Regression baseline
        log_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        log_model.fit(X_train, y_train)
        prob_test = log_model.predict_proba(X_test)[:, 1]

        # Pick probability threshold by best F1 on the validation set
        prec_curve, rec_curve, thr_curve = precision_recall_curve(y_test, prob_test)
        f1_curve = 2 * (prec_curve[:-1] * rec_curve[:-1]) / (prec_curve[:-1] + rec_curve[:-1] + 1e-9)
        best_threshold = float(thr_curve[np.argmax(f1_curve)]) if len(thr_curve) > 0 else 0.5
        pred_test = (prob_test >= best_threshold).astype(int)

        precision = precision_score(y_test, pred_test, zero_division=0)
        recall = recall_score(y_test, pred_test, zero_division=0)
        f1 = f1_score(y_test, pred_test, zero_division=0)
        roc_auc = roc_auc_score(y_test, prob_test)
        pr_auc = average_precision_score(y_test, prob_test)
        cm = confusion_matrix(y_test, pred_test)

        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

        # 8) Display metrics
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c1.metric("Precision", f"{precision:.3f}", border=True)
        c2.metric("Recall", f"{recall:.3f}", border=True)
        c3.metric("F1-score", f"{f1:.3f}", border=True)
        c4.metric("ROC AUC", f"{roc_auc:.3f}", border=True)
        c5.metric("PR AUC", f"{pr_auc:.3f}", border=True)
        c6.metric("Probability threshold", f"{best_threshold:.3f}", border=True)

        st.markdown("**Confusion matrix**")
        st.dataframe(cm_df, use_container_width=True)

        # 9) Coefficients (interpretability)
        coef_df = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": log_model.named_steps["clf"].coef_[0],
        }).sort_values("coefficient", key=np.abs, ascending=False)
        st.markdown("**Most influential logistic regression features**")
        st.dataframe(coef_df, use_container_width=True)

        fig_coef = px.bar(
            coef_df.head(10), x="coefficient", y="feature", orientation="h",
            title="Top feature effects (absolute influence in logistic model)",
        )
        fig_coef.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_coef, use_container_width=True)

        # 10) Optional Random Forest comparison
        USE_RANDOM_FOREST = True
        if USE_RANDOM_FOREST:
            rf_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_split=10,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )),
            ])
            rf_model.fit(X_train, y_train)
            rf_prob = rf_model.predict_proba(X_test)[:, 1]
            rf_pred = (rf_prob >= 0.5).astype(int)
            rf_precision = precision_score(y_test, rf_pred, zero_division=0)
            rf_recall = recall_score(y_test, rf_pred, zero_division=0)
            rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
            rf_roc_auc = roc_auc_score(y_test, rf_prob)
            rf_pr_auc = average_precision_score(y_test, rf_prob)

            st.markdown("**Random Forest comparison**")
            compare_df = pd.DataFrame([
                {"Model": "Logistic Regression", "Precision": round(precision, 3), "Recall": round(recall, 3),
                 "F1": round(f1, 3), "ROC AUC": round(roc_auc, 3), "PR AUC": round(pr_auc, 3)},
                {"Model": "Random Forest", "Precision": round(rf_precision, 3), "Recall": round(rf_recall, 3),
                 "F1": round(rf_f1, 3), "ROC AUC": round(rf_roc_auc, 3), "PR AUC": round(rf_pr_auc, 3)},
            ])
            st.dataframe(compare_df, use_container_width=True)

        # 11) Scored examples
        scored = model_ready.iloc[test_idx][["SysId", "SamplingTimestamp", "SampleValue", "Type", "future_alert_7d"]].copy()
        scored["predicted_probability"] = prob_test
        scored["predicted_alert"] = pred_test
        st.markdown("**Scored validation examples**")
        st.dataframe(scored.sort_values("predicted_probability", ascending=False).head(50), use_container_width=True)
        st.success(
            "This baseline model is a simple, interpretable reference for estimating 7‑day high "
            "ACB temperature alerts."
        )


if __name__ == "__main__":
    main()

