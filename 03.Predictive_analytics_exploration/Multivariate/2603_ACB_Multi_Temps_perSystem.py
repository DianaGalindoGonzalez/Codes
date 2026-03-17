from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(layout="wide", page_title="ACB Temperature Anomaly Detection")


# -----------------------------
# Inputs
# -----------------------------
FOLDER_PATTERN = r"C:/Users/320303731/Downloads/*_Temperature.csv"
EHA_PATH = r"C:/Users/320303731/OneDrive - Philips/Documents/Models/260317_EHAEventsSystemsWAlerts2.txt"

# Raw long-format columns in temperature csv
COL_SYSTEM = "SysID"
COL_TS = "SamplingTimestamp"
COL_SENSOR = "SamplingID"
COL_PARAM = "SampleID"
COL_VALUE = "SampleValue"

# Optional EHA columns
EHA_SYSID_COL = "ms_sysid"      # or "SysID" if that's your file
EHA_TS_COL = "EventTimestamp"
EHA_VAL_COL = "EHACode"

# Modeling options
SERIES_MODE = "SamplingID_SampleID"     # or "SampleID"
MISSING_THRESHOLD = 0.40
TRAIN_FRAC = 0.70
ANOMALY_QUANTILE = 0.95
MIN_ROWS_PER_DEVICE = 40
MIN_FEATURES = 2

# Target selection
TARGET_REGEX = r"(?i)(?:\bFPGA_(?:FUSION|XENON|ARGON)\b.*|TEMP)"


# -----------------------------
# Small UI helpers
# -----------------------------
def header(text: str):
    st.markdown(
        f"<h2 style='text-align:center; color:#00126E; margin-top: 0.5rem;'>{text}</h2>",
        unsafe_allow_html=True
    )


# -----------------------------
# Utilities
# -----------------------------
def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def extract_sysid_from_name(name: str) -> Optional[str]:
    """
    Extract SysID from names like:
    ['795200--US918B1315']_Temperature.csv
    """
    m = re.search(r"\['(.+?)'\]_Temperature\.csv$", name)
    if m:
        return m.group(1)

    m = re.search(r"\[([^\]]+)\]", name)
    if m:
        return m.group(1)

    return None

def is_temperature_like(col: str) -> bool:
    c = str(col).lower()
    return "temp" in c or "temperature" in c

def choose_target_column(columns: List[str], target_regex: str = TARGET_REGEX) -> Optional[str]:
    """
    Prefer regex match for ACB target.
    """
    matches = [c for c in columns if re.search(target_regex, str(c), flags=re.IGNORECASE)]
    if matches:
        return matches[0]
    return None


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_eha(eha_file: str) -> pd.DataFrame:
    if not Path(eha_file).exists():
        return pd.DataFrame()

    df = pd.read_csv(eha_file, sep=",")
    if EHA_TS_COL in df.columns:
        df[EHA_TS_COL] = safe_to_datetime(df[EHA_TS_COL])
    return df

@st.cache_data(show_spinner=False)
def load_csvs_from_pattern(pattern: str) -> List[Tuple[str, pd.DataFrame]]:
    """
    Returns list of (sysid, df)
    """
    paths = glob.glob(pattern)
    out = []

    for p in paths:
        name = Path(p).name
        sysid = extract_sysid_from_name(name)
        if sysid is None:
            continue

        df = pd.read_csv(p, sep=",")
        df[COL_SYSTEM] = sysid
        out.append((sysid, df))

    return out


# -----------------------------
# Wide dataset creation
# -----------------------------
def prepare_2h_wide_dataset(
    df: pd.DataFrame,
    system_col: str,
    ts_col: str,
    sensor_col: str,
    param_col: str,
    value_col: str,
    use_series_as: str = "SamplingID_SampleID",
    target_df: Optional[pd.DataFrame] = None,
    target_ts_col: str = "EventTimestamp",
    target_system_col: str = "SysID",
    target_value_col: str = "EHACode",
) -> pd.DataFrame:
    data = df.copy()
    data[ts_col] = safe_to_datetime(data[ts_col])
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[ts_col, value_col])

    data["ts_2h"] = data[ts_col].dt.floor("2h")

    if use_series_as == "SamplingID_SampleID":
        data["SERIES"] = data[sensor_col].astype(str) + "|" + data[param_col].astype(str)
    else:
        data["SERIES"] = data[param_col].astype(str)

    agg = (
        data.groupby([system_col, "ts_2h", "SERIES"], as_index=False)[value_col]
        .median()
    )

    wide = (
        agg.pivot_table(
            index=[system_col, "ts_2h"],
            columns="SERIES",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
    )

    if target_df is not None and not target_df.empty:
        y = target_df.copy()
        y[target_ts_col] = safe_to_datetime(y[target_ts_col])
        y = y.dropna(subset=[target_ts_col])

        y["ts_2h"] = y[target_ts_col].dt.floor("2h")

        y2 = (
            y.groupby([target_system_col, "ts_2h"], as_index=False)[target_value_col]
            .max()
        )

        wide = wide.merge(
            y2,
            left_on=[system_col, "ts_2h"],
            right_on=[target_system_col, "ts_2h"],
            how="left",
        )

        if target_value_col in wide.columns:
            wide[target_value_col] = pd.to_numeric(
                wide[target_value_col], errors="coerce"
            ).fillna(0)

        if target_system_col != system_col:
            wide = wide.drop(columns=[target_system_col], errors="ignore")

    return wide


# -----------------------------
# Cleaning
# -----------------------------
def clean_wide_dataset(
    wide: pd.DataFrame,
    id_cols: List[str],
    protected_cols: Optional[List[str]] = None,
    missing_threshold: float = 0.4,
) -> pd.DataFrame:
    data = wide.copy()
    protected_cols = protected_cols or []

    feat_cols = [c for c in data.columns if c not in id_cols + protected_cols]

    if not feat_cols:
        return data

    # Drop too-missing columns
    miss = data[feat_cols].isna().mean()
    keep = miss[miss <= missing_threshold].index.tolist()
    data = data[id_cols + keep + protected_cols]
    feat_cols = keep

    if not feat_cols:
        return data

    data = data.sort_values(id_cols)

    # ffill/bfill within system
    system_col = id_cols[0]
    data[feat_cols] = data.groupby(system_col)[feat_cols].ffill().bfill()

    # median fill leftovers
    for c in feat_cols:
        data[c] = data[c].fillna(data[c].median())

    # drop constant columns
    nunique = data[feat_cols].nunique(dropna=False)
    keep2 = nunique[nunique > 1].index.tolist()

    return data[id_cols + keep2 + protected_cols]


# -----------------------------
# Split
# -----------------------------
def time_split_single_system(
    data: pd.DataFrame,
    system_id: str,
    system_col: str = "SysID",
    ts_col: str = "ts_2h",
    train_frac: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = data[data[system_col] == system_id].copy().sort_values(ts_col)
    if df.empty:
        return df, df

    cut = int(len(df) * train_frac)
    cut = max(1, min(cut, len(df) - 1))  # ensure both sides non-empty
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# -----------------------------
# Model
# -----------------------------
def run_temperature_consistency_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_temp_col: str,
    temp_feature_cols: List[str],
    id_cols: List[str],
    anomaly_quantile: float,
):
    features = [c for c in temp_feature_cols if c != target_temp_col]

    if len(features) < MIN_FEATURES:
        raise ValueError(f"Not enough features. Need at least {MIN_FEATURES}, got {len(features)}.")

    X_train = train_df[features]
    y_train = train_df[target_temp_col]
    X_test = test_df[features]
    y_test = test_df[target_temp_col]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_abs_err = np.abs(y_train - pred_train)
    test_abs_err = np.abs(y_test - pred_test)

    threshold = float(np.quantile(train_abs_err, anomaly_quantile))

    results = test_df[id_cols].copy()
    results["actual_temp"] = y_test.values
    results["pred_temp"] = pred_test
    results["abs_residual"] = test_abs_err
    results["temp_anomaly_flag"] = (results["abs_residual"] > threshold).astype(int)

    metrics = {
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "residual_threshold": threshold,
        "n_test_rows": int(len(test_df)),
        "n_flagged": int(results["temp_anomaly_flag"].sum()),
        "n_features": int(len(features)),
    }

    importance = (
        pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return model, results, metrics, importance


# -----------------------------
# Per-device pipeline
# -----------------------------
def build_and_run_for_device(
    sysid: str,
    temp_df: pd.DataFrame,
    eha_df: Optional[pd.DataFrame] = None,
) -> Dict:
    wide = prepare_2h_wide_dataset(
        df=temp_df,
        system_col=COL_SYSTEM,
        ts_col=COL_TS,
        sensor_col=COL_SENSOR,
        param_col=COL_PARAM,
        value_col=COL_VALUE,
        use_series_as=SERIES_MODE,
        target_df=eha_df,
        target_ts_col=EHA_TS_COL,
        target_system_col=EHA_SYSID_COL,
        target_value_col=EHA_VAL_COL,
    )

    target_col_name = EHA_VAL_COL if (eha_df is not None and not eha_df.empty and EHA_VAL_COL in wide.columns) else None

    clean = clean_wide_dataset(
        wide,
        id_cols=[COL_SYSTEM, "ts_2h"],
        protected_cols=[target_col_name] if target_col_name else [],
        missing_threshold=MISSING_THRESHOLD,
    )

    if len(clean) < MIN_ROWS_PER_DEVICE:
        raise ValueError(f"Too few rows after cleaning: {len(clean)}")

    candidate_temp_cols = [
        c for c in clean.columns
        if c not in [COL_SYSTEM, "ts_2h"] + ([target_col_name] if target_col_name else [])
        and is_temperature_like(c)
    ]

    if len(candidate_temp_cols) < 3:
        raise ValueError(f"Too few temperature-like columns: {len(candidate_temp_cols)}")

    target = choose_target_column(candidate_temp_cols)
    if target is None:
        raise ValueError("No ACB target temperature column found.")

    train_df, test_df = time_split_single_system(
        data=clean,
        system_id=sysid,
        system_col=COL_SYSTEM,
        ts_col="ts_2h",
        train_frac=TRAIN_FRAC,
    )

    if len(train_df) < 10 or len(test_df) < 5:
        raise ValueError(f"Train/test split too small: train={len(train_df)}, test={len(test_df)}")

    model, results, metrics, importance = run_temperature_consistency_model(
        train_df=train_df,
        test_df=test_df,
        target_temp_col=target,
        temp_feature_cols=candidate_temp_cols,
        id_cols=[COL_SYSTEM, "ts_2h"],
        anomaly_quantile=ANOMALY_QUANTILE,
    )

    plot_df = results.copy()

    return {
        "sysid": sysid,
        "wide": wide,
        "clean": clean,
        "target": target,
        "candidate_temp_cols": candidate_temp_cols,
        "train_df": train_df,
        "test_df": test_df,
        "model": model,
        "results": results,
        "plot_df": plot_df,
        "metrics": metrics,
        "importance": importance,
    }


# -----------------------------
# Main app
# -----------------------------
header("ACB Temperature – Model based on other temperature sensors (per SysID)")

all_csvs = load_csvs_from_pattern(FOLDER_PATTERN)
eha_all = load_eha(EHA_PATH)

if not all_csvs:
    st.error(f"No temperature files found with pattern: {FOLDER_PATTERN}")
    st.stop()

device_results = {}
device_errors = {}

for sysid, df in all_csvs:
    try:
        df = df.copy()

        # basic checks
        needed = [COL_SYSTEM, COL_TS, COL_SENSOR, COL_PARAM, COL_VALUE]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        eha_for_dev = pd.DataFrame()
        if not eha_all.empty and EHA_SYSID_COL in eha_all.columns:
            eha_for_dev = eha_all[eha_all[EHA_SYSID_COL].astype(str) == str(sysid)].copy()

        device_results[sysid] = build_and_run_for_device(sysid, df, eha_for_dev)

    except Exception as e:
        device_errors[sysid] = str(e)

if not device_results:
    st.error("No device could be modeled successfully.")
    if device_errors:
        st.dataframe(
            pd.DataFrame({
                "SysID": list(device_errors.keys()),
                "Error": list(device_errors.values()),
            }),
            use_container_width=True
        )
    st.stop()


# -----------------------------
# Summary table
# -----------------------------
summary_rows = []
for sysid, res in device_results.items():
    m = res["metrics"]
    summary_rows.append({
        "SysID": sysid,
        "Target": res["target"],
        "Rows_clean": len(res["clean"]),
        "N_features": m["n_features"],
        "MAE_test": m["mae_test"],
        "RMSE_test": m["rmse_test"],
        "Residual_threshold": m["residual_threshold"],
        "N_flagged": m["n_flagged"],
        "N_test_rows": m["n_test_rows"],
    })

summary_df = pd.DataFrame(summary_rows).sort_values(["RMSE_test", "MAE_test"])
st.markdown("### Device summary")
st.dataframe(summary_df, use_container_width=True)

if device_errors:
    with st.expander("Devices skipped"):
        st.dataframe(
            pd.DataFrame({
                "SysID": list(device_errors.keys()),
                "Error": list(device_errors.values()),
            }),
            use_container_width=True
        )


# -----------------------------
# Device selector
# -----------------------------
selected_device = st.selectbox("Choose SysID", summary_df["SysID"].tolist())
res = device_results[selected_device]
plot_df = res["plot_df"]
importance = res["importance"]
metrics = res["metrics"]
target = res["target"]

st.markdown(f"### Selected device: **{selected_device}**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE (test)", f"{metrics['mae_test']:.3f}")
m2.metric("RMSE (test)", f"{metrics['rmse_test']:.3f}")
m3.metric("Residual threshold", f"{metrics['residual_threshold']:.3f}")
m4.metric("# flagged anomalies", f"{metrics['n_flagged']} / {metrics['n_test_rows']}")

tab_data, tab_model, tab_export = st.tabs(["Data", "Model & Plots", "Export"])

with tab_data:
    st.markdown("#### Wide table")
    st.dataframe(res["wide"].head(20), use_container_width=True)
    st.markdown("#### Clean table")
    st.dataframe(res["clean"].head(20), use_container_width=True)

with tab_model:
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=plot_df["ts_2h"],
        y=plot_df["actual_temp"],
        name="Actual",
        mode="lines"
    ))
    fig_ts.add_trace(go.Scatter(
        x=plot_df["ts_2h"],
        y=plot_df["pred_temp"],
        name="Predicted",
        mode="lines"
    ))

    anomalies = plot_df[plot_df["temp_anomaly_flag"] == 1]
    fig_ts.add_trace(go.Scatter(
        x=anomalies["ts_2h"],
        y=anomalies["actual_temp"],
        name="Flagged anomaly",
        mode="markers",
        marker=dict(color="red", size=8, symbol="x"),
    ))

    fig_ts.update_layout(
        title=f"Actual vs Predicted — {target}",
        xaxis_title="Timestamp (2h)",
        yaxis_title=target,
        height=450,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("#### Top feature importances")
    fig_imp = px.bar(
        importance.head(20),
        x="importance",
        y="feature",
        orientation="h",
        height=500
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Absolute residuals")
    fig_res = px.histogram(plot_df, x="abs_residual", nbins=40)
    fig_res.add_vline(
        x=metrics["residual_threshold"],
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
    )
    st.plotly_chart(fig_res, use_container_width=True)

with tab_export:
    out = plot_df.copy()
    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"temp_consistency_results_{selected_device}.csv",
        mime="text/csv",
    )