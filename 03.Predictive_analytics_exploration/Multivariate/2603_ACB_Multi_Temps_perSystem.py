from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import glob
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(layout="wide")
styled_header("Temperature Modeling per system")

FOLDER_PATTERN = r"C:/Users/320303731/Downloads/*_Temperature.csv"

COL_SYSTEM = "SysID"
COL_TS = "SamplingTimestamp"
COL_SENSOR = "SamplingID"
COL_PARAM = "SampleID"
COL_VALUE = "SampleValue"

SERIES_MODE = "SamplingID_SampleID"   # or "SampleID"
TRAIN_FRAC = 0.70
MISSING_THRESHOLD = 0.40
ANOMALY_QUANTILE = 0.95
MIN_ROWS_AFTER_CLEAN = 20


# =========================================================
# HELPERS
# =========================================================
def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")


def extract_sysid_from_name(name: str) -> Optional[str]:
    """
    Example:
    ['795200--US918B1315']_Temperature.csv
    """
    m = re.search(r"\['(.+?)'\]_Temperature\.csv$", name)
    if m:
        return m.group(1)

    m = re.search(r"\[([^\]]+)\]", name)
    if m:
        return m.group(1)

    return None


@st.cache_data(show_spinner=False)
def load_csvs_from_pattern(pattern: str) -> List[Tuple[str, pd.DataFrame]]:
    paths = glob.glob(pattern)
    out = []

    for p in paths:
        name = Path(p).name
        sysid = extract_sysid_from_name(name)
        if sysid is None:
            continue

        df = pd.read_csv(p)
        df[COL_SYSTEM] = sysid
        out.append((sysid, df))

    return out


def prepare_wide_dataset(
    df: pd.DataFrame,
    system_col: str,
    ts_col: str,
    sensor_col: str,
    param_col: str,
    value_col: str,
    use_series_as: str = "SamplingID_SampleID",
) -> pd.DataFrame:
    data = df.copy()

    data[ts_col] = safe_to_datetime(data[ts_col])
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[ts_col, value_col])

    data["ts_2h"] = data[ts_col].dt.floor("2h")

    if use_series_as == "SamplingID_SampleID":
        data["SERIES"] = data[sensor_col].astype(str) + "_" + data[param_col].astype(str)
    else:
        data["SERIES"] = data[param_col].astype(str)

    agg = data.groupby([system_col, "ts_2h", "SERIES"], as_index=False)[value_col].median()

    wide = (
        agg.pivot_table(
            index=[system_col, "ts_2h"],
            columns="SERIES",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None
    return wide


def clean_wide_dataset(
    wide: pd.DataFrame,
    id_cols: List[str],
    missing_threshold: float = 0.4,
) -> pd.DataFrame:
    data = wide.copy()

    feat_cols = [c for c in data.columns if c not in id_cols]
    if not feat_cols:
        return data

    # keep columns with acceptable missingness
    miss = data[feat_cols].isna().mean()
    keep = miss[miss <= missing_threshold].index.tolist()

    data = data[id_cols + keep]
    feat_cols = keep

    if not feat_cols:
        return data

    system_col = id_cols[0]
    data = data.sort_values(id_cols)

    # fill within each device
    data[feat_cols] = data.groupby(system_col)[feat_cols].ffill().bfill()

    # remaining missing -> median
    for c in feat_cols:
        data[c] = data[c].fillna(data[c].median())

    # remove constant columns
    nunique = data[feat_cols].nunique(dropna=False)
    keep2 = nunique[nunique > 1].index.tolist()

    return data[id_cols + keep2]


def time_split_single_system(
    data: pd.DataFrame,
    system_id: str,
    system_col: str = "SysID",
    ts_col: str = "ts_2h",
    train_frac: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = data[data[system_col] == system_id].copy().sort_values(ts_col)

    if len(df) < 2:
        return df, df

    cut = int(len(df) * train_frac)
    cut = max(1, min(cut, len(df) - 1))

    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def run_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    id_cols: List[str],
    anomaly_quantile: float = 0.95,
):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

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
    results["actual"] = y_test.values
    results["predicted"] = pred_test
    results["abs_residual"] = test_abs_err
    results["anomaly_flag"] = (results["abs_residual"] > threshold).astype(int)

    metrics = {
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "residual_threshold": threshold,
        "n_test_rows": int(len(test_df)),
        "n_flagged": int(results["anomaly_flag"].sum()),
        "n_features": int(len(feature_cols)),
    }

    importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return model, results, metrics, importance


def column_summary(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        rows.append(
            {
                "column": c,
                "is_id": c in id_cols,
                "dtype": str(df[c].dtype),
                "missing_frac": float(df[c].isna().mean()),
                "nunique": int(df[c].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# LOAD
# =========================================================
styled_subheader("Per-system temperature model")
st.write("Choose a device, then manually select the target and explanatory variables.")

all_csvs = load_csvs_from_pattern(FOLDER_PATTERN)

if not all_csvs:
    st.error(f"No files found with pattern: {FOLDER_PATTERN}")
    st.stop()

raw_by_sysid = {}
load_errors = {}

for sysid, df in all_csvs:
    try:
        needed = [COL_SYSTEM, COL_TS, COL_SENSOR, COL_PARAM, COL_VALUE]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        raw_by_sysid[sysid] = df.copy()
    except Exception as e:
        load_errors[sysid] = str(e)

if not raw_by_sysid:
    st.error("No valid system files could be loaded.")
    if load_errors:
        st.dataframe(
            pd.DataFrame(
                {"SysID": list(load_errors.keys()), "Error": list(load_errors.values())}
            ),
            use_container_width=True,
        )
    st.stop()


# =========================================================
# DEVICE SELECTION
# =========================================================
sysids = sorted(raw_by_sysid.keys())
selected_device = st.selectbox("Choose device (SysID)", sysids)

raw_df = raw_by_sysid[selected_device]

try:
    wide_df = prepare_wide_dataset(
        df=raw_df,
        system_col=COL_SYSTEM,
        ts_col=COL_TS,
        sensor_col=COL_SENSOR,
        param_col=COL_PARAM,
        value_col=COL_VALUE,
        use_series_as=SERIES_MODE,
    )

    clean_df = clean_wide_dataset(
        wide_df,
        id_cols=[COL_SYSTEM, "ts_2h"],
        missing_threshold=MISSING_THRESHOLD,
    )

except Exception as e:
    st.error(f"Failed to prepare system {selected_device}: {e}")
    st.stop()

if len(clean_df) < MIN_ROWS_AFTER_CLEAN:
    st.warning(f"Only {len(clean_df)} rows after cleaning. Model may be unstable.")


# =========================================================
# COLUMN INSPECTION
# =========================================================
id_cols = [COL_SYSTEM, "ts_2h"]

numeric_cols = [
    c for c in clean_df.columns
    if c not in id_cols and pd.api.types.is_numeric_dtype(clean_df[c])
]

if not numeric_cols:
    st.error("No numeric columns available for modeling after cleaning.")
    st.stop()

with st.expander("Inspect columns", expanded=True):
    st.dataframe(column_summary(clean_df, id_cols=id_cols), use_container_width=True)

styled_subheader("Manual variable selection")

default_target_idx = 0
target_col = st.selectbox(
    "Select target variable",
    options=numeric_cols,
    index=default_target_idx,
)

default_features = [c for c in numeric_cols if c != target_col][:10]

feature_cols = st.multiselect(
    "Select explanatory variables",
    options=[c for c in numeric_cols if c != target_col],
    default=default_features,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    train_frac = st.slider("Train fraction", 0.5, 0.9, float(TRAIN_FRAC), 0.05)
with col_b:
    anomaly_quantile = st.slider("Anomaly quantile", 0.80, 0.99, float(ANOMALY_QUANTILE), 0.01)
with col_c:
    missing_threshold = st.slider("Missing threshold", 0.0, 0.9, float(MISSING_THRESHOLD), 0.05)

if missing_threshold != MISSING_THRESHOLD:
    clean_df = clean_wide_dataset(
        wide_df,
        id_cols=[COL_SYSTEM, "ts_2h"],
        missing_threshold=missing_threshold,
    )
    numeric_cols = [
        c for c in clean_df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(clean_df[c])
    ]


# =========================================================
# MODEL RUN
# =========================================================
run_clicked = st.button("Run model")

if run_clicked:
    if target_col not in clean_df.columns:
        st.error("Selected target is not in cleaned dataframe.")
        st.stop()

    feature_cols = [c for c in feature_cols if c in clean_df.columns and c != target_col]

    if len(feature_cols) < 1:
        st.error("Please select at least one explanatory variable.")
        st.stop()

    model_df = clean_df[[COL_SYSTEM, "ts_2h", target_col] + feature_cols].dropna().copy()

    if len(model_df) < 10:
        st.error(f"Too few rows for modeling after selection: {len(model_df)}")
        st.stop()

    train_df, test_df = time_split_single_system(
        data=model_df,
        system_id=selected_device,
        system_col=COL_SYSTEM,
        ts_col="ts_2h",
        train_frac=train_frac,
    )

    if len(train_df) < 5 or len(test_df) < 3:
        st.error(f"Train/test split too small. Train={len(train_df)}, Test={len(test_df)}")
        st.stop()

    try:
        model, results, metrics, importance = run_model(
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            feature_cols=feature_cols,
            id_cols=[COL_SYSTEM, "ts_2h"],
            anomaly_quantile=anomaly_quantile,
        )
    except Exception as e:
        st.error(f"Model failed: {e}")
        st.stop()

    styled_subheader(f"Results for {selected_device}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['mae_test']:.3f}", border=True)
    m2.metric("RMSE", f"{metrics['rmse_test']:.3f}", border=True)
    m3.metric("Threshold", f"{metrics['residual_threshold']:.3f}",border=True)
    m4.metric("Flagged", f"{metrics['n_flagged']} / {metrics['n_test_rows']}",border=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Feature importance", "Data", "Export"])

    with tab1:
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=results["ts_2h"],
                y=results["actual"],
                name="Actual",
                mode="lines",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=results["ts_2h"],
                y=results["predicted"],
                name="Predicted",
                mode="lines",
            )
        )

        anomalies = results[results["anomaly_flag"] == 1]
        fig_ts.add_trace(
            go.Scatter(
                x=anomalies["ts_2h"],
                y=anomalies["actual"],
                name="Anomaly",
                mode="markers",
                marker=dict(color="red", size=8, symbol="x"),
            )
        )

        fig_ts.update_layout(
            title=f"Actual vs Predicted: {target_col}",
            xaxis_title="Timestamp (2h)",
            yaxis_title=target_col,
            height=500,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        fig_res = px.histogram(results, x="abs_residual", nbins=40)
        fig_res.add_vline(
            x=metrics["residual_threshold"],
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
        )
        st.plotly_chart(fig_res, use_container_width=True)

        st.dataframe(results, use_container_width=True)

    with tab2:
        st.dataframe(importance, use_container_width=True)

        fig_imp = px.bar(
            importance.head(20),
            x="importance",
            y="feature",
            orientation="h",
            height=500,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.markdown("#### Raw data")
        st.dataframe(raw_df.head(30), use_container_width=True)

        st.markdown("#### Wide data")
        st.dataframe(wide_df.head(30), use_container_width=True)

        st.markdown("#### Clean data")
        st.dataframe(clean_df.head(30), use_container_width=True)

    with tab4:
        out = results.copy()
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"manual_model_results_{selected_device}.csv",
            mime="text/csv",
        )

        selection_info = pd.DataFrame(
            {
                "target": [target_col],
                "features": [", ".join(feature_cols)],
                "train_rows": [len(train_df)],
                "test_rows": [len(test_df)],
                "mae": [metrics["mae_test"]],
                "rmse": [metrics["rmse_test"]],
            }
        )
        st.download_button(
            "Download model selection summary",
            data=selection_info.to_csv(index=False).encode("utf-8"),
            file_name=f"manual_model_selection_{selected_device}.csv",
            mime="text/csv",
        )