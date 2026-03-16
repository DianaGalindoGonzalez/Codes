import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# =========================================================
# 1. PREPARE 2H WIDE DATASET
# =========================================================
def prepare_2h_wide_dataset(
    df: pd.DataFrame,
    system_col: str = "SysID",
    ts_col: str = "SamplingTimestamp",
    sensor_col: str = "SamplingID",
    param_col: str = "SampleID",
    value_col: str = "SampleValue",
    target_df: pd.DataFrame | None = None,
    target_ts_col: str = "SamplingTimestamp",
    target_system_col: str = "SysID",
    target_value_col: str = "eha_error",
    use_series_as: str = "SampleID",  # "SampleID" or "SamplingID_SampleID"
) -> pd.DataFrame:
    """
    Creates a wide dataset aggregated by 2-hour median.
    Optionally merges an EHA target table already aggregated or timestamped.
    """
    data = df.copy()
    data[ts_col] = pd.to_datetime(data[ts_col])

    # 2-hour bucket
    data["ts_2h"] = data[ts_col].dt.floor("2H")

    if use_series_as == "SamplingID_SampleID":
        data["SERIES"] = data[sensor_col].astype(str) + "|" + data[param_col].astype(str)
    else:
        data["SERIES"] = data[param_col].astype(str)

    # aggregate median in each 2h bucket
    agg = (
        data.groupby([system_col, "ts_2h", "SERIES"], as_index=False)[value_col]
        .median()
    )

    wide = (
        agg.pivot_table(
            index=[system_col, "ts_2h"],
            columns="SERIES",
            values=value_col,
            aggfunc="first"
        )
        .reset_index()
    )

    # optional target merge
    if target_df is not None:
        y = target_df.copy()
        y[target_ts_col] = pd.to_datetime(y[target_ts_col])
        y["ts_2h"] = y[target_ts_col].dt.floor("2H")

        y2 = (
            y.groupby([target_system_col, "ts_2h"], as_index=False)[target_value_col]
            .max()
        )

        wide = wide.merge(
            y2,
            left_on=[system_col, "ts_2h"],
            right_on=[target_system_col, "ts_2h"],
            how="left"
        )

        wide[target_value_col] = wide[target_value_col].fillna(0).astype(int)

        if target_system_col != system_col:
            wide = wide.drop(columns=[target_system_col], errors="ignore")

    return wide


# =========================================================
# 2. BASIC CLEANING
# =========================================================
def clean_wide_dataset(
    wide: pd.DataFrame,
    id_cols: list[str] = ["SysID", "ts_2h"],
    target_col: str | None = None,
    missing_threshold: float = 0.4
) -> pd.DataFrame:
    data = wide.copy()

    protected = list(id_cols)
    if target_col is not None and target_col in data.columns:
        protected.append(target_col)

    feature_cols = [c for c in data.columns if c not in protected]

    # Drop columns with too much missingness
    if feature_cols:
        miss = data[feature_cols].isna().mean()
        keep = miss[miss <= missing_threshold].index.tolist()
        data = data[id_cols + keep + ([target_col] if target_col and target_col in data.columns else [])]
        feature_cols = keep

    # Fill within each system
    if feature_cols:
        data = data.sort_values(id_cols)
        system_col = id_cols[0]
        data[feature_cols] = data.groupby(system_col)[feature_cols].ffill().bfill()

        # Fill remaining with median
        for c in feature_cols:
            data[c] = data[c].fillna(data[c].median())

        # Drop constant columns
        nunique = data[feature_cols].nunique()
        keep2 = nunique[nunique > 1].index.tolist()
        data = data[id_cols + keep2 + ([target_col] if target_col and target_col in data.columns else [])]

    return data


# =========================================================
# 3. TIME-ORDERED TRAIN/TEST SPLIT
# =========================================================
def time_split_single_system(
    data: pd.DataFrame,
    system_id,
    system_col: str = "SysID",
    ts_col: str = "ts_2h",
    train_frac: float = 0.7
):
    df = data[data[system_col] == system_id].copy().sort_values(ts_col)
    n = len(df)
    cut = int(n * train_frac)

    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


# =========================================================
# 4. MODEL 1: PREDICT ACB TEMP FROM OTHER TEMPS
# =========================================================
def run_temperature_consistency_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_temp_col: str,
    temp_feature_cols: list[str],
    id_cols: list[str] = ["SysID", "ts_2h"],
    anomaly_quantile: float = 0.95
):
    # remove target from feature list if present
    feature_cols = [c for c in temp_feature_cols if c != target_temp_col]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_temp_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_temp_col].copy()

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_abs_err = np.abs(y_train - pred_train)
    test_abs_err = np.abs(y_test - pred_test)

    threshold = np.quantile(train_abs_err, anomaly_quantile)

    results = test_df[id_cols].copy()
    results["actual_acb_temp"] = y_test.values
    results["pred_acb_temp"] = pred_test
    results["abs_residual"] = test_abs_err.values
    results["temp_anomaly_flag"] = (results["abs_residual"] > threshold).astype(int)

    metrics = {
        "mae_test": mean_absolute_error(y_test, pred_test),
        "rmse_test": mean_squared_error(y_test, pred_test) ** 0.5,
        "residual_threshold": threshold,
        "n_test_rows": len(test_df),
        "n_flagged": int(results["temp_anomaly_flag"].sum())
    }

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, results, metrics, importance


# =========================================================
# 5. MODEL 2: PREDICT EHA ERROR
# =========================================================
def run_eha_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "eha_error",
    id_cols: list[str] = ["SysID", "ts_2h"]
):
    feature_cols = [c for c in train_df.columns if c not in id_cols + [target_col]]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # handle imbalance
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    class_weight_ratio = neg / pos

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight={0: 1, 1: class_weight_ratio}
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results = test_df[id_cols + [target_col]].copy()
    results["pred_eha_error"] = pred
    results["prob_eha_error"] = prob

    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    metrics = {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob) if y_test.nunique() > 1 else np.nan,
        "pr_auc": average_precision_score(y_test, prob) if y_test.nunique() > 1 else np.nan,
        "confusion_matrix": cm
    }

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, results, metrics, importance
    
    
temp_cols = [c for c in wide.columns if "TEMP" in c.upper()]
print(temp_cols)

target_temp = "ACB_TEMP"   # replace with your exact column name
temp_model, temp_results, temp_metrics, temp_importance = run_temperature_consistency_model(
    train_df=train_df,
    test_df=test_df,
    target_temp_col=target_temp,
    temp_feature_cols=temp_cols,
    id_cols=["SysID", "ts_2h"],
    anomaly_quantile=0.95
)

print(temp_metrics)
print(temp_importance.head(10))
print(temp_results.head())