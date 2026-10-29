"""
Uber NCR Ride Bookings - Multi-Feature Predictive Analytics Dashboard
Run with:  streamlit run main.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

CSV_PATH = Path(__file__).parent / "ncr_ride_bookings.csv"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Uber NCR Analytics", page_icon="🚖", layout="wide")


# =====================================================================
# 1. DATA PIPELINE  (extraction + cleaning + feature engineering)
# =====================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    # Clean ID columns (they contain literal quote characters)
    for col in ["Booking ID", "Customer ID"]:
        df[col] = df[col].astype(str).str.strip('"')

    # Date / time features
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["Date"] = df["Datetime"].dt.normalize()
    df["Hour"] = df["Datetime"].dt.hour
    df["DOW"] = df["Datetime"].dt.dayofweek
    df["DayName"] = df["Datetime"].dt.day_name()
    df["Month"] = df["Datetime"].dt.month
    df["MonthName"] = df["Datetime"].dt.strftime("%b")

    # Target flag for classification
    df["Is Completed"] = (df["Booking Status"] == "Completed").astype(int)
    return df


df_all = load_data()

# =====================================================================
# SIDEBAR FILTERS (used by the Overview / Insights / Correlation tabs)
# =====================================================================
st.sidebar.title("🔎 Filters")
min_d, max_d = df_all["Date"].min().date(), df_all["Date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
vehicles = st.sidebar.multiselect(
    "Vehicle type", sorted(df_all["Vehicle Type"].unique()), default=sorted(df_all["Vehicle Type"].unique())
)
statuses = st.sidebar.multiselect(
    "Booking status", sorted(df_all["Booking Status"].unique()), default=sorted(df_all["Booking Status"].unique())
)
st.sidebar.caption("Filters apply to Overview, Insights and Correlation. The ML tabs always train on the full dataset.")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = pd.Timestamp(min_d), pd.Timestamp(max_d)

df = df_all[
    (df_all["Date"] >= start)
    & (df_all["Date"] <= end)
    & (df_all["Vehicle Type"].isin(vehicles))
    & (df_all["Booking Status"].isin(statuses))
]

st.title("🚖 Uber NCR Ride Bookings — Predictive Analytics Dashboard")
if df.empty:
    st.warning("No data for the selected filters. Please widen your selection.")
    st.stop()

tab_overview, tab_insights, tab_corr, tab_reg, tab_ts, tab_clf = st.tabs(
    ["📊 Overview", "🧭 Insights", "🔗 Correlation", "📈 Regression", "⏳ Time-Series", "🎯 Classification"]
)

# =====================================================================
# 2. OVERVIEW
# =====================================================================
with tab_overview:
    completed = df[df["Booking Status"] == "Completed"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total bookings", f"{len(df):,}")
    c2.metric("Success rate", f"{(df['Booking Status'] == 'Completed').mean():.1%}")
    c3.metric("Revenue (completed)", f"₹{completed['Booking Value'].sum():,.0f}")
    c4.metric("Avg ride distance", f"{completed['Ride Distance'].mean():.1f} km" if len(completed) else "–")
    c5.metric("Avg customer rating", f"{completed['Customer Rating'].mean():.2f}" if len(completed) else "–")

    left, right = st.columns(2)
    with left:
        status_counts = df["Booking Status"].value_counts().reset_index()
        status_counts.columns = ["Booking Status", "Count"]
        st.plotly_chart(
            px.pie(status_counts, names="Booking Status", values="Count", hole=0.45, title="Booking status split"),
            width="stretch",
        )
    with right:
        veh = df["Vehicle Type"].value_counts().reset_index()
        veh.columns = ["Vehicle Type", "Bookings"]
        st.plotly_chart(
            px.bar(veh, x="Vehicle Type", y="Bookings", color="Vehicle Type", title="Bookings by vehicle type"),
            width="stretch",
        )

    monthly = df.groupby(df["Date"].dt.to_period("M").astype(str)).size().reset_index(name="Bookings")
    monthly.columns = ["Month", "Bookings"]
    st.plotly_chart(px.line(monthly, x="Month", y="Bookings", markers=True, title="Monthly bookings"), width="stretch")

    pay = completed["Payment Method"].value_counts().reset_index()
    pay.columns = ["Payment Method", "Rides"]
    st.plotly_chart(
        px.bar(pay, x="Payment Method", y="Rides", color="Payment Method", title="Payment methods (completed rides)"),
        width="stretch",
    )

# =====================================================================
# 3. INSIGHTS  (complex extraction)
# =====================================================================
with tab_insights:
    st.subheader("Demand patterns")
    heat = df.groupby(["DayName", "Hour"]).size().reset_index(name="Bookings")
    heat["DayName"] = pd.Categorical(heat["DayName"], categories=DAY_ORDER, ordered=True)
    heat = heat.pivot(index="DayName", columns="Hour", values="Bookings")
    st.plotly_chart(
        px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd", title="Bookings: day of week × hour of day"),
        width="stretch",
    )

    a, b = st.columns(2)
    with a:
        top_pick = df["Pickup Location"].value_counts().head(10).reset_index()
        top_pick.columns = ["Pickup Location", "Bookings"]
        st.plotly_chart(
            px.bar(top_pick, x="Bookings", y="Pickup Location", orientation="h", title="Top 10 pickup locations")
            .update_yaxes(autorange="reversed"),
            width="stretch",
        )
    with b:
        top_drop = df["Drop Location"].value_counts().head(10).reset_index()
        top_drop.columns = ["Drop Location", "Bookings"]
        st.plotly_chart(
            px.bar(top_drop, x="Bookings", y="Drop Location", orientation="h", title="Top 10 drop locations")
            .update_yaxes(autorange="reversed"),
            width="stretch",
        )

    st.subheader("Why do rides fail?")
    x, y, z = st.columns(3)
    for col_widget, col, title in [
        (x, "Reason for cancelling by Customer", "Customer cancellation reasons"),
        (y, "Driver Cancellation Reason", "Driver cancellation reasons"),
        (z, "Incomplete Rides Reason", "Incomplete ride reasons"),
    ]:
        with col_widget:
            r = df[col].dropna().value_counts().reset_index()
            r.columns = ["Reason", "Count"]
            if r.empty:
                st.info(f"No data: {title}")
            else:
                st.plotly_chart(px.bar(r, x="Count", y="Reason", orientation="h", title=title), width="stretch")

    st.subheader("Revenue & service quality by vehicle type")
    comp = df[df["Booking Status"] == "Completed"]
    summary = (
        comp.groupby("Vehicle Type")
        .agg(
            Rides=("Booking ID", "count"),
            Revenue=("Booking Value", "sum"),
            Avg_Fare=("Booking Value", "mean"),
            Avg_Distance_km=("Ride Distance", "mean"),
            Avg_Driver_Rating=("Driver Ratings", "mean"),
            Avg_Customer_Rating=("Customer Rating", "mean"),
            Avg_VTAT=("Avg VTAT", "mean"),
            Avg_CTAT=("Avg CTAT", "mean"),
        )
        .round(2)
        .sort_values("Revenue", ascending=False)
    )
    st.dataframe(summary, width="stretch")

# =====================================================================
# 4. CORRELATION MAPPING
# =====================================================================
with tab_corr:
    st.subheader("Correlation heatmap (completed rides)")
    num_cols = ["Booking Value", "Ride Distance", "Avg VTAT", "Avg CTAT", "Driver Ratings", "Customer Rating", "Hour", "DOW", "Month"]
    comp = df[df["Booking Status"] == "Completed"][num_cols].dropna()
    if len(comp) < 10:
        st.info("Not enough completed rides in the current filter.")
    else:
        corr = comp.corr()
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"),
            width="stretch",
        )
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs["Abs"] = pairs["Correlation"].abs()
        st.markdown("**Strongest feature pairs**")
        st.dataframe(pairs.sort_values("Abs", ascending=False).drop(columns="Abs").head(8).round(4), width="stretch")
        st.caption(
            "Correlations close to 0 mean there is no linear relationship. In this dataset almost all pairs are near 0, "
            "which directly explains the modest model accuracy in the Regression and Classification tabs."
        )

    st.subheader("Explore a pair")
    p1, p2 = st.columns(2)
    xa = p1.selectbox("X axis", num_cols, index=1)
    ya = p2.selectbox("Y axis", num_cols, index=0)
    pts = df[df["Booking Status"] == "Completed"].dropna(subset=[xa, ya])
    if not pts.empty:
        sample = pts.sample(n=min(3000, len(pts)), random_state=1)
        st.plotly_chart(px.scatter(sample, x=xa, y=ya, color="Vehicle Type", opacity=0.5), width="stretch")


# =====================================================================
# 5. REGRESSION  (predict fare / Booking Value)
# =====================================================================
REG_FEATURES = ["Ride Distance", "Avg VTAT", "Avg CTAT", "Hour", "DOW", "Month"]


@st.cache_resource
def train_regression(model_name: str):
    d = df_all[df_all["Booking Status"] == "Completed"].dropna(subset=REG_FEATURES + ["Booking Value"])
    X = pd.get_dummies(d[REG_FEATURES + ["Vehicle Type"]], columns=["Vehicle Type"])
    y = d["Booking Value"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "R²": r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "Baseline MAE": mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean())),
    }
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns)
    else:
        imp = pd.Series(np.abs(model.coef_) * X_tr.std().values, index=X.columns)
    return model, X.columns.tolist(), metrics, y_te.reset_index(drop=True), pd.Series(pred), imp.sort_values(ascending=False)


with tab_reg:
    st.subheader("Predict ride fare (Booking Value)")
    model_name = st.radio("Model", ["Random Forest", "Linear Regression"], horizontal=True)
    with st.spinner("Training model..."):
        reg_model, reg_cols, m, y_true, y_pred, imp = train_regression(model_name)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² (test)", f"{m['R²']:.3f}")
    k2.metric("MAE (₹)", f"{m['MAE']:.1f}")
    k3.metric("RMSE (₹)", f"{m['RMSE']:.1f}")
    k4.metric("Baseline MAE (always predict mean)", f"{m['Baseline MAE']:.1f}")
    if m["R²"] < 0.1:
        st.warning(
            "R² is very low: fare has almost no relationship with distance, vehicle type or time in this dataset "
            "(see the Correlation tab). The model is barely better than predicting the average fare. "
            "This is a property of the data, not a bug — include it as a finding in your report."
        )

    c_left, c_right = st.columns(2)
    with c_left:
        idx = np.random.RandomState(0).choice(len(y_true), size=min(2000, len(y_true)), replace=False)
        fig = px.scatter(x=y_true.iloc[idx], y=y_pred.iloc[idx], opacity=0.4,
                         labels={"x": "Actual fare", "y": "Predicted fare"}, title="Actual vs predicted")
        mx = float(max(y_true.max(), y_pred.max()))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect fit", line=dict(dash="dash")))
        st.plotly_chart(fig, width="stretch")
    with c_right:
        st.plotly_chart(
            px.bar(imp.head(10).rename("Importance").rename_axis("Feature").reset_index(),
                   x="Importance", y="Feature", orientation="h", title="Top feature importance")
            .update_yaxes(autorange="reversed"),
            width="stretch",
        )

    st.markdown("### 🧮 Try it: predict a fare")
    i1, i2, i3, i4 = st.columns(4)
    in_dist = i1.slider("Distance (km)", 1.0, 50.0, 20.0)
    in_veh = i2.selectbox("Vehicle", sorted(df_all["Vehicle Type"].unique()), key="reg_veh")
    in_hour = i3.slider("Hour", 0, 23, 18, key="reg_hour")
    in_dow = i4.selectbox("Day", DAY_ORDER, key="reg_dow")
    j1, j2, j3 = st.columns(3)
    in_month = j1.slider("Month", 1, 12, 6, key="reg_month")
    in_vtat = j2.slider("Avg VTAT (min)", 1.0, 20.0, 8.0)
    in_ctat = j3.slider("Avg CTAT (min)", 5.0, 45.0, 25.0)

    row = pd.DataFrame([{
        "Ride Distance": in_dist, "Avg VTAT": in_vtat, "Avg CTAT": in_ctat,
        "Hour": in_hour, "DOW": DAY_ORDER.index(in_dow), "Month": in_month,
    }])
    for v in df_all["Vehicle Type"].unique():
        row[f"Vehicle Type_{v}"] = int(v == in_veh)
    row = row.reindex(columns=reg_cols, fill_value=0)
    st.success(f"Predicted fare: **₹{reg_model.predict(row)[0]:,.0f}**")


# =====================================================================
# 6. TIME-SERIES FORECASTING  (Holt-Winters)
# =====================================================================
with tab_ts:
    st.subheader("Demand / revenue forecasting")
    t1, t2, t3 = st.columns(3)
    metric_choice = t1.selectbox("Forecast target", ["Daily bookings", "Daily revenue (completed rides)"])
    ts_vehicle = t2.selectbox("Vehicle type", ["All"] + sorted(df_all["Vehicle Type"].unique()))
    horizon = t3.slider("Forecast horizon (days)", 7, 90, 30)

    d = df_all if ts_vehicle == "All" else df_all[df_all["Vehicle Type"] == ts_vehicle]
    if metric_choice == "Daily bookings":
        series = d.groupby("Date").size()
    else:
        series = d[d["Booking Status"] == "Completed"].groupby("Date")["Booking Value"].sum()
    series = series.asfreq("D").fillna(0).astype(float)

    # Hold-out evaluation on the last 30 days
    train, test = series.iloc[:-30], series.iloc[-30:]
    hw_eval = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit()
    test_pred = pd.Series(hw_eval.forecast(30).values, index=test.index)
    mae = mean_absolute_error(test, test_pred)
    mape = float(np.mean(np.abs((test - test_pred) / test.replace(0, np.nan)).dropna()) * 100)
    base_mae = mean_absolute_error(test, np.full(len(test), train.mean()))

    k1, k2, k3 = st.columns(3)
    k1.metric("Hold-out MAE (last 30 days)", f"{mae:,.1f}")
    k2.metric("Hold-out MAPE", f"{mape:.1f}%")
    k3.metric("Baseline MAE (historical mean)", f"{base_mae:,.1f}")

    # Final model on all data -> future forecast
    hw = ExponentialSmoothing(series, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit()
    fc = hw.forecast(horizon)
    resid_std = float(np.std(series - hw.fittedvalues))
    fut_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc = pd.Series(fc.values, index=fut_idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="History", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=series.index, y=series.rolling(7).mean(), name="7-day average", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Forecast", line=dict(dash="dash", width=3)))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values + 1.96 * resid_std, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values - 1.96 * resid_std, fill="tonexty",
                             line=dict(width=0), name="95% band", opacity=0.2))
    fig.update_layout(title=f"{metric_choice} — {horizon}-day forecast ({ts_vehicle})", xaxis_title="Date")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Model: Holt-Winters exponential smoothing (damped trend + weekly seasonality). "
        "If its hold-out MAE is close to the baseline, daily demand is essentially flat with random noise."
    )

    dow_profile = series.groupby(series.index.day_name()).mean().reindex(DAY_ORDER).reset_index()
    dow_profile.columns = ["Day", "Average"]
    st.plotly_chart(px.bar(dow_profile, x="Day", y="Average", title="Average by day of week"), width="stretch")


# =====================================================================
# 7. CLASSIFICATION  (will a booking be completed?)
# =====================================================================
# Only features known AT BOOKING TIME are used, to avoid data leakage
# (e.g. Booking Value / Ride Distance are empty for cancelled rides).
CLF_NUM = ["Hour", "DOW", "Month"]
CLF_CAT = ["Vehicle Type", "Pickup Location"]


@st.cache_resource
def train_classifier(model_name: str):
    X = pd.get_dummies(df_all[CLF_NUM + CLF_CAT], columns=CLF_CAT)
    y = df_all["Is Completed"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "Baseline accuracy": max(y_te.mean(), 1 - y_te.mean()),
    }
    return model, X.columns.tolist(), metrics, confusion_matrix(y_te, pred), fpr, tpr


with tab_clf:
    st.subheader("Predict whether a booking will be completed")
    clf_name = st.radio("Model", ["Random Forest", "Logistic Regression"], horizontal=True)
    with st.spinner("Training model..."):
        clf, clf_cols, cm_metrics, cm, fpr, tpr = train_classifier(clf_name)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{cm_metrics['Accuracy']:.1%}")
    k2.metric("F1 score", f"{cm_metrics['F1']:.3f}")
    k3.metric("ROC-AUC", f"{cm_metrics['ROC-AUC']:.3f}")
    k4.metric("Baseline (always predict majority)", f"{cm_metrics['Baseline accuracy']:.1%}")
    if cm_metrics["ROC-AUC"] < 0.55:
        st.warning(
            "ROC-AUC ≈ 0.5 means the model cannot separate completed from failed bookings better than chance. "
            "Time, vehicle type and pickup area do not influence the outcome in this dataset."
        )

    a, b = st.columns(2)
    with a:
        st.plotly_chart(
            px.imshow(cm, text_auto=True, x=["Not completed", "Completed"], y=["Not completed", "Completed"],
                      labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues", title="Confusion matrix"),
            width="stretch",
        )
    with b:
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=fpr, y=tpr, name="Model"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(dash="dash")))
        roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(roc, width="stretch")

    st.markdown("### 🎯 Try it: booking success probability")
    q1, q2, q3, q4, q5 = st.columns(5)
    c_veh = q1.selectbox("Vehicle", sorted(df_all["Vehicle Type"].unique()), key="clf_veh")
    c_pick = q2.selectbox("Pickup location", sorted(df_all["Pickup Location"].unique()), key="clf_pick")
    c_hour = q3.slider("Hour", 0, 23, 9, key="clf_hour")
    c_dow = q4.selectbox("Day", DAY_ORDER, key="clf_dow")
    c_month = q5.slider("Month", 1, 12, 6, key="clf_month")

    row = pd.DataFrame([{"Hour": c_hour, "DOW": DAY_ORDER.index(c_dow), "Month": c_month}])
    row = row.reindex(columns=clf_cols, fill_value=0)
    row[f"Vehicle Type_{c_veh}"] = 1
    row[f"Pickup Location_{c_pick}"] = 1
    p = clf.predict_proba(row[clf_cols])[0, 1]
    st.success(f"Probability this booking is completed: **{p:.1%}**")
