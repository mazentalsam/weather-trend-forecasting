"""Reusable Plotly and Matplotlib visualization functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_white"
COLORS = {
    "temperature": "#e74c3c",
    "precipitation": "#3498db",
    "humidity": "#2ecc71",
    "wind": "#9b59b6",
    "actual": "#000000",
    "train": "#888888",
}
MODEL_COLORS = {
    "ARIMA": "#e74c3c",
    "SARIMA": "#c0392b",
    "Holt-Winters": "#3498db",
    "Prophet": "#f39c12",
    "Ridge": "#2ecc71",
    "Random Forest": "#9b59b6",
    "XGBoost": "#e67e22",
    "Gradient Boosting": "#1abc9c",
    "XGBoost (Tuned)": "#d35400",
    "RF (Tuned)": "#8e44ad",
    "Ensemble (Simple)": "#2c3e50",
    "Ensemble (Weighted)": "#e74c3c",
    "Neural Network (MLP)": "#16a085",
}


def plot_temperature_trend(daily_temp: pd.DataFrame, temp_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_temp["date"], y=daily_temp[temp_col],
        mode="lines", name="Daily Avg",
        line=dict(color=COLORS["temperature"], width=1), opacity=0.5,
    ))
    if len(daily_temp) >= 7:
        rolling7 = daily_temp[temp_col].rolling(7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=daily_temp["date"], y=rolling7,
            mode="lines", name="7-Day Rolling",
            line=dict(color="darkred", width=2.5),
        ))
    if len(daily_temp) >= 30:
        rolling30 = daily_temp[temp_col].rolling(30, center=True).mean()
        fig.add_trace(go.Scatter(
            x=daily_temp["date"], y=rolling30,
            mode="lines", name="30-Day Rolling",
            line=dict(color="black", width=2, dash="dash"),
        ))
    fig.update_layout(
        title="Global Average Temperature Over Time",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        template=TEMPLATE, hovermode="x unified", height=500,
    )
    return fig


def plot_forecast_comparison(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict[str, pd.Series],
    prediction_interval: tuple[pd.Series, pd.Series] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train, mode="lines", name="Training",
        line=dict(color=COLORS["train"], width=1), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=test.index, y=test, mode="lines", name="Actual (Test)",
        line=dict(color=COLORS["actual"], width=2.5),
    ))
    for name, pred in forecasts.items():
        fig.add_trace(go.Scatter(
            x=pred.index, y=pred, mode="lines", name=name,
            line=dict(color=MODEL_COLORS.get(name, "gray"), width=1.5, dash="dash"),
        ))

    if prediction_interval:
        lower, upper = prediction_interval
        fig.add_trace(go.Scatter(
            x=list(upper.index) + list(lower.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor="rgba(52,152,219,0.1)",
            line=dict(color="rgba(255,255,255,0)"), name="95% PI",
        ))

    fig.add_vline(x=test.index[0], line_dash="dot", line_color="red", opacity=0.5)
    fig.update_layout(
        title="All Model Forecasts vs Actual",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        template=TEMPLATE, hovermode="x unified", height=550,
    )
    return fig


def plot_model_ranking(results_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        results_df, x="RMSE", y="Model", orientation="h",
        color="RMSE", color_continuous_scale="RdYlGn_r",
        title="Model Ranking by RMSE (lower = better)",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total descending"},
        template=TEMPLATE, height=500,
    )
    return fig


def plot_global_map(
    loc_df: pd.DataFrame,
    lat_col: str, lon_col: str,
    color_col: str, hover_col: str,
) -> go.Figure:
    fig = px.scatter_geo(
        loc_df, lat=lat_col, lon=lon_col, color=color_col,
        hover_name=hover_col, color_continuous_scale="RdYlBu_r",
        projection="natural earth",
        title="Global Temperature Distribution",
        labels={color_col: "Avg Temp (°C)"},
    )
    fig.update_layout(height=600, template=TEMPLATE)
    fig.update_geos(showcountries=True, countrycolor="lightgray",
                    showcoastlines=True, coastlinecolor="gray")
    return fig


def plot_climate_zones(zone_monthly: pd.DataFrame, temp_col: str) -> go.Figure:
    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    zone_monthly = zone_monthly.copy()
    zone_monthly["month_name"] = zone_monthly["month"].map(month_map)
    fig = px.line(
        zone_monthly, x="month_name", y=temp_col, color="climate_zone",
        category_orders={
            "month_name": list(month_map.values()),
            "climate_zone": ["Tropical", "Subtropical", "Temperate", "Subarctic", "Polar"],
        },
        markers=True,
        title="Temperature by Climate Zone & Month",
        labels={temp_col: "Avg Temperature (°C)", "month_name": "Month"},
    )
    fig.update_layout(template=TEMPLATE, height=500)
    return fig


def plot_residual_diagnostics(residuals: pd.Series) -> go.Figure:
    from scipy import stats as sp_stats

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Residual Distribution", "Q-Q Plot",
            "Residuals Over Time", "Residual ACF",
        ],
    )

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, marker_color="steelblue",
        opacity=0.7, name="Residuals",
    ), row=1, col=1)

    sorted_r = np.sort(residuals)
    theoretical = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_r)))
    fig.add_trace(go.Scatter(
        x=theoretical, y=sorted_r, mode="markers",
        marker=dict(size=3, color="steelblue"), name="Q-Q",
    ), row=1, col=2)
    qq_line_x = np.array([theoretical.min(), theoretical.max()])
    fig.add_trace(go.Scatter(
        x=qq_line_x,
        y=residuals.mean() + residuals.std() * qq_line_x,
        mode="lines", line=dict(color="red", dash="dash"), name="Normal Ref",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=residuals.index, y=residuals, mode="lines",
        line=dict(color="steelblue", width=1), name="Temporal",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    from statsmodels.tsa.stattools import acf
    max_lags = min(30, len(residuals) // 2 - 1)
    if max_lags > 1:
        acf_vals = acf(residuals.dropna(), nlags=max_lags)
        fig.add_trace(go.Bar(
            x=list(range(len(acf_vals))), y=acf_vals,
            marker_color="steelblue", name="ACF",
        ), row=2, col=2)
        ci = 1.96 / np.sqrt(len(residuals))
        fig.add_hline(y=ci, line_dash="dot", line_color="red", row=2, col=2)
        fig.add_hline(y=-ci, line_dash="dot", line_color="red", row=2, col=2)

    fig.update_layout(
        title="Weighted Ensemble — Residual Diagnostics",
        template=TEMPLATE, height=700, showlegend=False,
    )
    return fig


def plot_feature_importance(
    rf_imp: pd.Series, perm_imp: pd.Series, corr_imp: pd.Series, top_n: int = 15
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Random Forest", "Permutation", "Correlation-Based"],
    )
    for col_idx, (imp, color) in enumerate([
        (rf_imp.head(top_n).sort_values(), "#27ae60"),
        (perm_imp.head(top_n).sort_values(), "#e67e22"),
        (corr_imp.head(top_n).sort_values(), "#8e44ad"),
    ], 1):
        fig.add_trace(go.Bar(
            x=imp.values, y=imp.index, orientation="h",
            marker_color=color,
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Feature Importance Analysis — Predicting Temperature",
        template=TEMPLATE, height=500, showlegend=False,
    )
    return fig


def plot_shap_bar(mean_abs_shap: pd.Series, top_n: int = 15) -> go.Figure:
    top = mean_abs_shap.head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker=dict(
            color=top.values,
            colorscale="Reds",
        ),
    ))
    fig.update_layout(
        title=f"SHAP Feature Importance (Top {top_n})",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        template=TEMPLATE, height=500,
    )
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray, X: pd.DataFrame, top_n: int = 15
) -> go.Figure:
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:]

    fig = go.Figure()
    for rank, idx in enumerate(top_idx):
        vals = shap_values[:, idx]
        feat_vals = X.iloc[:, idx].values.astype(float)
        lo, hi = np.nanmin(feat_vals), np.nanmax(feat_vals)
        norm = (feat_vals - lo) / (hi - lo + 1e-10)
        jitter = np.random.default_rng(42).normal(0, 0.15, len(vals))

        fig.add_trace(go.Scatter(
            x=vals,
            y=rank + jitter,
            mode="markers",
            marker=dict(size=3, color=norm, colorscale="RdBu_r",
                        cmin=0, cmax=1, opacity=0.6),
            hovertemplate=(
                f"{X.columns[idx]}<br>"
                "SHAP: %{x:.3f}<br>Feature value: %{text}<extra></extra>"
            ),
            text=[f"{v:.2f}" for v in feat_vals],
            showlegend=False,
        ))

    fig.update_layout(
        title=f"SHAP Beeswarm Plot (Top {top_n} Features)",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis=dict(
            tickvals=list(range(len(top_idx))),
            ticktext=[X.columns[i] for i in top_idx],
        ),
        template=TEMPLATE, height=600,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def plot_importance_comparison(
    rf_imp: pd.Series, perm_imp: pd.Series,
    corr_imp: pd.Series, shap_imp: pd.Series | None = None,
    top_n: int = 10,
) -> go.Figure:
    methods = {"RF (MDI)": rf_imp, "Permutation": perm_imp, "Correlation": corr_imp}
    if shap_imp is not None:
        methods["SHAP"] = shap_imp

    all_features = set()
    for imp in methods.values():
        all_features.update(imp.head(top_n).index)

    rows = []
    for feat in all_features:
        row = {"Feature": feat}
        for name, imp in methods.items():
            ranked = imp.rank(ascending=False)
            row[name] = int(ranked.get(feat, len(imp) + 1))
        rows.append(row)

    rank_df = pd.DataFrame(rows).sort_values("RF (MDI)")
    method_cols = [c for c in rank_df.columns if c != "Feature"]

    fig = go.Figure()
    for method in method_cols:
        fig.add_trace(go.Scatter(
            x=rank_df["Feature"], y=rank_df[method],
            mode="lines+markers", name=method,
        ))

    fig.update_layout(
        title="Feature Ranking Comparison Across Methods",
        xaxis_title="Feature", yaxis_title="Rank (lower = more important)",
        yaxis=dict(autorange="reversed"),
        template=TEMPLATE, height=500,
        xaxis_tickangle=-45,
    )
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto", title=title,
    )
    fig.update_layout(height=700, width=900, template=TEMPLATE)
    return fig


def plot_anomaly_scatter(
    df: pd.DataFrame,
    x_col: str, y_col: str,
    anomaly_mask: pd.Series,
    title: str = "Anomaly Detection",
    max_normal: int = 10_000,
) -> go.Figure:
    normal = df[~anomaly_mask]
    anomalies = df[anomaly_mask]

    if len(normal) > max_normal:
        normal = normal.sample(n=max_normal, random_state=42)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=normal[x_col], y=normal[y_col], mode="markers",
        marker=dict(size=3, color="steelblue", opacity=0.15), name="Normal",
    ))
    fig.add_trace(go.Scattergl(
        x=anomalies[x_col], y=anomalies[y_col], mode="markers",
        marker=dict(size=6, color="red", opacity=0.6), name="Anomaly",
    ))
    fig.update_layout(title=title, template=TEMPLATE, height=500)
    return fig
