"""
utils/plots.py
==============
Reusable Plotly / Matplotlib charting helpers for ML Explorer.

All functions return a plotly Figure object (or a matplotlib Figure where noted)
so the caller can do `st.plotly_chart(fig, use_container_width=True)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    silhouette_score,
)
from sklearn.model_selection import learning_curve

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Plotly
ACCENT = "#636EFA"
GOOD_COLOR = "#00CC96"
WARN_COLOR = "#EF553B"


# =============================================================================
# CLASSIFICATION
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Matriz de Confusión",
) -> go.Figure:
    """Interactive confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = class_names if class_names else [str(i) for i in range(cm.shape[0])]

    # Normalised version for text annotation
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text = [
        [f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ]

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Real",
        yaxis={"autorange": "reversed"},
        height=420,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Curva ROC",
) -> go.Figure:
    """ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {roc_auc:.3f})",
            line=dict(color=ACCENT, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(99,110,250,0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Clasificador aleatorio",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos",
        legend=dict(x=0.6, y=0.1),
        height=420,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    title: str = "Importancia de Variables",
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        title=title,
        labels={"importance": "Importancia", "feature": "Variable"},
    )
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(350, top_n * 22),
        margin=dict(l=160, r=20, t=60, b=40),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    title: str = "Frontera de Decisión (2D)",
    resolution: int = 150,
) -> go.Figure:
    """Mesh-based decision boundary for 2-feature datasets."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fn = feature_names or ["Feature 0", "Feature 1"]
    cn = class_names or [str(c) for c in np.unique(y)]

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=Z,
            showscale=False,
            colorscale="Pastel",
            opacity=0.6,
            hoverinfo="skip",
            contours=dict(coloring="fill"),
        )
    )
    for cls_idx, cls in enumerate(np.unique(y)):
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=cn[cls_idx] if cls_idx < len(cn) else str(cls),
                marker=dict(size=6, line=dict(width=0.5, color="white")),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=fn[0],
        yaxis_title=fn[1],
        height=450,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# =============================================================================
# REGRESSION
# =============================================================================

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicho vs. Real",
) -> go.Figure:
    """Scatter of predicted vs actual values with identity line."""
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color=ACCENT, size=5, opacity=0.6),
            name="Muestras",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            line=dict(color=WARN_COLOR, dash="dash", width=1.5),
            name="Predicción perfecta",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Valor Real",
        yaxis_title="Valor Predicho",
        height=420,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Gráfico de Residuos",
) -> go.Figure:
    """Residuals vs fitted values + histogram side by side."""
    residuals = y_true - y_pred

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Residuos vs. Predichos", "Distribución de Residuos"),
    )
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=ACCENT, size=5, opacity=0.6),
            name="Residuo",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=WARN_COLOR, row=1, col=1)
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color=ACCENT,
            opacity=0.75,
            name="Distribución",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Predicho", row=1, col=1)
    fig.update_yaxes(title_text="Residuo", row=1, col=1)
    fig.update_xaxes(title_text="Residuo", row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
    fig.update_layout(title=title, height=420, showlegend=False,
                      margin=dict(l=60, r=20, t=80, b=60))
    return fig


def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    title: str = "Curva de Aprendizaje",
) -> go.Figure:
    """Train / validation score vs training-set size."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="r2",
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = go.Figure()
    # Training band
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean, mode="lines+markers",
        name="Entrenamiento", line=dict(color=ACCENT),
    ))
    # Validation band
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill="toself", fillcolor="rgba(0,204,150,0.15)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean, mode="lines+markers",
        name="Validación", line=dict(color=GOOD_COLOR),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Tamaño del conjunto de entrenamiento",
        yaxis_title="R²",
        height=420,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_coefficients(
    coef: np.ndarray,
    feature_names: list[str],
    title: str = "Coeficientes del Modelo",
) -> go.Figure:
    """Signed coefficient bar chart for linear models."""
    df = pd.DataFrame({"feature": feature_names, "coef": coef}).sort_values("coef")
    colors = [GOOD_COLOR if v >= 0 else WARN_COLOR for v in df["coef"]]
    fig = go.Figure(
        go.Bar(
            x=df["coef"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Coeficiente",
        yaxis_title="Variable",
        height=max(350, len(feature_names) * 22),
        margin=dict(l=160, r=20, t=60, b=40),
    )
    return fig


# =============================================================================
# CLUSTERING
# =============================================================================

def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Clusters (2D)",
    x_label: str = "Componente 1",
    y_label: str = "Componente 2",
) -> go.Figure:
    """Scatter coloured by cluster label. Label -1 → noise (DBSCAN)."""
    df = pd.DataFrame({"x": X_2d[:, 0], "y": X_2d[:, 1],
                        "cluster": labels.astype(str)})
    # Replace "-1" with "Ruido" for DBSCAN
    df["cluster"] = df["cluster"].replace("-1", "Ruido")

    fig = px.scatter(
        df, x="x", y="y", color="cluster",
        color_discrete_sequence=PALETTE,
        title=title,
        labels={"x": x_label, "y": y_label, "cluster": "Cluster"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8,
                                  line=dict(width=0.5, color="white")))
    fig.update_layout(
        height=480,
        margin=dict(l=60, r=20, t=60, b=60),
        legend_title_text="Cluster",
    )
    return fig


def plot_elbow_method(
    inertias: list[float],
    k_range: range | list[int],
    title: str = "Método del Codo (K-Means)",
) -> go.Figure:
    """Inertia vs k to help choose optimal number of clusters."""
    k_vals = list(k_range)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_vals, y=inertias,
        mode="lines+markers",
        marker=dict(color=ACCENT, size=8),
        line=dict(color=ACCENT, width=2),
        name="Inercia",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Número de clusters (k)",
        yaxis_title="Inercia (suma de distancias²)",
        height=380,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# =============================================================================
# GEOGRAPHIC / WORLD DATA
# =============================================================================

def plot_choropleth(
    df: pd.DataFrame,
    iso_col: str = "iso3",
    value_col: str = "value",
    country_col: str = "country",
    title: str = "Mapa Mundial",
    color_scale: str = "Viridis",
    label: str = "Valor",
) -> go.Figure:
    """
    Choropleth map using Plotly's built-in country geometries (ISO-3 codes).

    Parameters
    ----------
    df : DataFrame with at least iso3 codes and a numeric value column.
    iso_col : column name holding ISO alpha-3 country codes.
    value_col : column name with the numeric values to map.
    country_col : column used for hover labels (country name).
    color_scale : any Plotly color scale name ('Viridis', 'RdYlGn', etc.).
    label : legend / colorbar title.
    """
    fig = px.choropleth(
        df,
        locations=iso_col,
        color=value_col,
        hover_name=country_col,
        hover_data={iso_col: False, value_col: ":.2f"},
        color_continuous_scale=color_scale,
        title=title,
        labels={value_col: label},
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#334155",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0f172a",
        showframe=False,
        projection_type="natural earth",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        coloraxis_colorbar=dict(
            title=label,
            tickfont=dict(size=11),
            len=0.6,
        ),
        geo=dict(bgcolor="#0f172a"),
    )
    return fig
