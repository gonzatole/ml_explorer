"""
pages/5_ML_Datos_Reales.py
==========================
Módulo: ML en Datos Reales
Entrena modelos de ML (regresión / clasificación) sobre indicadores
reales del Banco Mundial. Permite seleccionar features y target,
entrenar, evaluar y comparar algoritmos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

from utils.worldbank import get_indicator, INDICATORS
from utils.plots import plot_confusion_matrix, plot_predicted_vs_actual
from utils.model_export import build_pipeline, export_pipeline, get_model_metadata
from utils.styles import inject_base_css

st.set_page_config(page_title="ML Datos Reales · ML Explorer", layout="wide")

inject_base_css()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    ">
        <h1 style="color:#fff; margin:0; font-size:2.2rem;">🤖 ML en Datos Reales</h1>
        <p style="color:#a0aec0; margin-top:0.5rem;">
            Entrena modelos sobre indicadores reales del Banco Mundial
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: Configuration ────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

year_ml = st.sidebar.slider("Año para el dataset", 2000, 2023, 2022)

ind_names = list(INDICATORS.keys())
feature_names = st.sidebar.multiselect(
    "Variables predictoras (features)",
    ind_names,
    default=ind_names[1:3],  # PIB per cápita, Esperanza de vida
    max_selections=5,
    key="ml_features",
)

target_name = st.sidebar.selectbox(
    "Variable objetivo (target)",
    ind_names,
    index=0,  # Población total por defecto
    key="ml_target",
)

if target_name in feature_names:
    st.sidebar.warning("⚠️ El target no puede ser una feature. Removido de features.")
    feature_names = [f for f in feature_names if f != target_name]

if not feature_names:
    st.error("Selecciona al menos una variable predictora.")
    st.stop()

test_size = st.sidebar.slider("Test set %", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random state", 0, 10000, 42, key="rs")

# ── Load and merge data ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Cargando datos…")
def load_merged_data(year: int, feature_codes: list[str], target_code: str) -> pd.DataFrame:
    """Load and merge all indicators into a single table."""
    dfs = []
    for ind_code in feature_codes + [target_code]:
        df = get_indicator(ind_code, year)
        if not df.empty:
            df = df[["iso3", "country", "value"]].rename(
                columns={"value": ind_code}
            )
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=["iso3", "country"], how="inner")

    return result


feature_codes = [INDICATORS[f] for f in feature_names]
target_code = INDICATORS[target_name]

df_merged = load_merged_data(year_ml, feature_codes, target_code)

if df_merged.empty:
    st.error("No hay datos disponibles para esta combinación de indicadores.")
    st.stop()

st.info(f"📊 Dataset: {len(df_merged)} países con datos completos")

# ── Prepare data for ML ────────────────────────────────────────────────────────
X = df_merged[feature_codes].values
y = df_merged[target_code].values
countries = df_merged["country"].values

# Detect problem type (regression vs classification)
n_unique_y = len(np.unique(y))
is_regression = n_unique_y > 10  # Arbitrary threshold

if is_regression:
    problem_type = "Regresión"
    model_options = {
        "Linear Regression": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "SVR (RBF)": SVR(kernel="rbf", C=100),
    }
else:
    problem_type = "Clasificación"
    # Convert to classes if continuous
    y_binned = pd.cut(y, bins=3, labels=[f"Clase {i}" for i in range(3)], duplicates='drop')
    y = pd.factorize(y_binned)[0]
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", C=100, random_state=random_state),
    }

# ── Model selection and training ───────────────────────────────────────────────
st.subheader(f"🎯 Problema: {problem_type}")

col_algo, col_train = st.columns([2, 1])
with col_algo:
    selected_model_name = st.selectbox(
        "Algoritmo",
        list(model_options.keys()),
        key="model_select",
    )
with col_train:
    train_button = st.button("🚀 Entrenar modelo", use_container_width=True)

# ── Training and evaluation ────────────────────────────────────────────────────
if train_button:
    with st.spinner("Entrenando…"):
        # Split
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)),
            test_size=test_size,
            random_state=int(random_state),
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = model_options[selected_model_name]
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Store in session state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.X_train = X_train_scaled
        st.session_state.X_test = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y_pred_train = y_pred_train
        st.session_state.y_pred_test = y_pred_test
        st.session_state.idx_train = idx_train
        st.session_state.idx_test = idx_test
        st.session_state.problem_type = problem_type
        st.session_state.feature_names = feature_names
        st.session_state.target_name = target_name
        st.session_state.trained = True

# ── Display results if model trained ───────────────────────────────────────────
if st.session_state.get("trained"):
    st.success("✅ Modelo entrenado exitosamente")
    st.markdown("<br>", unsafe_allow_html=True)

    y_pred_test = st.session_state.y_pred_test
    y_test = st.session_state.y_test
    feature_names = st.session_state.feature_names

    if st.session_state.problem_type == "Regresión":
        # ── Regression Metrics ─────────────────────────────────────────────
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score", f"{r2:.4f}")
        m2.metric("MAE", f"{mae:,.2f}")
        m3.metric("RMSE", f"{rmse:,.2f}")
        m4.metric("Test set", f"{len(y_test)} países")

        st.markdown("<br>", unsafe_allow_html=True)

        # Plot predictions vs actual
        fig_pred = plot_predicted_vs_actual(y_test, y_pred_test)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Feature importance if available
        if hasattr(st.session_state.model, 'feature_importances_'):
            importances = st.session_state.model.feature_importances_
            import plotly.express as px
            df_imp = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=True)
            fig_imp = px.bar(
                df_imp, x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
                title="Importancia de Variables",
            )
            fig_imp.update_layout(
                coloraxis_showscale=False,
                height=300,
                margin=dict(l=160, r=20, t=60, b=40),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        # Coefficients if available
        elif hasattr(st.session_state.model, 'coef_'):
            coef = st.session_state.model.coef_
            import plotly.express as px
            df_coef = pd.DataFrame({
                "feature": feature_names,
                "coef": coef,
            }).sort_values("coef")
            colors = ["#00CC96" if v >= 0 else "#EF553B" for v in df_coef["coef"]]
            fig_coef = px.bar(
                df_coef, x="coef", y="feature",
                orientation="h",
                color_discrete_sequence=[colors[0]],
                title="Coeficientes del Modelo",
            )
            fig_coef.update_layout(
                height=300,
                margin=dict(l=160, r=20, t=60, b=40),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_coef, use_container_width=True)

    else:
        # ── Classification Metrics ────────────────────────────────────────
        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("F1 Score (weighted)", f"{f1:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        class_names = [f"Clase {i}" for i in range(cm.shape[0])]
        fig_cm = plot_confusion_matrix(y_test, y_pred_test, class_names=class_names)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        with st.expander("📋 Reporte detallado"):
            report = classification_report(y_test, y_pred_test, output_dict=False)
            st.code(report, language="text")

    # ── Model Export Section ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📥 Descargar Modelo")
    st.markdown(
        "El modelo entrenado incluye normalización automática (StandardScaler) "
        "y puede reutilizarse en otros proyectos (FastAPI, Jupyter, etc)."
    )

    # Build pipeline
    pipeline = build_pipeline(st.session_state.scaler, st.session_state.model)
    metadata = get_model_metadata(
        pipeline,
        st.session_state.feature_names,
        st.session_state.target_name
    )

    # Display metadata
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Tipo de modelo", metadata["model_type"])
        st.metric("Variables predictoras", metadata["n_features"])
    with col_info2:
        st.metric("Problema", st.session_state.problem_type)
        st.metric("Fecha de entrenamiento", metadata["exported_at"][:10])

    # Download button
    model_name = (
        f"{selected_model_name.lower().replace(' ', '_')}_"
        f"{st.session_state.target_name.lower().replace(' ', '_')}"
    )
    pipeline_bytes = export_pipeline(pipeline, model_name)

    st.download_button(
        label=f"⬇️ Descargar {model_name}.joblib",
        data=pipeline_bytes,
        file_name=f"{model_name}.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )

    st.caption(
        "💡 **Tip:** Carga el modelo con `from joblib import load; "
        "pipeline = load('modelo.joblib'); predicción = pipeline.predict(X_new)`"
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.85rem;">
        Datos: <a href="https://data.worldbank.org" style="color:#636EFA;">Banco Mundial</a>
        · Modelos: scikit-learn · Sprint 2–3 — ML en Datos Reales + Artefactos
    </div>
    """,
    unsafe_allow_html=True,
)
