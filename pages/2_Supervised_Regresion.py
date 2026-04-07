"""
pages/2_Supervised_Regresion.py
=================================
Módulo interactivo de Regresión Supervisada.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.plots import (
    plot_predicted_vs_actual,
    plot_residuals,
    plot_learning_curve,
    plot_feature_importance,
    plot_coefficients,
)
from utils.styles import inject_base_css

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Regresión · ML Explorer", page_icon="📈", layout="wide")

inject_base_css()
st.markdown(
    "<style>.page-header { background: linear-gradient(135deg, #1a1a2e, #1a0a2e); }</style>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-header">
        <h1>📈 Regresión Supervisada</h1>
        <p>Predice valores numéricos continuos y evalúa el rendimiento de distintos modelos de regresión.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("📁 Dataset")
    dataset_name = st.selectbox(
        "Selecciona dataset",
        ["California Housing", "Diabetes", "Sintético"],
        help=(
            "California Housing: precios de viviendas en California.\n"
            "Diabetes: progresión de la enfermedad a un año.\n"
            "Sintético: generado con make_regression."
        ),
    )
    if dataset_name == "Sintético":
        n_samples_synth = st.slider("N° muestras", 100, 2000, 500, step=100)
        n_features_synth = st.slider("N° variables", 5, 50, 15, step=5)
        noise_synth = st.slider("Ruido", 0, 100, 20, step=5)

    st.subheader("🧠 Algoritmo")
    algo_name = st.selectbox(
        "Selecciona algoritmo",
        ["Linear Regression", "Ridge", "Lasso",
         "Decision Tree Regressor", "Random Forest Regressor"],
    )

    st.subheader("✂️ División de datos")
    test_size = st.slider("% datos de prueba", 10, 40, 20, step=5) / 100
    random_state = st.number_input("Semilla aleatoria", 0, 999, 42, step=1)

    st.subheader("🔧 Hiperparámetros")
    hparams: dict = {}

    if algo_name == "Linear Regression":
        hparams["fit_intercept"] = st.checkbox("Ajustar intercepto", value=True)

    elif algo_name == "Ridge":
        hparams["alpha"] = st.select_slider(
            "Alpha (regularización)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], value=1.0,
            help="Alpha alto → mayor penalización, coeficientes más pequeños.",
        )
        hparams["fit_intercept"] = st.checkbox("Ajustar intercepto", value=True)

    elif algo_name == "Lasso":
        hparams["alpha"] = st.select_slider(
            "Alpha (regularización)", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], value=0.1,
            help="Lasso tiende a poner coeficientes exactamente a 0 (selección de variables).",
        )
        hparams["max_iter"] = st.slider("Max iteraciones", 500, 5000, 1000, step=500)

    elif algo_name == "Decision Tree Regressor":
        hparams["max_depth"] = st.slider("Profundidad máx.", 1, 20, 5)
        hparams["min_samples_split"] = st.slider("min_samples_split", 2, 30, 5)
        hparams["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 20, 2)

    elif algo_name == "Random Forest Regressor":
        hparams["n_estimators"] = st.slider("N° de árboles", 10, 300, 100, step=10)
        hparams["max_depth"] = st.slider("Profundidad máx.", 1, 20, 8)
        hparams["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)

    show_lc = st.checkbox(
        "Mostrar curva de aprendizaje",
        value=False,
        help="Más lento (usa cross-validation). Desactiva para mayor velocidad.",
    )

    st.markdown("---")
    train_btn = st.button("🚀 Entrenar modelo", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_dataset(name: str, **kwargs):
    if name == "California Housing":
        bunch = fetch_california_housing()
        return bunch.data, bunch.target, list(bunch.feature_names), "Precio mediano (×$100k)"
    elif name == "Diabetes":
        bunch = load_diabetes()
        return bunch.data, bunch.target, list(bunch.feature_names), "Progresión enfermedad"
    else:  # Sintético
        X, y = make_regression(
            n_samples=kwargs.get("n_samples", 500),
            n_features=kwargs.get("n_features", 15),
            noise=kwargs.get("noise", 20),
            random_state=42,
        )
        fn = [f"x{i+1}" for i in range(X.shape[1])]
        return X, y, fn, "Variable objetivo (sintético)"


def build_model(algo: str, hp: dict, rs: int):
    if algo == "Linear Regression":
        return LinearRegression(fit_intercept=hp.get("fit_intercept", True))
    elif algo == "Ridge":
        return Ridge(alpha=hp["alpha"], fit_intercept=hp.get("fit_intercept", True))
    elif algo == "Lasso":
        return Lasso(alpha=hp["alpha"], max_iter=hp.get("max_iter", 1000), random_state=rs)
    elif algo == "Decision Tree Regressor":
        return DecisionTreeRegressor(
            max_depth=hp["max_depth"], min_samples_split=hp["min_samples_split"],
            min_samples_leaf=hp["min_samples_leaf"], random_state=rs,
        )
    elif algo == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"], random_state=rs, n_jobs=-1,
        )


ALGO_DESCRIPTIONS = {
    "Linear Regression": (
        "**Regresión Lineal** modela la relación entre variables predictoras y la variable "
        "objetivo como una función lineal: **ŷ = β₀ + β₁x₁ + … + βₙxₙ**. "
        "Minimiza la suma de errores cuadráticos (OLS). Es muy interpretable: cada "
        "coeficiente indica el cambio esperado en *y* por una unidad de cambio en *xᵢ*. "
        "No funciona bien si la relación real es no lineal."
    ),
    "Ridge": (
        "**Ridge** añade una penalización L2 (β₁² + β₂² + …) a la función de pérdida. "
        "Esto reduce la magnitud de los coeficientes sin llevarlos a cero exactamente, "
        "reduciendo el sobreajuste cuando hay muchas variables correlacionadas. "
        "El parámetro **alpha** controla la intensidad de la regularización."
    ),
    "Lasso": (
        "**Lasso** usa regularización L1 (|β₁| + |β₂| + …). A diferencia de Ridge, "
        "puede llevar coeficientes exactamente a **cero**, haciendo selección automática "
        "de variables. Ideal cuando se sospecha que sólo unas pocas variables son relevantes."
    ),
    "Decision Tree Regressor": (
        "**Árbol de Decisión para Regresión** divide el espacio de variables en regiones "
        "rectangulares y predice la media de la variable objetivo en cada región. Puede "
        "capturar relaciones no lineales. La **profundidad máxima** es el principal control "
        "de complejidad: árboles muy profundos sobreajustan fácilmente."
    ),
    "Random Forest Regressor": (
        "**Random Forest para Regresión** promedia las predicciones de muchos árboles de "
        "decisión entrenados sobre submuestras aleatorias de datos y variables (bagging + "
        "feature randomness). Reduce drásticamente la varianza respecto a un árbol individual "
        "y ofrece importancias de variables robustas."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
synth_kwargs = {}
if dataset_name == "Sintético":
    synth_kwargs = {
        "n_samples": n_samples_synth,
        "n_features": n_features_synth,
        "noise": noise_synth,
    }

X, y, feature_names, target_label = get_dataset(dataset_name, **synth_kwargs)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Muestras", X.shape[0])
col2.metric("Variables predictoras", X.shape[1])
col3.metric("Objetivo: min", f"{y.min():.2f}")
col4.metric("Objetivo: max", f"{y.max():.2f}")

st.markdown("---")

with st.expander(f"📖 ¿Cómo funciona {algo_name}?", expanded=False):
    st.markdown(ALGO_DESCRIPTIONS[algo_name])

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
if train_btn or "reg_model" in st.session_state:
    if train_btn:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = build_model(algo_name, hparams, int(random_state))

        with st.spinner(f"Entrenando {algo_name}..."):
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

        st.session_state["reg_model"]  = model
        st.session_state["reg_scaler"] = scaler
        st.session_state["reg_data"]   = (X_train_sc, X_test_sc, y_train, y_test, y_pred)
        st.session_state["reg_full"]   = (X, y)
        st.session_state["reg_meta"]   = {
            "algo": algo_name, "dataset": dataset_name,
            "feature_names": feature_names, "target_label": target_label,
            "show_lc": show_lc,
        }
        st.success("✅ Modelo entrenado exitosamente.")

    model   = st.session_state["reg_model"]
    X_train_sc, X_test_sc, y_train, y_test, y_pred = st.session_state["reg_data"]
    X_full, y_full = st.session_state["reg_full"]
    meta = st.session_state["reg_meta"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.markdown("### 📊 Métricas de Evaluación")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{r2:.4f}",
              help="1 = predicción perfecta; 0 = igual que la media; negativo = peor que la media.")
    m2.metric("MAE", f"{mae:.4f}",
              help="Error Absoluto Medio: promedio de |real − predicho|.")
    m3.metric("MSE", f"{mse:.4f}",
              help="Error Cuadrático Medio: penaliza más los errores grandes.")
    m4.metric("RMSE", f"{rmse:.4f}",
              help="Raíz del MSE; misma unidad que la variable objetivo.")

    st.markdown("---")

    # ── Result tabs ───────────────────────────────────────────────────────────
    tab_pred, tab_res, tab_coef, tab_lc = st.tabs(
        ["🎯 Predicho vs Real", "📉 Residuos", "📊 Coeficientes / Importancia", "📚 Curva Aprendizaje"]
    )

    with tab_pred:
        st.markdown("#### Predicho vs. Real")
        st.markdown(
            "Los puntos deberían estar sobre la línea punteada (predicción perfecta). "
            "La dispersión indica el error del modelo."
        )
        fig_pva = plot_predicted_vs_actual(
            y_test, y_pred,
            title=f"Predicho vs. Real — {meta['algo']} · {meta['dataset']}",
        )
        st.plotly_chart(fig_pva, use_container_width=True)

    with tab_res:
        st.markdown("#### Análisis de Residuos")
        st.markdown(
            "Los residuos deberían distribuirse aleatoriamente alrededor del cero. "
            "Patrones sistemáticos indican que el modelo no captura toda la señal."
        )
        fig_res = plot_residuals(y_test, y_pred)
        st.plotly_chart(fig_res, use_container_width=True)

        # Estadísticas de residuos
        residuals = y_test - y_pred
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Residuo medio", f"{residuals.mean():.4f}")
        rc2.metric("Desv. estándar", f"{residuals.std():.4f}")
        rc3.metric("Residuo mín.", f"{residuals.min():.4f}")
        rc4.metric("Residuo máx.", f"{residuals.max():.4f}")

    with tab_coef:
        fn = meta["feature_names"]
        if hasattr(model, "feature_importances_"):
            st.markdown("#### Importancia de Variables")
            fig_fi = plot_feature_importance(
                model.feature_importances_, fn,
                title=f"Importancia — {meta['algo']}",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        elif hasattr(model, "coef_"):
            coef = model.coef_.ravel()
            st.markdown("#### Coeficientes del Modelo")
            st.markdown(
                "Barras verdes = efecto positivo sobre la predicción; "
                "rojas = efecto negativo. Los coeficientes están sobre variables escaladas."
            )
            fig_coef = plot_coefficients(coef, fn,
                                          title=f"Coeficientes — {meta['algo']}")
            st.plotly_chart(fig_coef, use_container_width=True)

            # Table
            df_coef = pd.DataFrame({"Variable": fn, "Coeficiente": coef})
            df_coef["Abs"] = df_coef["Coeficiente"].abs()
            df_coef = df_coef.sort_values("Abs", ascending=False).drop(columns="Abs")
            st.dataframe(df_coef, use_container_width=True, hide_index=True)
        else:
            st.info("Este modelo no expone coeficientes ni importancias directamente.")

    with tab_lc:
        if meta.get("show_lc", False):
            st.markdown("#### Curva de Aprendizaje")
            st.markdown(
                "Muestra cómo varía el R² de entrenamiento y validación a medida que "
                "se añaden más datos. Si ambas líneas convergen a un valor alto → el modelo "
                "generaliza bien. Si hay gran diferencia → sobreajuste."
            )
            with st.spinner("Calculando curva de aprendizaje (cross-validation)..."):
                scaler_lc = StandardScaler()
                X_lc = scaler_lc.fit_transform(X_full)
                from sklearn.base import clone
                model_lc = clone(model)
                fig_lc = plot_learning_curve(model_lc, X_lc, y_full, cv=5)
            st.plotly_chart(fig_lc, use_container_width=True)
        else:
            st.info(
                "Activa **Mostrar curva de aprendizaje** en el panel lateral y vuelve a "
                "entrenar para ver este gráfico (requiere cross-validation, puede tardar)."
            )

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem; color: #64748b;">
            <div style="font-size: 3rem;">⬅️</div>
            <p style="font-size: 1.1rem;">Selecciona dataset, algoritmo e hiperparámetros en el panel lateral,
            luego pulsa <strong>Entrenar modelo</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
