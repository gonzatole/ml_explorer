"""
pages/1_Supervised_Clasificacion.py
=====================================
Módulo interactivo de Clasificación Supervisada.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, auc,
)
from sklearn.decomposition import PCA
from sklearn.base import clone

from utils.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_decision_boundary,
)
from utils.styles import inject_base_css

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clasificación · ML Explorer", page_icon="🎯", layout="wide")

inject_base_css()
st.markdown(
    """
    <style>
    .page-header { background: linear-gradient(135deg, #1a1a2e, #16213e); }
    .algo-box {
        background: #1e293b; border: 1px solid #334155; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }
    .algo-box h4 { color: #636EFA; margin: 0 0 0.3rem; }
    .algo-box p  { color: #94a3b8; margin: 0; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-header">
        <h1>🎯 Clasificación Supervisada</h1>
        <p>Entrena y evalúa clasificadores clásicos sobre datasets reales de scikit-learn.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — controles
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuración")

    # ── Dataset ──────────────────────────────────────────────────────────────
    st.subheader("📁 Dataset")
    dataset_name = st.selectbox(
        "Selecciona dataset",
        ["Iris", "Breast Cancer", "Digits"],
        help="Datasets clásicos incluidos en scikit-learn.",
    )

    # ── Algorithm ────────────────────────────────────────────────────────────
    st.subheader("🧠 Algoritmo")
    algo_name = st.selectbox(
        "Selecciona algoritmo",
        ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"],
    )

    # ── Train/test split ──────────────────────────────────────────────────────
    st.subheader("✂️ División de datos")
    test_size = st.slider(
        "% datos de prueba", 10, 40, 20, step=5,
        help="Porcentaje de muestras reservadas para evaluar el modelo.",
    ) / 100
    random_state = st.number_input("Semilla aleatoria", 0, 999, 42, step=1)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    st.subheader("🔧 Hiperparámetros")
    hparams: dict = {}

    if algo_name == "Logistic Regression":
        hparams["C"] = st.select_slider(
            "C (regularización inversa)",
            options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0,
            help="Valores pequeños → más regularización.",
        )
        hparams["max_iter"] = st.slider("Max iteraciones", 100, 2000, 300, step=100)
        hparams["solver"] = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])

    elif algo_name == "Decision Tree":
        hparams["max_depth"] = st.slider("Profundidad máx.", 1, 20, 5,
                                          help="Controla la complejidad del árbol.")
        hparams["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
        hparams["criterion"] = st.selectbox("Criterio", ["gini", "entropy", "log_loss"])

    elif algo_name == "Random Forest":
        hparams["n_estimators"] = st.slider("N° de árboles", 10, 300, 100, step=10)
        hparams["max_depth"] = st.slider("Profundidad máx.", 1, 20, 5)
        hparams["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)

    elif algo_name == "KNN":
        hparams["n_neighbors"] = st.slider("N° vecinos (k)", 1, 30, 5,
                                             help="k=1 → muy sensible; k grande → suaviza fronteras.")
        hparams["weights"] = st.selectbox("Pesos", ["uniform", "distance"])
        hparams["metric"] = st.selectbox("Métrica", ["euclidean", "manhattan", "minkowski"])

    elif algo_name == "SVM":
        hparams["C"] = st.select_slider(
            "C", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0
        )
        hparams["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        if hparams["kernel"] == "poly":
            hparams["degree"] = st.slider("Grado del polinomio", 2, 6, 3)

    st.markdown("---")
    train_btn = st.button("🚀 Entrenar modelo", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_dataset(name: str):
    loaders = {
        "Iris": load_iris,
        "Breast Cancer": load_breast_cancer,
        "Digits": load_digits,
    }
    bunch = loaders[name]()
    return bunch.data, bunch.target, list(bunch.feature_names), list(bunch.target_names)


def build_model(algo: str, hp: dict, rs: int):
    if algo == "Logistic Regression":
        return LogisticRegression(
            C=hp["C"], max_iter=hp["max_iter"], solver=hp["solver"],
            multi_class="auto", random_state=rs,
        )
    elif algo == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=hp["max_depth"], min_samples_split=hp["min_samples_split"],
            criterion=hp["criterion"], random_state=rs,
        )
    elif algo == "Random Forest":
        return RandomForestClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"], random_state=rs, n_jobs=-1,
        )
    elif algo == "KNN":
        return KNeighborsClassifier(
            n_neighbors=hp["n_neighbors"], weights=hp["weights"], metric=hp["metric"],
        )
    elif algo == "SVM":
        return SVC(
            C=hp["C"], kernel=hp["kernel"],
            degree=hp.get("degree", 3),
            probability=True, random_state=rs,
        )


@st.cache_data(show_spinner="Calculando frontera de decisión...")
def compute_decision_boundary_fig(
    dataset_name: str,
    algo_name: str,
    hparams_key: tuple,   # tuple(sorted(hparams.items())) — hashable
    test_size: float,
    random_state: int,
) -> go.Figure:
    """Entrena un modelo 2D (PCA) y devuelve la figura de frontera de decisión.
    Cacheada por parámetros: solo se recalcula cuando cambia dataset/algo/hiperparámetros."""
    X, y, _, class_names = get_dataset(dataset_name)
    hp = dict(hparams_key)
    model_2d = build_model(algo_name, hp, random_state)

    X_2d = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(X)
    )
    X_2d_train, _, y_2d_train, _ = train_test_split(
        X_2d, y, test_size=test_size, random_state=42, stratify=y
    )
    model_2d.fit(X_2d_train, y_2d_train)
    return plot_decision_boundary(
        model_2d, X_2d, y,
        feature_names=["PC1", "PC2"],
        class_names=list(class_names),
    )


ALGO_DESCRIPTIONS = {
    "Logistic Regression": (
        "**Regresión Logística** aplica la función sigmoide sobre una combinación lineal "
        "de las variables para estimar probabilidades de clase. Es simple, rápida e "
        "interpretable; funciona muy bien cuando las clases son aproximadamente linealmente "
        "separables. El parámetro **C** controla la regularización: C pequeño → modelo más "
        "simple; C grande → más ajuste a los datos de entrenamiento."
    ),
    "Decision Tree": (
        "**Árbol de Decisión** divide el espacio de variables mediante preguntas binarias "
        "sucesivas (p. ej. *¿longitud del sépalo > 5.0?*). Cada nodo hoja devuelve la clase "
        "más frecuente. Son muy interpretables pero propensos a sobreajuste; limitar la "
        "**profundidad máxima** actúa como regularización."
    ),
    "Random Forest": (
        "**Random Forest** entrena muchos árboles de decisión sobre submuestras aleatorias "
        "de datos y variables, luego agrega sus predicciones por voto. Reduce drásticamente "
        "el sobreajuste respecto a un árbol individual. El parámetro clave es **n_estimators**: "
        "más árboles → mayor estabilidad, pero mayor costo computacional."
    ),
    "KNN": (
        "**K-Nearest Neighbors** clasifica cada punto asignándole la clase mayoritaria entre "
        "sus **k** vecinos más cercanos. No tiene fase de entrenamiento real (es *lazy learning*). "
        "k=1 → muy sensible al ruido; k grande → fronteras más suaves. Escalar las variables "
        "es crucial porque usa distancias."
    ),
    "SVM": (
        "**Support Vector Machine** busca el hiperplano que maximiza el margen entre clases. "
        "Con el **kernel RBF** puede aprender fronteras no lineales. El parámetro **C** "
        "penaliza los errores de clasificación: C alto → intenta clasificar todo correctamente "
        "(riesgo de sobreajuste)."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — always visible dataset info
# ═══════════════════════════════════════════════════════════════════════════════
X, y, feature_names, class_names = get_dataset(dataset_name)
n_classes = len(class_names)
is_binary = n_classes == 2

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Muestras totales", X.shape[0])
col_info2.metric("Variables", X.shape[1])
col_info3.metric("Clases", n_classes)
col_info4.metric("Clases", ", ".join(class_names[:4]) + ("…" if n_classes > 4 else ""))

st.markdown("---")

# ── Algorithm explanation ──────────────────────────────────────────────────────
with st.expander(f"📖 ¿Cómo funciona {algo_name}?", expanded=False):
    st.markdown(ALGO_DESCRIPTIONS[algo_name])

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
if train_btn or "clf_model" in st.session_state:
    if train_btn:
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y
        )
        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        model = build_model(algo_name, hparams, int(random_state))

        with st.spinner(f"Entrenando {algo_name}..."):
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_proba = model.predict_proba(X_test_sc) if hasattr(model, "predict_proba") else None

        # Persist in session state
        st.session_state["clf_model"] = model
        st.session_state["clf_scaler"] = scaler
        st.session_state["clf_data"] = (X_train, X_train_sc, X_test, X_test_sc,
                                         y_train, y_test, y_pred, y_proba)
        st.session_state["clf_meta"] = {
            "algo": algo_name, "dataset": dataset_name,
            "feature_names": feature_names, "class_names": class_names,
            "n_classes": n_classes, "is_binary": is_binary,
            "test_size": test_size,
        }
        st.session_state["clf_hparams"] = hparams
        st.session_state["clf_random_state"] = int(random_state)
        st.success(f"✅ Modelo entrenado exitosamente.", icon="✅")

    # Load from session state
    model = st.session_state["clf_model"]
    (X_train, X_train_sc, X_test, X_test_sc,
     y_train, y_test, y_pred, y_proba) = st.session_state["clf_data"]
    meta = st.session_state["clf_meta"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    st.markdown("### 📊 Métricas de Evaluación")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.4f}", help="Fracción de predicciones correctas.")
    m2.metric("Precision", f"{prec:.4f}", help="De los que predije positivo, ¿cuántos lo eran? (weighted)")
    m3.metric("Recall", f"{rec:.4f}", help="De los positivos reales, ¿cuántos detecté? (weighted)")
    m4.metric("F1-Score", f"{f1:.4f}", help="Media armónica de Precision y Recall.")

    st.markdown("---")

    # ── Result tabs ───────────────────────────────────────────────────────────
    tab_cm, tab_roc, tab_fi, tab_db, tab_report = st.tabs(
        ["🟦 Matriz de Confusión", "📉 Curva ROC", "📊 Importancia", "🗺️ Frontera", "📋 Reporte"]
    )

    with tab_cm:
        st.markdown("#### Matriz de Confusión")
        st.markdown(
            "Filas = clases **reales**; columnas = clases **predichas**. "
            "La diagonal principal son los aciertos; el resto son errores."
        )
        fig_cm = plot_confusion_matrix(y_test, y_pred, class_names=meta["class_names"])
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab_roc:
        if meta["is_binary"] and y_proba is not None:
            st.markdown("#### Curva ROC (clasificación binaria)")
            st.markdown(
                "El **AUC** (área bajo la curva) mide la capacidad del modelo para "
                "distinguir entre clases. AUC=1 → perfecto; AUC=0.5 → aleatorio."
            )
            fig_roc = plot_roc_curve(y_test, y_proba[:, 1])
            st.plotly_chart(fig_roc, use_container_width=True)
        elif y_proba is not None and meta["n_classes"] > 2:
            st.markdown("#### Curvas ROC multi-clase (One-vs-Rest)")
            Y_bin = label_binarize(y_test, classes=np.unique(y_test))
            fig_mroc = go.Figure()
            for i, cls in enumerate(meta["class_names"]):
                fpr, tpr, _ = roc_curve(Y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                fig_mroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                               name=f"{cls} (AUC={roc_auc:.2f})"))
            fig_mroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                           line=dict(dash="dash", color="gray"),
                                           name="Aleatorio"))
            fig_mroc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                                    title="Curvas ROC por clase", height=430)
            st.plotly_chart(fig_mroc, use_container_width=True)
        else:
            st.info("La curva ROC requiere que el modelo soporte probabilidades.")

    with tab_fi:
        st.markdown("#### Importancia de Variables")
        if hasattr(model, "feature_importances_"):
            fig_fi = plot_feature_importance(
                model.feature_importances_,
                meta["feature_names"],
                title=f"Importancia de Variables — {meta['algo']}",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).mean(axis=0)
            fig_fi = plot_feature_importance(
                coef,
                meta["feature_names"],
                title=f"Importancia (|coeficientes|) — {meta['algo']}",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Este algoritmo no expone importancias de variables directamente (ej. KNN, SVM-RBF).")

    with tab_db:
        st.markdown("#### Frontera de Decisión (2D — PCA)")
        st.markdown(
            "Se proyectan las variables al plano de **las 2 primeras componentes principales** "
            "y se visualiza la frontera de decisión del clasificador."
        )
        fig_db = compute_decision_boundary_fig(
            meta["dataset"],
            meta["algo"],
            tuple(sorted(st.session_state.get("clf_hparams", {}).items())),
            meta["test_size"],
            st.session_state.get("clf_random_state", 42),
        )
        st.plotly_chart(fig_db, use_container_width=True)

    with tab_report:
        st.markdown("#### Reporte de Clasificación detallado")
        report_str = classification_report(
            y_test, y_pred,
            target_names=meta["class_names"],
            zero_division=0,
        )
        st.code(report_str, language="text")

        # Muestras por clase
        st.markdown("**Distribución del conjunto de prueba:**")
        unique, counts = np.unique(y_test, return_counts=True)
        df_dist = pd.DataFrame({
            "Clase": [meta["class_names"][i] for i in unique],
            "N real": counts,
            "N predicho": [np.sum(y_pred == i) for i in unique],
        })
        st.dataframe(df_dist, use_container_width=True, hide_index=True)

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
