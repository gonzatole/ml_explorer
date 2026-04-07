"""
app.py — ML Explorer: Página de Inicio
=======================================
Punto de entrada principal de la aplicación Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine

from utils.styles import inject_base_css

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_base_css()

# ── Custom CSS (home-page only) ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 3rem 2.5rem; margin-bottom: 2rem;
        text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero h1 { font-size: 3rem; font-weight: 700; color: #ffffff; margin: 0; }
    .hero p  { font-size: 1.2rem; color: #a0aec0; margin-top: 0.75rem; }

    .nav-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155; border-radius: 12px; padding: 1.5rem;
        text-align: center; transition: transform .2s, box-shadow .2s;
        cursor: pointer; height: 100%; min-height: 180px;
    }
    .nav-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(99,110,250,0.25); border-color: #636EFA;
    }
    .nav-card .icon { font-size: 2.8rem; }
    .nav-card h3 { color: #e2e8f0; font-size: 1.1rem; margin: 0.5rem 0 0.3rem; }
    .nav-card p  { color: #94a3b8; font-size: 0.85rem; margin: 0; }

    .section-title {
        font-size: 1.5rem; font-weight: 700; color: #e2e8f0;
        border-left: 4px solid #636EFA; padding-left: 0.75rem; margin: 2rem 0 1rem;
    }

    .concept-card { border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 0.5rem; }
    .concept-sup  { background: rgba(99,110,250,0.12); border: 1px solid rgba(99,110,250,0.3); }
    .concept-uns  { background: rgba(0,204,150,0.10); border: 1px solid rgba(0,204,150,0.3); }
    .concept-card h4 { margin: 0 0 0.4rem; font-size: 1.05rem; }
    .concept-card p  { margin: 0; font-size: 0.88rem; color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🤖 ML Explorer</h1>
        <p>Tu laboratorio interactivo de Machine Learning en Python</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Nav cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Módulos de Aprendizaje</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="nav-card">
            <div class="icon">🎯</div>
            <h3>Clasificación Supervisada</h3>
            <p>Logistic Regression, Decision Tree, Random Forest, KNN, SVM — con métricas,
            matriz de confusión y curva ROC.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="nav-card">
            <div class="icon">📈</div>
            <h3>Regresión Supervisada</h3>
            <p>Linear, Ridge, Lasso, Decision Tree, Random Forest — con R², MAE, RMSE
            y curva de aprendizaje.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="nav-card">
            <div class="icon">🔵</div>
            <h3>Clustering No Supervisado</h3>
            <p>K-Means, DBSCAN, Agglomerative — con método del codo, Silhouette score
            y visualización PCA.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

c4, _, _ = st.columns(3)
with c4:
    st.markdown(
        """
        <div class="nav-card">
            <div class="icon">🌍</div>
            <h3>Datos Reales del Mundo</h3>
            <p>Banco Mundial API · Mapas coropléticos, rankings, series temporales
            y comparador de indicadores entre países.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

c5, _, _ = st.columns(3)
with c5:
    st.markdown(
        """
        <div class="nav-card">
            <div class="icon">🔬</div>
            <h3>ML en Datos Reales</h3>
            <p>Entrena modelos de regresión y clasificación sobre indicadores
            reales del Banco Mundial. Compara algoritmos y visualiza predicciones.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

c6, _, _ = st.columns(3)
with c6:
    st.markdown(
        """
        <div class="nav-card">
            <div class="icon">📊</div>
            <h3>Business Intelligence</h3>
            <p>Dashboard ejecutivo con KPIs globales, análisis regional,
            tendencias temporales, y predicciones ML en vivo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Usa el menú lateral para navegar entre los módulos.", icon="ℹ️")

# ── ML concept overview ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">¿Qué es el Machine Learning?</div>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        """
        <div class="concept-card concept-sup">
            <h4 style="color:#636EFA;">🎓 Aprendizaje Supervisado</h4>
            <p>
            El modelo aprende a partir de datos <strong>etiquetados</strong>: se le muestran
            pares (entrada → salida correcta) y aprende a generalizar la función que los relaciona.<br><br>
            <strong>Clasificación:</strong> la salida es una categoría discreta (spam / no-spam,
            tipo de flor, dígito 0–9).<br>
            <strong>Regresión:</strong> la salida es un número continuo (precio de casa,
            temperatura futura).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="concept-card concept-uns">
            <h4 style="color:#00CC96;">🔍 Aprendizaje No Supervisado</h4>
            <p>
            El modelo trabaja con datos <strong>sin etiquetas</strong>: busca estructura,
            patrones o grupos ocultos por sí solo.<br><br>
            <strong>Clustering:</strong> agrupa observaciones similares (K-Means, DBSCAN).<br>
            <strong>Reducción de dimensionalidad:</strong> comprime muchas variables en unas
            pocas conservando la mayor información posible (PCA, t-SNE).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Visual comparison chart ────────────────────────────────────────────────────
st.markdown('<div class="section-title">Panorama de Algoritmos</div>', unsafe_allow_html=True)

algo_data = {
    "Algoritmo": [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "KNN", "SVM",
        "Linear Regression", "Ridge / Lasso", "Decision Tree Reg.", "RF Regressor",
        "K-Means", "DBSCAN", "Agglomerative",
    ],
    "Tipo": [
        "Clasificación", "Clasificación", "Clasificación", "Clasificación", "Clasificación",
        "Regresión", "Regresión", "Regresión", "Regresión",
        "Clustering", "Clustering", "Clustering",
    ],
    "Interpretabilidad": [8, 9, 5, 6, 4, 9, 8, 8, 5, 7, 6, 6],
    "Flexibilidad": [5, 7, 9, 7, 8, 4, 5, 7, 9, 6, 8, 7],
}

df_algo = pd.DataFrame(algo_data)

color_map = {
    "Clasificación": "#636EFA",
    "Regresión": "#EF553B",
    "Clustering": "#00CC96",
}

fig_bubble = px.scatter(
    df_algo,
    x="Interpretabilidad",
    y="Flexibilidad",
    color="Tipo",
    text="Algoritmo",
    color_discrete_map=color_map,
    size=[14] * len(df_algo),
    title="Comparativa: Interpretabilidad vs. Flexibilidad",
)
fig_bubble.update_traces(textposition="top center", marker=dict(opacity=0.85))
fig_bubble.update_layout(
    height=480,
    xaxis=dict(range=[2, 11], title="← Caja negra   |   Interpretable →"),
    yaxis=dict(range=[2, 11], title="← Rígido   |   Flexible →"),
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font_color="#e2e8f0",
    margin=dict(l=60, r=20, t=60, b=60),
)
st.plotly_chart(fig_bubble, use_container_width=True)

# ── Dataset previews ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Datasets incluidos</div>', unsafe_allow_html=True)

tab_iris, tab_cancer, tab_wine, tab_digits = st.tabs(
    ["🌸 Iris", "🏥 Breast Cancer", "🍷 Wine", "🔢 Digits"]
)

@st.cache_data
def _load_iris():
    d = load_iris(as_frame=True)
    df = d.frame.copy()
    df["target_name"] = d.target_names[d.target]
    return df, d

@st.cache_data
def _load_cancer():
    d = load_breast_cancer(as_frame=True)
    df = d.frame.copy()
    df["target_name"] = d.target_names[d.target]
    return df, d

@st.cache_data
def _load_wine():
    d = load_wine(as_frame=True)
    df = d.frame.copy()
    df["target_name"] = d.target_names[d.target]
    return df, d

@st.cache_data
def _load_digits():
    d = load_digits(as_frame=True)
    return d.frame, d

with tab_iris:
    df_iris, iris = _load_iris()
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.metric("Muestras", iris.data.shape[0])
        st.metric("Variables", iris.data.shape[1])
        st.metric("Clases", len(iris.target_names))
        with st.expander("Ver datos brutos"):
            st.dataframe(df_iris.head(10), use_container_width=True)
    with col_b:
        fig_iris = px.scatter(
            df_iris,
            x="sepal length (cm)", y="petal length (cm)",
            color="target_name",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title="Iris — Sépalo vs. Pétalo",
            labels={"target_name": "Especie"},
        )
        fig_iris.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                                font_color="#e2e8f0", height=350)
        st.plotly_chart(fig_iris, use_container_width=True)

with tab_cancer:
    df_cancer, cancer = _load_cancer()
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.metric("Muestras", cancer.data.shape[0])
        st.metric("Variables", cancer.data.shape[1])
        st.metric("Clases", 2)
        st.markdown("**Diagnóstico:** maligno / benigno")
        with st.expander("Ver datos brutos"):
            st.dataframe(df_cancer[["mean radius","mean texture","mean perimeter","mean area","target_name"]].head(10),
                         use_container_width=True)
    with col_b:
        fig_cancer = px.scatter(
            df_cancer,
            x="mean radius", y="mean concavity",
            color="target_name",
            color_discrete_map={"malignant": "#EF553B", "benign": "#00CC96"},
            title="Breast Cancer — Radio vs. Concavidad",
            labels={"target_name": "Diagnóstico"},
        )
        fig_cancer.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                                  font_color="#e2e8f0", height=350)
        st.plotly_chart(fig_cancer, use_container_width=True)

with tab_wine:
    df_wine, wine = _load_wine()
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.metric("Muestras", wine.data.shape[0])
        st.metric("Variables", wine.data.shape[1])
        st.metric("Clases", 3)
        st.markdown("**Cultivares:** class_0 / class_1 / class_2")
        with st.expander("Ver datos brutos"):
            st.dataframe(df_wine[["alcohol","malic_acid","ash","target_name"]].head(10),
                         use_container_width=True)
    with col_b:
        fig_wine = px.scatter(
            df_wine,
            x="alcohol", y="flavanoids",
            color="target_name",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Wine — Alcohol vs. Flavonoides",
            labels={"target_name": "Cultivar"},
        )
        fig_wine.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                                font_color="#e2e8f0", height=350)
        st.plotly_chart(fig_wine, use_container_width=True)

with tab_digits:
    df_digits, digits = _load_digits()
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.metric("Muestras", digits.data.shape[0])
        st.metric("Píxeles (vars)", digits.data.shape[1])
        st.metric("Clases", 10)
        st.markdown("**Dígitos:** 0 – 9 (imágenes 8×8 px)")
        with st.expander("Ver datos brutos"):
            st.dataframe(df_digits.head(5), use_container_width=True)
    with col_b:
        # Show a grid of 16 sample digit images
        import matplotlib.pyplot as plt
        fig_d, axes = plt.subplots(2, 8, figsize=(10, 2.8))
        fig_d.patch.set_facecolor("#0f172a")
        indices = np.random.choice(len(digits.images), 16, replace=False)
        for ax, idx in zip(axes.ravel(), indices):
            ax.imshow(digits.images[idx], cmap="Blues")
            ax.set_title(str(int(digits.target[idx])), color="white", fontsize=9, pad=2)
            ax.axis("off")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.85rem;">
        ML Explorer · Construido con
        <a href="https://streamlit.io" style="color:#636EFA;">Streamlit</a> &
        <a href="https://scikit-learn.org" style="color:#636EFA;">scikit-learn</a>
        · Datos: sklearn.datasets
    </div>
    """,
    unsafe_allow_html=True,
)
