"""
pages/3_Unsupervised_Clustering.py
=====================================
Módulo interactivo de Clustering No Supervisado.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris, make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import plotly.express as px
import plotly.graph_objects as go

from utils.plots import plot_clusters_2d, plot_elbow_method
from utils.styles import inject_base_css

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clustering · ML Explorer", page_icon="🔵", layout="wide")

inject_base_css()
st.markdown(
    """
    <style>
    .page-header { background: linear-gradient(135deg, #0a1628, #0a2818); }
    .concept-box {
        background: rgba(0,204,150,0.08); border: 1px solid rgba(0,204,150,0.25);
        border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    }
    .concept-box h4 { color: #00CC96; margin: 0 0 0.4rem; }
    .concept-box p  { color: #94a3b8; margin: 0; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-header">
        <h1>🔵 Clustering No Supervisado</h1>
        <p>Descubre grupos ocultos en los datos sin necesidad de etiquetas.</p>
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
        ["Iris (sin etiquetas)", "make_blobs", "make_moons", "make_circles"],
        help=(
            "Iris: dataset clásico sin las etiquetas de especie.\n"
            "make_blobs: grupos gaussianos bien separados.\n"
            "make_moons / make_circles: formas no convexas (desafían K-Means)."
        ),
    )

    if dataset_name == "make_blobs":
        n_samples_gen = st.slider("N° muestras", 100, 2000, 500, step=100)
        n_centers_gen = st.slider("N° centros reales", 2, 8, 4)
        cluster_std_gen = st.slider("Desv. estándar", 0.3, 3.0, 1.0, step=0.1)
    elif dataset_name in ("make_moons", "make_circles"):
        n_samples_gen = st.slider("N° muestras", 100, 1000, 300, step=100)
        noise_gen = st.slider("Ruido", 0.0, 0.5, 0.1, step=0.01)

    st.subheader("🧠 Algoritmo")
    algo_name = st.selectbox(
        "Selecciona algoritmo",
        ["K-Means", "DBSCAN", "Agglomerative Clustering"],
    )

    st.subheader("🔧 Hiperparámetros")
    hparams: dict = {}

    if algo_name == "K-Means":
        hparams["n_clusters"] = st.slider(
            "N° clusters (k)", 2, 12, 3,
            help="Número de grupos que el algoritmo intentará encontrar.",
        )
        hparams["init"] = st.selectbox(
            "Método de inicialización", ["k-means++", "random"],
            help="k-means++ suele converger más rápido y a mejores soluciones.",
        )
        hparams["max_iter"] = st.slider("Max iteraciones", 50, 500, 300, step=50)
        hparams["n_init"] = st.slider(
            "N° inicializaciones", 1, 20, 10,
            help="Se elige la mejor entre varias inicializaciones.",
        )
        show_elbow = st.checkbox("Mostrar método del codo", value=True)
        elbow_max_k = st.slider("K máximo para el codo", 3, 15, 10) if show_elbow else 10

    elif algo_name == "DBSCAN":
        hparams["eps"] = st.slider(
            "Epsilon (ε)", 0.05, 3.0, 0.5, step=0.05,
            help="Radio máximo para considerar dos puntos vecinos.",
        )
        hparams["min_samples"] = st.slider(
            "min_samples", 2, 30, 5,
            help="Mínimo de vecinos para que un punto sea núcleo del cluster.",
        )
        hparams["metric"] = st.selectbox(
            "Métrica de distancia", ["euclidean", "manhattan", "cosine"],
        )
        show_elbow = False

    elif algo_name == "Agglomerative Clustering":
        hparams["n_clusters"] = st.slider("N° clusters", 2, 12, 3)
        hparams["linkage"] = st.selectbox(
            "Criterio de enlace",
            ["ward", "complete", "average", "single"],
            help=(
                "ward: minimiza varianza intra-cluster.\n"
                "complete: maximiza distancia entre clusters.\n"
                "average: promedio de distancias.\n"
                "single: mínima distancia."
            ),
        )
        show_elbow = False

    st.markdown("---")
    fit_btn = st.button("🚀 Ajustar clustering", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_dataset(name: str, **kwargs):
    if name == "Iris (sin etiquetas)":
        bunch = load_iris()
        return bunch.data, list(bunch.feature_names)
    elif name == "make_blobs":
        X, _ = make_blobs(
            n_samples=kwargs["n_samples"],
            centers=kwargs["centers"],
            cluster_std=kwargs["cluster_std"],
            random_state=42,
        )
        return X, [f"x{i+1}" for i in range(X.shape[1])]
    elif name == "make_moons":
        X, _ = make_moons(n_samples=kwargs["n_samples"], noise=kwargs["noise"], random_state=42)
        return X, ["x1", "x2"]
    elif name == "make_circles":
        X, _ = make_circles(n_samples=kwargs["n_samples"], noise=kwargs["noise"],
                             factor=0.5, random_state=42)
        return X, ["x1", "x2"]


def build_model(algo: str, hp: dict, rs: int):
    if algo == "K-Means":
        return KMeans(
            n_clusters=hp["n_clusters"], init=hp["init"],
            max_iter=hp["max_iter"], n_init=hp["n_init"], random_state=rs,
        )
    elif algo == "DBSCAN":
        return DBSCAN(eps=hp["eps"], min_samples=hp["min_samples"], metric=hp["metric"])
    elif algo == "Agglomerative Clustering":
        return AgglomerativeClustering(n_clusters=hp["n_clusters"], linkage=hp["linkage"])


ALGO_DESCRIPTIONS = {
    "K-Means": (
        "**K-Means** divide el espacio en **k** clusters asignando cada punto al centroide "
        "más cercano y recalculando los centroides iterativamente. Minimiza la inercia "
        "(suma de distancias cuadradas a los centroides). Asume clusters esféricos y de "
        "tamaño similar. El principal desafío es elegir **k**; el **método del codo** ayuda: "
        "se busca el punto donde la inercia empieza a disminuir más lentamente."
    ),
    "DBSCAN": (
        "**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) define "
        "clusters como regiones de alta densidad separadas por zonas de baja densidad. "
        "Un punto es **núcleo** si tiene al menos `min_samples` vecinos dentro de radio `ε`. "
        "Puede encontrar clusters de formas arbitrarias y detectar **ruido** (puntos que no "
        "pertenecen a ningún cluster, etiquetados como -1). No requiere especificar k de antemano."
    ),
    "Agglomerative Clustering": (
        "**Clustering Aglomerativo** es un método jerárquico ascendente: comienza con cada "
        "punto como su propio cluster y va fusionando los más similares iterativamente. "
        "El criterio de enlace (*linkage*) determina cómo se mide la similitud entre clusters: "
        "`ward` minimiza la varianza intra-cluster y generalmente da mejores resultados prácticos."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Load dataset
ds_kwargs = {}
if dataset_name == "make_blobs":
    ds_kwargs = {"n_samples": n_samples_gen, "centers": n_centers_gen, "cluster_std": cluster_std_gen}
elif dataset_name in ("make_moons", "make_circles"):
    ds_kwargs = {"n_samples": n_samples_gen, "noise": noise_gen}

X_raw, feature_names = get_dataset(dataset_name, **ds_kwargs)

col1, col2, col3 = st.columns(3)
col1.metric("Muestras", X_raw.shape[0])
col2.metric("Variables originales", X_raw.shape[1])
col3.metric("Algoritmo seleccionado", algo_name)

st.markdown("---")

# ── Unsupervised learning explanation ─────────────────────────────────────────
st.markdown(
    """
    <div class="concept-box">
        <h4>🔍 ¿Qué es el Aprendizaje No Supervisado?</h4>
        <p>
        En el aprendizaje no supervisado no se dispone de etiquetas: el modelo debe
        encontrar estructura por sí solo. El <strong>clustering</strong> agrupa observaciones
        similares. A diferencia del supervisado, no hay una "respuesta correcta" con la que
        comparar; la evaluación se basa en métricas internas como el
        <strong>Silhouette score</strong> (cohesión y separación de clusters) o la
        <strong>inercia</strong> (K-Means).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(f"📖 ¿Cómo funciona {algo_name}?", expanded=False):
    st.markdown(ALGO_DESCRIPTIONS[algo_name])

# ═══════════════════════════════════════════════════════════════════════════════
# FITTING
# ═══════════════════════════════════════════════════════════════════════════════
if fit_btn or "clust_labels" in st.session_state:
    if fit_btn:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_raw)

        model = build_model(algo_name, hparams, rs=42)

        with st.spinner(f"Ajustando {algo_name}..."):
            labels = model.fit_predict(X_sc)

            # PCA to 2D for visualisation
            n_components = min(2, X_sc.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_2d = pca.fit_transform(X_sc)
            if X_sc.shape[1] == 2:
                X_2d = X_sc  # already 2D, skip PCA

        # Elbow for K-Means
        elbow_data = None
        if algo_name == "K-Means" and show_elbow:
            with st.spinner("Calculando método del codo..."):
                inertias = []
                k_range = range(2, elbow_max_k + 1)
                for k in k_range:
                    km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
                    km.fit(X_sc)
                    inertias.append(km.inertia_)
                elbow_data = (list(k_range), inertias)

        st.session_state["clust_labels"]  = labels
        st.session_state["clust_X_2d"]    = X_2d
        st.session_state["clust_X_sc"]    = X_sc
        st.session_state["clust_model"]   = model
        st.session_state["clust_elbow"]   = elbow_data
        st.session_state["clust_meta"]    = {
            "algo": algo_name, "dataset": dataset_name,
            "feature_names": feature_names,
            "pca_var": pca.explained_variance_ratio_ if X_sc.shape[1] > 2 else None,
            "n_components": n_components,
        }
        st.success("✅ Clustering completado.")

    labels  = st.session_state["clust_labels"]
    X_2d    = st.session_state["clust_X_2d"]
    X_sc    = st.session_state["clust_X_sc"]
    model   = st.session_state["clust_model"]
    elbow   = st.session_state["clust_elbow"]
    meta    = st.session_state["clust_meta"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    # Silhouette (needs > 1 cluster and no all-noise)
    valid_mask = labels != -1
    sil = None
    dbi = None
    chi = None
    if n_clusters_found > 1 and valid_mask.sum() > n_clusters_found:
        try:
            sil = silhouette_score(X_sc[valid_mask], labels[valid_mask])
            dbi = davies_bouldin_score(X_sc[valid_mask], labels[valid_mask])
            chi = calinski_harabasz_score(X_sc[valid_mask], labels[valid_mask])
        except Exception:
            pass

    st.markdown("### 📊 Métricas de Evaluación")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Clusters encontrados", n_clusters_found,
               help="Número de grupos distintos identificados (sin contar ruido).")
    mc2.metric(
        "Silhouette Score",
        f"{sil:.4f}" if sil is not None else "N/A",
        help="Entre -1 y 1. Más alto → clusters más compactos y separados.",
    )
    mc3.metric(
        "Davies-Bouldin",
        f"{dbi:.4f}" if dbi is not None else "N/A",
        help="Más bajo → mejor. Ratio compacidad / separación.",
    )
    mc4.metric(
        "Puntos de ruido" if algo_name == "DBSCAN" else "Inercia",
        f"{n_noise}" if algo_name == "DBSCAN"
        else (f"{model.inertia_:.1f}" if hasattr(model, "inertia_") else "N/A"),
        help=(
            "Puntos clasificados como ruido (DBSCAN)."
            if algo_name == "DBSCAN"
            else "Suma de distancias cuadradas a los centroides (K-Means)."
        ),
    )

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_scatter, tab_elbow, tab_stats = st.tabs(
        ["🗺️ Visualización 2D", "📉 Método del Codo", "📋 Estadísticas por Cluster"]
    )

    with tab_scatter:
        pca_var = meta.get("pca_var")
        if pca_var is not None:
            x_label = f"PC1 ({pca_var[0]:.1%} var.)"
            y_label = f"PC2 ({pca_var[1]:.1%} var.)"
            note = (
                f"Proyección PCA: las 2 primeras componentes explican el "
                f"**{pca_var.sum():.1%}** de la varianza total."
            )
        else:
            x_label, y_label = meta["feature_names"][0], meta["feature_names"][1]
            note = "Los datos ya son 2D; no se aplicó PCA."

        st.markdown(f"#### Scatter 2D de Clusters")
        st.markdown(note)

        fig_scatter = plot_clusters_2d(
            X_2d, labels,
            title=f"Clusters — {meta['algo']} · {meta['dataset']}",
            x_label=x_label, y_label=y_label,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        if algo_name == "K-Means" and hasattr(model, "cluster_centers_"):
            pca_for_centers = PCA(n_components=2, random_state=42)
            pca_for_centers.fit(X_sc)
            centers_2d = pca_for_centers.transform(model.cluster_centers_) if X_sc.shape[1] > 2 else model.cluster_centers_
            st.markdown("**Centroides** (marcados con ✕ en el scatter — recarga el gráfico):")
            df_centers = pd.DataFrame(
                centers_2d, columns=["PC1", "PC2"] if X_sc.shape[1] > 2 else meta["feature_names"][:2]
            )
            df_centers.index.name = "Cluster"
            st.dataframe(df_centers.round(4), use_container_width=True)

    with tab_elbow:
        if algo_name == "K-Means":
            if elbow is not None:
                k_vals, inertias = elbow
                st.markdown("#### Método del Codo")
                st.markdown(
                    "Busca el **'codo'** de la curva: el valor de k donde añadir más clusters "
                    "apenas reduce la inercia. Ese suele ser el número óptimo de grupos."
                )
                fig_elbow = plot_elbow_method(inertias, k_vals)
                # Highlight current k
                current_k = hparams.get("n_clusters", 3) if algo_name == "K-Means" else None
                if current_k and current_k in k_vals:
                    idx = k_vals.index(current_k)
                    fig_elbow.add_trace(go.Scatter(
                        x=[current_k], y=[inertias[idx]],
                        mode="markers", marker=dict(size=14, color="#EF553B", symbol="star"),
                        name="K actual",
                    ))
                st.plotly_chart(fig_elbow, use_container_width=True)
            else:
                st.info("Activa **Mostrar método del codo** en el panel lateral y re-ajusta.")
        else:
            st.info("El método del codo sólo aplica a K-Means.")

    with tab_stats:
        st.markdown("#### Estadísticas por Cluster")
        X_orig = X_raw  # unscaled for interpretability
        df_full = pd.DataFrame(X_orig, columns=meta["feature_names"])
        df_full["cluster"] = labels

        # Replace -1 with "Ruido"
        df_full["cluster"] = df_full["cluster"].replace(-1, -999)
        cluster_labels_display = {-999: "Ruido"}

        summary_rows = []
        for cl in sorted(df_full["cluster"].unique()):
            mask = df_full["cluster"] == cl
            name = "Ruido" if cl == -999 else f"Cluster {cl}"
            row = {"Cluster": name, "N muestras": mask.sum(),
                   "% del total": f"{mask.mean():.1%}"}
            for feat in meta["feature_names"]:
                row[f"{feat} (media)"] = df_full.loc[mask, feat].mean()
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary.round(4), use_container_width=True, hide_index=True)

        # Bar chart: mean per feature per cluster (excluding noise)
        st.markdown("#### Media de variables por cluster")
        df_means = (
            df_full[df_full["cluster"] != -999]
            .groupby("cluster")[meta["feature_names"]]
            .mean()
            .reset_index()
        )
        df_means["cluster"] = df_means["cluster"].astype(str)
        df_melt = df_means.melt(id_vars="cluster", var_name="Variable", value_name="Media")

        if len(meta["feature_names"]) <= 10:
            fig_bar = px.bar(
                df_melt, x="Variable", y="Media", color="cluster",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                title="Media de Variables por Cluster",
            )
            fig_bar.update_layout(height=400, xaxis_tickangle=-30,
                                  plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                                  font_color="#e2e8f0")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.markdown("*(Demasiadas variables para el gráfico de barras; consulta la tabla superior.)*")

        # Silhouette details
        if sil is not None:
            st.markdown("---")
            st.markdown(
                f"**Silhouette Score global: `{sil:.4f}`** "
                f"(rango: -1 peor → 1 mejor)\n\n"
                f"**Davies-Bouldin Index: `{dbi:.4f}`** (menor es mejor)\n\n"
                f"**Calinski-Harabasz Score: `{chi:.1f}`** (mayor es mejor)"
            )

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem; color: #64748b;">
            <div style="font-size: 3rem;">⬅️</div>
            <p style="font-size: 1.1rem;">Selecciona dataset, algoritmo e hiperparámetros en el panel lateral,
            luego pulsa <strong>Ajustar clustering</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
