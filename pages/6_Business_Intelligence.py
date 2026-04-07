"""
pages/6_Business_Intelligence.py
================================
Módulo: Business Intelligence / Dashboard Gerencial
Visualizaciones ejecutivas de indicadores mundiales con análisis regional,
tendencias temporales, predicciones ML en vivo y exportación de datos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

from utils.worldbank import get_indicator, get_indicator_timeseries, INDICATORS
from utils.plots import plot_choropleth
from utils.model_export import load_pipeline
from utils.styles import inject_base_css

st.set_page_config(page_title="Business Intelligence · ML Explorer", layout="wide")

inject_base_css()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    ">
        <h1 style="color:#fff; margin:0; font-size:2.2rem;">📊 Business Intelligence</h1>
        <p style="color:#a0aec0; margin-top:0.5rem;">
            Dashboard ejecutivo de indicadores globales · Análisis regional · Predicciones ML
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: Configuration ────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

bi_year = st.sidebar.slider("Año de análisis", 2000, 2023, 2022)

# Region mapping: país → región
REGION_MAP = {
    "Argentina": "América Latina", "Bolivia": "América Latina", "Brazil": "América Latina",
    "Chile": "América Latina", "Colombia": "América Latina", "Costa Rica": "América Latina",
    "Ecuador": "América Latina", "El Salvador": "América Latina", "Guatemala": "América Latina",
    "Honduras": "América Latina", "Mexico": "América Latina", "Nicaragua": "América Latina",
    "Panama": "América Latina", "Paraguay": "América Latina", "Peru": "América Latina",
    "Uruguay": "América Latina", "Venezuela": "América Latina",
    "Canada": "América del Norte", "United States": "América del Norte",
    "China": "Asia", "India": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "Indonesia": "Asia", "Vietnam": "Asia", "Thailand": "Asia", "Philippines": "Asia",
    "Malaysia": "Asia", "Singapore": "Asia", "Hong Kong": "Asia", "Pakistan": "Asia",
    "Germany": "Europa", "France": "Europa", "United Kingdom": "Europa", "Italy": "Europa",
    "Spain": "Europa", "Poland": "Europa", "Netherlands": "Europa", "Belgium": "Europa",
    "Sweden": "Europa", "Switzerland": "Europa", "Austria": "Europa", "Denmark": "Europa",
    "Finland": "Europa", "Greece": "Europa", "Portugal": "Europa", "Czech Republic": "Europa",
    "Nigeria": "África", "Egypt": "África", "South Africa": "África", "Kenya": "África",
    "Morocco": "África", "Tunisia": "África", "Ghana": "África", "Ethiopia": "África",
    "Australia": "Oceanía", "New Zealand": "Oceanía",
}

regions = ["Todas"] + sorted(set(REGION_MAP.values()))
selected_region = st.sidebar.selectbox("Región", regions, key="bi_region")

indicator_name = st.sidebar.selectbox("Indicador principal", list(INDICATORS.keys()), index=1)
indicator_code = INDICATORS[indicator_name]

# ── Load main data ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Cargando datos…")
def load_bi_data(indicator_code: str, year: int, region_filter: str) -> pd.DataFrame:
    df = get_indicator(indicator_code, year)
    if df.empty:
        return pd.DataFrame()

    # Add region column
    df["region"] = df["country"].map(REGION_MAP).fillna("Otro")

    # Filter by region if selected
    if region_filter != "Todas":
        df = df[df["region"] == region_filter]

    return df

df_bi = load_bi_data(indicator_code, bi_year, selected_region)

if df_bi.empty:
    st.error("⚠️ No hay datos disponibles para esta configuración.")
    st.stop()

# ── KPI Row ─────────────────────────────────────────────────────────────────
st.markdown("### 📈 Indicadores Clave")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

n_countries = len(df_bi)
total_value = df_bi["value"].sum()
avg_value = df_bi["value"].mean()
max_value = df_bi["value"].max()

kpi1.metric("Países analizados", n_countries)
kpi2.metric("Promedio global", f"{avg_value:,.2f}")
kpi3.metric("Máximo", f"{max_value:,.2f}")
kpi4.metric("Año", bi_year)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Panorama Global", "🏆 Top Países", "📈 Tendencias", "🌐 Análisis Regional", "🤖 Predicciones"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Panorama Global (Choropleth)
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Mapa Global — {indicator_name} ({bi_year})")

    scale = st.selectbox(
        "Escala de color",
        ["Viridis", "Plasma", "Blues", "RdYlGn", "YlOrRd"],
        key="bi_scale",
    )

    fig_map = plot_choropleth(
        df_bi,
        title=f"{indicator_name} ({bi_year})",
        color_scale=scale,
        label=indicator_name,
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Top 10 Países
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Top 10 Países — {indicator_name}")

    top10 = df_bi.nlargest(10, "value").sort_values("value")

    fig_top = px.bar(
        top10,
        x="value", y="country",
        orientation="h",
        color="value",
        color_continuous_scale="Greens",
        labels={"value": indicator_name, "country": "País"},
    )
    fig_top.update_layout(
        height=450,
        coloraxis_showscale=False,
        margin=dict(l=130, r=20, t=40, b=40),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # Data table
    with st.expander("📋 Ver tabla completa"):
        st.dataframe(
            df_bi.sort_values("value", ascending=False)[["country", "region", "value"]]
            .rename(columns={"value": indicator_name})
            .reset_index(drop=True),
            use_container_width=True,
            height=350,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tendencias Temporales
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Evolución Temporal — {indicator_name}")

    ts_start, ts_end = st.slider(
        "Selecciona rango de años",
        min_value=2000, max_value=2023,
        value=(2010, 2023),
        key="bi_ts_range",
    )

    df_ts = get_indicator_timeseries(indicator_code, ts_start, ts_end)

    if df_ts.empty:
        st.warning("No hay datos de serie temporal para este indicador.")
    else:
        # Select countries
        all_countries = sorted(df_ts["country"].unique())
        default_candidates = [
            "United States", "China", "India", "Germany", "Japan",
            "Brazil", "Mexico", "Argentina", "Chile",
        ]
        defaults = [c for c in default_candidates if c in all_countries][:5]

        selected_countries = st.multiselect(
            "Selecciona países para comparar",
            all_countries,
            default=defaults,
            key="bi_ts_countries",
        )

        if selected_countries:
            df_ts_filt = df_ts[df_ts["country"].isin(selected_countries)]

            fig_ts = px.line(
                df_ts_filt,
                x="year", y="value",
                color="country",
                markers=True,
                title=f"{indicator_name} — Evolución {ts_start}–{ts_end}",
                labels={"value": indicator_name, "year": "Año", "country": "País"},
            )
            fig_ts.update_layout(
                height=490,
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font_color="#e2e8f0",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=60, r=20, t=60, b=60),
            )
            fig_ts.update_xaxes(gridcolor="#334155", dtick=2)
            fig_ts.update_yaxes(gridcolor="#334155")
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Selecciona al menos un país.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Análisis Regional
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Análisis por Región — {indicator_name} ({bi_year})")

    if selected_region == "Todas":
        # Show all regions
        df_regional = df_bi.groupby("region").agg({
            "value": ["mean", "count", "max", "min"]
        }).round(2)
        df_regional.columns = ["Promedio", "Países", "Máximo", "Mínimo"]
        df_regional = df_regional.sort_values("Promedio", ascending=False)

        st.markdown("**Métricas por región:**")
        st.dataframe(df_regional, use_container_width=True)

        # Regional comparison chart
        df_reg_summary = df_bi.groupby("region")["value"].mean().reset_index()
        df_reg_summary = df_reg_summary.sort_values("value", ascending=False)

        fig_reg = px.bar(
            df_reg_summary,
            x="value", y="region",
            orientation="h",
            color="value",
            color_continuous_scale="Blues",
            labels={"value": f"Promedio {indicator_name}", "region": "Región"},
        )
        fig_reg.update_layout(
            height=400,
            coloraxis_showscale=False,
            margin=dict(l=120, r=20, t=40, b=40),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        # Show region details
        st.info(f"📍 Región: **{selected_region}** ({len(df_bi)} países)")

        col_reg1, col_reg2, col_reg3 = st.columns(3)
        col_reg1.metric("Promedio regional", f"{df_bi['value'].mean():,.2f}")
        col_reg2.metric("País con mayor valor", df_bi.loc[df_bi["value"].idxmax(), "country"])
        col_reg3.metric("País con menor valor", df_bi.loc[df_bi["value"].idxmin(), "country"])

        st.markdown("**Países en esta región:**")
        st.dataframe(
            df_bi[["country", "value"]]
            .rename(columns={"value": indicator_name})
            .sort_values(indicator_name, ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Predicciones con Modelos ML
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 Predicciones en Vivo")
    st.markdown(
        "Carga un modelo .joblib exportado desde la página 'ML en Datos Reales' "
        "e ingresa valores de features para obtener predicciones."
    )

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_model = st.file_uploader(
            "📂 Sube tu modelo (.joblib)",
            type=["joblib"],
            key="bi_model_upload",
        )

    if uploaded_model is not None:
        try:
            # Load model from bytes
            from joblib import load
            import io
            pipeline = load(io.BytesIO(uploaded_model.getvalue()))

            st.success("✅ Modelo cargado exitosamente")

            # Get model info
            model = pipeline.named_steps['model']
            model_type = type(model).__name__

            st.info(f"**Tipo de modelo:** {model_type}")

            # Input features
            st.markdown("**Ingresa valores de features:**")

            # Number of features
            try:
                n_features = model.n_features_in_
            except:
                n_features = 2  # Default fallback

            features_input = []
            cols = st.columns(n_features)
            for i in range(n_features):
                with cols[i % len(cols)]:
                    val = st.number_input(
                        f"Feature {i+1}",
                        value=0.0,
                        step=0.1,
                        key=f"bi_feature_{i}",
                    )
                    features_input.append(val)

            # Prediction button
            if st.button("🔮 Generar predicción", use_container_width=True):
                try:
                    X_new = np.array([features_input]).reshape(1, -1)
                    prediction = pipeline.predict(X_new)[0]

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Display prediction in a nice box
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.metric("Predicción", f"{prediction:,.2f}")
                    with col_pred2:
                        st.metric("Confianza", "Alta" if hasattr(model, 'score') else "Media")

                    st.success(f"✅ Predicción generada correctamente")
                except Exception as e:
                    st.error(f"⚠️ Error en predicción: {str(e)}")

        except Exception as e:
            st.error(f"⚠️ Error al cargar el modelo: {str(e)}")
            st.info("Asegúrate de que el archivo es un .joblib válido exportado desde ML Explorer")

    else:
        st.info(
            "📌 **Cómo obtener un modelo:**\n\n"
            "1. Ve a la página **ML en Datos Reales**\n"
            "2. Entrena un modelo seleccionando features y target\n"
            "3. Haz click en **'⬇️ Descargar modelo.joblib'**\n"
            "4. Vuelve aquí y sube el archivo"
        )


# ── Export Section ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📥 Exportar Datos")

# Convert dataframe to CSV
csv_buffer = df_bi[["country", "region", "value"]].rename(
    columns={"value": indicator_name}
).sort_values(indicator_name, ascending=False).to_csv(index=False)

st.download_button(
    label=f"⬇️ Descargar datos (.csv)",
    data=csv_buffer,
    file_name=f"bi_{indicator_name.lower()}_{bi_year}.csv",
    mime="text/csv",
    use_container_width=True,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.85rem;">
        Datos: <a href="https://data.worldbank.org" style="color:#636EFA;">Banco Mundial</a>
        · Visualización: Plotly · Sprint 4 — Business Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
