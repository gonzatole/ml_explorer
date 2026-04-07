"""
pages/4_Datos_Mundiales.py
==========================
Módulo: Datos Reales del Mundo
Conecta a la API gratuita del Banco Mundial y visualiza indicadores
con mapas coropléticos, rankings, series temporales y comparadores.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.worldbank import get_indicator, get_indicator_timeseries, INDICATORS
from utils.plots import plot_choropleth
from utils.styles import inject_base_css

st.set_page_config(page_title="Datos Mundiales · ML Explorer", layout="wide")

inject_base_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🌍 Configuración")

indicator_name = st.sidebar.selectbox(
    "Indicador", list(INDICATORS.keys()), index=1
)
indicator_code = INDICATORS[indicator_name]

year = st.sidebar.slider("Año (mapa / ranking / comparador)", 2000, 2023, 2022)

start_year, end_year = st.sidebar.slider(
    "Rango de años (serie temporal)",
    min_value=2000, max_value=2023,
    value=(2010, 2022),
)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    ">
        <h1 style="color:#fff; margin:0; font-size:2.2rem;">🌍 Datos Reales del Mundo</h1>
        <p style="color:#a0aec0; margin-top:0.5rem;">
            Indicadores del Banco Mundial · API gratuita · Sin API key requerida
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load point-in-time data ────────────────────────────────────────────────────
df = get_indicator(indicator_code, year)

if df.empty:
    st.error(
        "⚠️ No se pudo cargar el indicador. "
        "Verifica tu conexión a internet o intenta con otro indicador/año."
    )
    st.stop()

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Países con datos", len(df))
k2.metric(
    "Máximo",
    f"{df['value'].max():,.2f}",
    df.loc[df["value"].idxmax(), "country"],
)
k3.metric(
    "Mínimo",
    f"{df['value'].min():,.2f}",
    df.loc[df["value"].idxmin(), "country"],
)
k4.metric("Promedio global", f"{df['value'].mean():,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Mapa", "🏆 Ranking", "📈 Evolución temporal", "⚖️ Comparador"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Choropleth map
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"{indicator_name} — {year}")

    scale = st.selectbox(
        "Escala de color",
        ["Viridis", "Plasma", "RdYlGn", "Blues", "Reds", "YlOrRd", "Turbo"],
        key="choro_scale",
    )
    fig_map = plot_choropleth(
        df,
        title=f"{indicator_name} ({year})",
        color_scale=scale,
        label=indicator_name,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    with st.expander("Ver tabla de datos"):
        st.dataframe(
            df[["country", "iso3", "value"]]
            .rename(columns={"value": indicator_name})
            .sort_values(indicator_name, ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=350,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Ranking Top/Bottom 20
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Top y Bottom 20 — {indicator_name} ({year})")

    top20 = df.nlargest(20, "value").sort_values("value")
    bot20 = df.nsmallest(20, "value").sort_values("value", ascending=False)

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("**🔼 Top 20 — Mayor valor**")
        fig_top = px.bar(
            top20,
            x="value", y="country",
            orientation="h",
            color="value",
            color_continuous_scale="Greens",
            labels={"value": indicator_name, "country": "País"},
        )
        fig_top.update_layout(
            height=560,
            coloraxis_showscale=False,
            margin=dict(l=130, r=20, t=30, b=40),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        fig_top.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig_top, use_container_width=True)

    with rc2:
        st.markdown("**🔽 Bottom 20 — Menor valor**")
        fig_bot = px.bar(
            bot20,
            x="value", y="country",
            orientation="h",
            color="value",
            color_continuous_scale="Reds_r",
            labels={"value": indicator_name, "country": "País"},
        )
        fig_bot.update_layout(
            height=560,
            coloraxis_showscale=False,
            margin=dict(l=130, r=20, t=30, b=40),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        fig_bot.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig_bot, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Serie temporal
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Evolución temporal — {indicator_name} ({start_year}–{end_year})")

    df_ts = get_indicator_timeseries(indicator_code, start_year, end_year)

    if df_ts.empty:
        st.warning("No hay datos de serie temporal disponibles para este indicador.")
    else:
        all_countries = sorted(df_ts["country"].unique())
        # Suggest a handful of recognisable countries as default selection
        default_candidates = [
            "Chile", "Argentina", "Brazil", "Colombia", "Mexico",
            "United States", "China", "Germany", "Japan", "India",
        ]
        defaults = [c for c in default_candidates if c in all_countries][:5]

        selected = st.multiselect(
            "Seleccionar países",
            all_countries,
            default=defaults,
            key="ts_countries",
        )

        if not selected:
            st.info("Selecciona al menos un país para visualizar la evolución.")
        else:
            df_filt = df_ts[df_ts["country"].isin(selected)]
            fig_ts = px.line(
                df_filt,
                x="year", y="value",
                color="country",
                markers=True,
                title=f"{indicator_name} — {start_year} a {end_year}",
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


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Comparador de indicadores (scatter)
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Comparador de indicadores — {year}")
    st.markdown("Cruza dos indicadores para descubrir correlaciones entre países.")

    comp1, comp2 = st.columns(2)
    with comp1:
        ind_x_name = st.selectbox(
            "Indicador X (horizontal)", list(INDICATORS.keys()), index=1, key="cx"
        )
    with comp2:
        ind_y_name = st.selectbox(
            "Indicador Y (vertical)", list(INDICATORS.keys()), index=2, key="cy"
        )

    df_x = get_indicator(INDICATORS[ind_x_name], year)
    df_y = get_indicator(INDICATORS[ind_y_name], year)

    if df_x.empty or df_y.empty:
        st.warning("No se pudieron cargar datos para uno o ambos indicadores.")
    else:
        df_merged = (
            df_x.merge(df_y, on=["iso3", "country"], suffixes=("_x", "_y"))
            .dropna(subset=["value_x", "value_y"])
            .reset_index(drop=True)
        )

        if df_merged.empty:
            st.warning("Sin países con datos disponibles para ambos indicadores en este año.")
        else:
            opt1, opt2 = st.columns(2)
            log_x = opt1.checkbox("Escala log en X", value=(ind_x_name == "Población total"))
            log_y = opt2.checkbox("Escala log en Y", value=False)

            # Manual trendline via numpy (statsmodels not installed)
            vx = np.log10(df_merged["value_x"]) if log_x else df_merged["value_x"].values
            vy = np.log10(df_merged["value_y"]) if log_y else df_merged["value_y"].values
            m, b = np.polyfit(vx, vy, 1)
            x_line = np.linspace(vx.min(), vx.max(), 100)
            y_line = m * x_line + b
            if log_x:
                x_line = 10 ** x_line
            if log_y:
                y_line = 10 ** y_line

            fig_sc = go.Figure()
            # Points
            fig_sc.add_trace(go.Scatter(
                x=df_merged["value_x"],
                y=df_merged["value_y"],
                mode="markers",
                text=df_merged["country"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{ind_x_name}: %{{x:,.2f}}<br>"
                    f"{ind_y_name}: %{{y:,.2f}}<extra></extra>"
                ),
                marker=dict(
                    size=8, opacity=0.78, color="#636EFA",
                    line=dict(width=0.5, color="white"),
                ),
                name="Países",
            ))
            # Trendline
            fig_sc.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color="#EF553B", dash="dash", width=1.8),
                name="Tendencia (OLS)",
                hoverinfo="skip",
            ))
            fig_sc.update_layout(
                title=f"{ind_x_name} vs. {ind_y_name} ({year})",
                xaxis=dict(
                    title=ind_x_name,
                    type="log" if log_x else "linear",
                    gridcolor="#334155",
                ),
                yaxis=dict(
                    title=ind_y_name,
                    type="log" if log_y else "linear",
                    gridcolor="#334155",
                ),
                height=530,
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font_color="#e2e8f0",
                margin=dict(l=60, r=20, t=60, b=60),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption(f"Países con datos para ambos indicadores: **{len(df_merged)}**")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.85rem;">
        Fuente: <a href="https://data.worldbank.org" style="color:#636EFA;">Banco Mundial Open Data</a>
        · Datos disponibles bajo licencia CC BY 4.0
    </div>
    """,
    unsafe_allow_html=True,
)
