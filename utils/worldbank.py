"""
utils/worldbank.py
==================
Conector a la API gratuita del Banco Mundial.
Sin API key requerida. Datos actualizados ~anualmente.

Referencia: https://datahelpdesk.worldbank.org/knowledgebase/articles/889386
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://api.worldbank.org/v2"

INDICATORS: dict[str, str] = {
    "Población total":              "SP.POP.TOTL",
    "PIB per cápita (USD)":         "NY.GDP.PCAP.CD",
    "Esperanza de vida (años)":     "SP.DYN.LE00.IN",
    "Desigualdad GINI":             "SI.POV.GINI",
    "Inflación (%)":                "FP.CPI.TOTL.ZG",
    "Desempleo (%)":                "SL.UEM.TOTL.ZS",
    "CO₂ per cápita (ton)":         "EN.ATM.CO2E.PC",
    "Gasto en educación (% PIB)":   "SE.XPD.TOTL.GD.ZS",
    "Acceso a electricidad (%)":    "EG.ELC.ACCS.ZS",
}


@st.cache_data(ttl=3600, show_spinner="Descargando datos del Banco Mundial…")
def get_indicator(indicator_code: str, year: int = 2022) -> pd.DataFrame:
    """
    Descarga un indicador para todos los países en un año dado.

    Returns:
        DataFrame con columnas: iso3, country, year, value
        Solo incluye filas con value no-nulo e iso3 de 3 letras
        (excluye agregados regionales del Banco Mundial).
    """
    url = (
        f"{BASE_URL}/country/all/indicator/{indicator_code}"
        f"?format=json&per_page=500&date={year}&mrv=1"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame(columns=["iso3", "country", "year", "value"])

    if len(data) < 2 or not data[1]:
        return pd.DataFrame(columns=["iso3", "country", "year", "value"])

    rows = []
    for item in data[1]:
        iso3 = item.get("countryiso3code", "")
        value = item.get("value")
        if len(iso3) == 3 and value is not None:
            rows.append({
                "iso3":    iso3,
                "country": item["country"]["value"],
                "year":    int(item["date"]),
                "value":   float(value),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["iso3", "country", "year", "value"])


@st.cache_data(ttl=3600, show_spinner="Descargando serie temporal…")
def get_indicator_timeseries(
    indicator_code: str,
    start_year: int = 2000,
    end_year: int = 2022,
) -> pd.DataFrame:
    """
    Descarga un indicador para todos los países en un rango de años.

    Returns:
        DataFrame con columnas: iso3, country, year, value
    """
    url = (
        f"{BASE_URL}/country/all/indicator/{indicator_code}"
        f"?format=json&per_page=1000&date={start_year}:{end_year}"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame(columns=["iso3", "country", "year", "value"])

    if len(data) < 2 or not data[1]:
        return pd.DataFrame(columns=["iso3", "country", "year", "value"])

    # Handle pagination
    total_pages = data[0].get("pages", 1)
    all_records = list(data[1])

    for page in range(2, total_pages + 1):
        try:
            r2 = requests.get(url + f"&page={page}", timeout=20)
            r2.raise_for_status()
            d2 = r2.json()
            if len(d2) >= 2 and d2[1]:
                all_records.extend(d2[1])
        except Exception:
            break

    rows = []
    for item in all_records:
        iso3 = item.get("countryiso3code", "")
        value = item.get("value")
        if len(iso3) == 3 and value is not None:
            rows.append({
                "iso3":    iso3,
                "country": item["country"]["value"],
                "year":    int(item["date"]),
                "value":   float(value),
            })

    if not rows:
        return pd.DataFrame(columns=["iso3", "country", "year", "value"])

    return pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)
