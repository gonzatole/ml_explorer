"""
utils/styles.py
===============
CSS base compartido para todos los módulos de ML Explorer.
Evita duplicar la hoja de estilos en cada página.
"""

import streamlit as st

_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: #0f172a; }

[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

/* Estructura base del encabezado de página — el gradiente lo define cada página */
.page-header {
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.page-header h1 { color: #fff; margin: 0; font-size: 2.2rem; }
.page-header p  { color: #94a3b8; margin: 0.4rem 0 0; }
</style>
"""


def inject_base_css() -> None:
    """Inyecta el CSS global de ML Explorer. Llamar una vez al inicio de cada página."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
