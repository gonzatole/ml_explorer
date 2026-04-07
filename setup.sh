#!/usr/bin/env bash
# =============================================================================
# setup.sh — ML Explorer environment bootstrap
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "============================================"
echo "  ML Explorer — Setup del entorno Python"
echo "============================================"
echo ""

# ── 1. Create virtual environment ─────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] El entorno virtual ya existe en .venv/"
else
    echo "[INFO] Creando entorno virtual en .venv/ ..."
    python -m venv "$VENV_DIR"
    echo "[OK]   Entorno virtual creado."
fi

# ── 2. Activate ───────────────────────────────────────────────────────────────
echo "[INFO] Activando entorno virtual ..."
# shellcheck disable=SC1091
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32" ]]; then
    # Git-Bash / Windows
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi
echo "[OK]   Entorno activado: $(python --version)"

# ── 3. Upgrade pip silently ───────────────────────────────────────────────────
echo "[INFO] Actualizando pip ..."
python -m pip install --upgrade pip --quiet
echo "[OK]   pip actualizado."

# ── 4. Install requirements ───────────────────────────────────────────────────
echo "[INFO] Instalando dependencias desde requirements.txt ..."
pip install -r "$PROJECT_DIR/requirements.txt"
echo "[OK]   Dependencias instaladas."

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Instalacion completada con exito!"
echo "============================================"
echo ""
echo "Para ejecutar la aplicacion:"
echo ""
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32" ]]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi
echo "  streamlit run app.py"
echo ""
echo "La app se abrira en http://localhost:8501"
echo ""
