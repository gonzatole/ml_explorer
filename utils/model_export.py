"""
utils/model_export.py
====================
Helpers para serializar y exportar modelos ML entrenados como Pipelines sklearn.
Permite reutilizar modelos en otros proyectos (FastAPI, Jupyter, Next.js via API, etc).
"""

from io import BytesIO
from datetime import datetime
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(scaler: StandardScaler, model) -> Pipeline:
    """
    Construye un sklearn Pipeline con StandardScaler + modelo entrenado.

    Args:
        scaler: StandardScaler ajustado a los datos de entrenamiento
        model: Modelo entrenado (LinearRegression, RandomForest, SVC, etc)

    Returns:
        sklearn.pipeline.Pipeline con pasos ['scaler', 'model']

    Uso:
        >>> pipeline = build_pipeline(scaler, model)
        >>> X_new = [[1.5, 2.3, 4.1]]
        >>> predictions = pipeline.predict(X_new)  # Escala automáticamente
    """
    return Pipeline([
        ('scaler', scaler),
        ('model', model),
    ])


def export_pipeline(pipeline: Pipeline, name: str) -> bytes:
    """
    Serializa un Pipeline a bytes usando joblib (útil para st.download_button).

    Args:
        pipeline: sklearn.pipeline.Pipeline entrenado
        name: Nombre descriptivo para el archivo (ej: 'random_forest_gdp')

    Returns:
        Bytes del archivo .joblib (serializado)

    Uso en Streamlit:
        >>> pipeline_bytes = export_pipeline(pipeline, 'mi_modelo')
        >>> st.download_button(
        ...     label='⬇️ Descargar modelo',
        ...     data=pipeline_bytes,
        ...     file_name=f'{name}.joblib',
        ...     mime='application/octet-stream'
        ... )
    """
    # Serializar a BytesIO
    buffer = BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def load_pipeline(path: str) -> Pipeline:
    """
    Carga un Pipeline .joblib desde disco.

    Args:
        path: Ruta del archivo .joblib (ej: 'models/random_forest_gdp.joblib')

    Returns:
        sklearn.pipeline.Pipeline listo para predicciones

    Uso:
        >>> pipeline = load_pipeline('models/mi_modelo.joblib')
        >>> pred = pipeline.predict([[5.2, 3.1, 1.9]])
    """
    return joblib.load(path)


def get_model_metadata(pipeline: Pipeline, feature_names: list[str], target_name: str) -> dict:
    """
    Extrae metadata útil de un pipeline entrenado.

    Args:
        pipeline: sklearn.pipeline.Pipeline
        feature_names: Lista de nombres de features usados en entrenamiento
        target_name: Nombre de la variable objetivo

    Returns:
        Dict con metadata: timestamp, model_type, features, target, etc.
    """
    model = pipeline.named_steps['model']

    metadata = {
        'exported_at': datetime.now().isoformat(),
        'model_type': type(model).__name__,
        'feature_names': feature_names,
        'target_name': target_name,
        'n_features': len(feature_names),
    }

    # Agregar info extra si el modelo lo tiene
    if hasattr(model, 'feature_importances_'):
        metadata['has_feature_importances'] = True
    if hasattr(model, 'coef_'):
        metadata['has_coefficients'] = True

    return metadata
