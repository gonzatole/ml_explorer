# Artefactos ML — Modelos Entrenados

Carpeta de almacenamiento para modelos ML reutilizables exportados desde ML Explorer.

## 📦 Qué encontrarás aquí

Archivos `.joblib` que contienen:
- **sklearn.Pipeline** con StandardScaler + modelo entrenado
- Información de features y target variable
- Historial de entrenamiento (metadata)

## 🚀 Cómo usar un modelo exportado

### Desde Python / Jupyter
```python
from joblib import load

# Cargar el pipeline
pipeline = load('models/random_forest_gdp.joblib')

# El pipeline incluye escalado automático
# No necesitas escalar manualmente
X_new = [[1.5, 2.3, 4.1]]  # Valores en escala original
prediccion = pipeline.predict(X_new)
print(prediccion)  # [output]
```

### Desde FastAPI
```python
from fastapi import FastAPI
from joblib import load

app = FastAPI()
pipeline = load('models/random_forest_gdp.joblib')

@app.post('/predict')
def predict(features: list[float]):
    pred = pipeline.predict([features])
    return {'prediccion': pred[0]}
```

### Desde Next.js / Frontend
```javascript
// Llamar a un endpoint FastAPI que cargó el modelo
const response = await fetch('/api/predict', {
  method: 'POST',
  body: JSON.stringify({ features: [1.5, 2.3, 4.1] })
});
const { prediccion } = await response.json();
```

## 🔍 Estructura de un .joblib

```python
from joblib import load
pipeline = load('models/modelo.joblib')

# El pipeline tiene dos pasos:
# 1. 'scaler' → StandardScaler (normalización)
# 2. 'model' → Modelo entrenado (LinearRegression, RandomForest, SVC, etc)

# Acceder a componentes:
scaler = pipeline.named_steps['scaler']
modelo = pipeline.named_steps['model']

# Feature importance (si RandomForest, GradientBoosting, etc)
if hasattr(modelo, 'feature_importances_'):
    importances = modelo.feature_importances_
    # [...ranking de features...]
```

## ✅ Checklist al exportar un modelo

- ✓ Features y target están debidamente documentados
- ✓ El Pipeline incluye StandardScaler (no data leakage)
- ✓ Se validó en test set antes de exportar
- ✓ Nombre del archivo describe el modelo (ej: `rf_vida_por_pib_gini.joblib`)

## 📝 Ejemplo de metadata

Cada modelo exportado desde ML Explorer incluye metadata útil:

```python
{
  'exported_at': '2026-03-24T14:35:22.123456',
  'model_type': 'RandomForestRegressor',
  'feature_names': ['PIB per cápita', 'Esperanza de vida'],
  'target_name': 'GINI (desigualdad)',
  'n_features': 2,
  'has_feature_importances': True
}
```

---

**Fuente:** Modelos entrenados en ML Explorer (ml_explorer/pages/5_ML_Datos_Reales.py)
