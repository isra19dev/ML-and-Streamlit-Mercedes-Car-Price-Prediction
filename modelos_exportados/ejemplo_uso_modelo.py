"""
SCRIPT DE EJEMPLO: Cómo cargar y usar el modelo exportado
"""

import joblib
import pandas as pd
import json

# ============================================================================
# PASO 1: Cargar el modelo completo (opción más simple)
# ============================================================================

print("Opción 1: Usar el modelo COMPLETO (pipeline incluido)")
print("=" * 60)

# Cargar el modelo entrenado
modelo = joblib.load('modelos_exportados/modelo_final_*.joblib')

# Hacer una predicción
datos_nuevos = pd.DataFrame({
    'year': [2020],
    'mileage': [50000],
    'engineSize': [2.0],
    'transmission': ['Automatic'],
    'fuelType': ['Petrol'],
    'brand': ['BMW'],
    'model': ['Series 5']
})

prediccion = modelo.predict(datos_nuevos)
print(f"Precio predicho: ${prediccion[0]:,.2f}")

# ============================================================================
# PASO 2: Cargar componentes por separado (opción más flexible)
# ============================================================================

print("\nOpción 2: Cargar preprocessor y modelo por separado")
print("=" * 60)

# Cargar componentes
preprocessor = joblib.load('modelos_exportados/preprocessor.joblib')
modelo_ml = joblib.load('modelos_exportados/modelo_ml_*.joblib')

# Transformar datos
X_procesados = preprocessor.transform(datos_nuevos)

# Hacer predicción
prediccion = modelo_ml.predict(X_procesados)
print(f"Precio predicho: ${prediccion[0]:,.2f}")

# ============================================================================
# PASO 3: Acceder a metadatos del modelo
# ============================================================================

print("\nPaso 3: Acceder a información del modelo")
print("=" * 60)

# Cargar metadatos
with open('modelos_exportados/metadatos_modelo.json', 'r') as f:
    metadatos = json.load(f)

print(f"Nombre del modelo: {metadatos['nombre_modelo']}")
print(f"R² Score: {metadatos['metricas']['r2_score']:.4f}")
print(f"RMSE: ${metadatos['metricas']['rmse']:,.2f}")

# Cargar mapeo de categorías
with open('modelos_exportados/categorias_mapping.json', 'r') as f:
    categorias = json.load(f)

print(f"\nCategorías disponibles en 'transmission': {categorias['transmission']['clases']}")
