# Guía: Exportación y Uso del Modelo de Predicción de Precios

## 📋 Resumen

La exportación del modelo se realiza en **2 PARTES**:

### PARTE 4: Optimización de Hiperparámetros
- GridSearchCV busca los mejores parámetros para el modelo seleccionado
- Valida con 5-fold cross-validation
- Calcula métricas en el conjunto de test

### PARTE 5: Exportación de Archivos
Se guardan múltiples archivos para máxima flexibilidad:

---

## 🗂️ Archivos Generados

### 1. **modelo_final_*.joblib** (RECOMENDADO)
```
- Contiene: Pipeline COMPLETO (preprocesador + modelo)
- Usar cuando: Quieras hacer predicciones directamente
- Ventaja: Una sola línea de código para predicciones
- Formato: joblib (más eficiente que pickle)
```

**Ejemplo de uso:**
```python
import joblib

# Cargar
modelo = joblib.load('modelos_exportados/modelo_final_random_forest_*.joblib')

# Predicción
prediccion = modelo.predict(datos_nuevos)
```

### 2. **modelo_final_*.pkl** (COMPATIBILIDAD)
```
- Contiene: Pipeline COMPLETO (preprocesador + modelo)
- Formato: pickle (compatible con pickle estándar)
- Usar cuando: joblib no esté disponible
```

---

## 🔧 Archivos de Componentes Separados

### 3. **preprocessor.joblib**
```
- Contiene: SOLO los transformadores (StandardScaler + OneHotEncoder)
- Usar cuando: Quieras separar transformación de predicción
- Incluye: Escaladores y encoders entrenados
```

**Uso con predicción separada:**
```python
import joblib

# Cargar componentes
preprocessor = joblib.load('modelos_exportados/preprocessor.joblib')
modelo_ml = joblib.load('modelos_exportados/modelo_ml_random_forest.joblib')

# Transformar datos nuevos
X_procesados = preprocessor.transform(datos_nuevos)

# Predicción
prediccion = modelo_ml.predict(X_procesados)
```

### 4. **modelo_ml_*.joblib**
```
- Contiene: SOLO el modelo ML (sin preprocessor)
- Usar cuando: Ya tengas datos preprocesados
- Necesita: Datos escalados y categóricamente codificados
```

---

## 📊 Archivos de Metadatos

### 5. **metadatos_modelo.json**
```json
{
  "timestamp": "20260108_143025",
  "nombre_modelo": "Random Forest",
  "dataset_size_train": 10500,
  "dataset_size_test": 2625,
  "num_features": 25,
  "metricas": {
    "r2_score": 0.8543,
    "rmse": 5234.50,
    "mae": 3456.75
  },
  "features": ["year", "mileage", "engineSize", ...],
  "variables_precio": {
    "min": 1500,
    "max": 150000,
    "media": 25000,
    "mediana": 18000
  }
}
```

**Usar para:**
- Documentar versión del modelo
- Registrar timestamp de entrenamiento
- Verificar métricas de rendimiento
- Rastrear cambios en el modelo

---

### 6. **categorias_mapping.json**
```json
{
  "transmission": {
    "clases": ["Automatic", "Manual", "Semi-Auto"],
    "num_clases": 3
  },
  "fuelType": {
    "clases": ["Diesel", "Hybrid", "Petrol"],
    "num_clases": 3
  },
  "brand": {
    "clases": ["Audi", "BMW", "Mercedes", ...],
    "num_clases": 45
  },
  "model": {
    "clases": ["A Class", "A4", "A6", ...],
    "num_clases": 287
  }
}
```

**Usar para:**
- Validar inputs del usuario en la aplicación web
- Crear dropdowns/selects con valores válidos
- Prevenir errores por categorías desconocidas

---

## 🚀 Cómo Usar en tu Aplicación Web

### Opción 1: Carga Simple (RECOMENDADA)

```python
# En tu aplicación Flask/Django
import joblib

# Cargar UNA SOLA VEZ al iniciar la aplicación
modelo = joblib.load('path/to/modelo_final_random_forest.joblib')

@app.route('/predict', methods=['POST'])
def predecir_precio():
    # Usuario envía datos
    datos = {
        'year': request.json['year'],
        'mileage': request.json['mileage'],
        'engineSize': request.json['engineSize'],
        'transmission': request.json['transmission'],
        'fuelType': request.json['fuelType'],
        'brand': request.json['brand'],
        'model': request.json['model']
    }
    
    # Convertir a DataFrame
    import pandas as pd
    df = pd.DataFrame([datos])
    
    # Predicción (¡el preprocessor está INCLUIDO!)
    precio = modelo.predict(df)[0]
    
    return {'precio_estimado': f'${precio:,.2f}'}
```

---

### Opción 2: Componentes Separados (AVANZADO)

```python
import joblib
import pandas as pd

# Cargar componentes
preprocessor = joblib.load('modelos_exportados/preprocessor.joblib')
modelo_ml = joblib.load('modelos_exportados/modelo_ml_random_forest.joblib')

def predecir_con_componentes(datos_dict):
    # Paso 1: Preparar datos
    df = pd.DataFrame([datos_dict])
    
    # Paso 2: Transformar (aplicar escalado y encoding)
    X_procesados = preprocessor.transform(df)
    
    # Paso 3: Predicción
    prediccion = modelo_ml.predict(X_procesados)
    
    return prediccion[0]
```

---

## ⚙️ Detalles Técnicos

### Diferencia: joblib vs pickle

| Aspecto | joblib | pickle |
|---------|--------|--------|
| Eficiencia | ✓ Mejor | Estándar |
| Tamaño | ✓ Más pequeño | Mayor |
| Velocidad | ✓ Más rápido | Más lento |
| Compresión | ✓ Automática | No |
| Paralelo | ✓ Soporte | No |

**Conclusión:** Usa **joblib** para modelos sklearn siempre que sea posible.

---

### Pipeline de Preprocesamiento Incluido

El modelo exportado contiene automáticamente:

```
┌─────────────────────────────────────────┐
│  Pipeline Completo (modelo_final.joblib) │
└─────────────────────────────────────────┘
         │
         ├─ Paso 1: Preprocessor
         │    ├─ StandardScaler (numeric features)
         │    │   └─ year, mileage, engineSize
         │    └─ OneHotEncoder (categorical features)
         │        └─ transmission, fuelType, brand, model
         │
         └─ Paso 2: Modelo ML
              └─ Random Forest / Gradient Boosting / Linear Regression
```

**Esto significa:**
- ✓ No necesitas preparar los datos manualmente
- ✓ Los datos se transforman automáticamente
- ✓ La predicción es directa

---

## 🐛 Troubleshooting

### Error: "No module named 'joblib'"
```bash
pip install joblib
```

### Error: "File not found"
Verifica que:
- La ruta del archivo es correcta
- El archivo existe en `modelos_exportados/`
- Los permisos de lectura están habilitados

### Error: "Modelo incompatible"
- Asegúrate de usar la versión de scikit-learn con la que fue entrenado
- Usa `pip install --upgrade scikit-learn` si es necesario

---

## 📝 Checklist de Implementación

- [ ] Ejecutar `practica_coches_2.py` completamente
- [ ] Verificar que se creó el directorio `modelos_exportados/`
- [ ] Verificar que existen todos los 6 archivos
- [ ] Cargar el modelo con `joblib.load()`
- [ ] Hacer una predicción de prueba
- [ ] Integrar en la aplicación web
- [ ] Documentar en README de la app

---

## 📚 Referencias

- [joblib Documentation](https://joblib.readthedocs.io/)
- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Model Persistence](https://scikit-learn.org/stable/modules/model_persistence.html)

