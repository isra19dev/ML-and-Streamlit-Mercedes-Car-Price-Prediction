# ESTRUCTURA DE EXPORTACIÓN DEL MODELO

## 📦 Archivos Generados

```
modelos_exportados/
│
├── 📊 MODELO COMPLETO (Pipeline)
│   ├── modelo_final_random_forest_20260108_143025.joblib ⭐ USA ESTE
│   └── modelo_final_random_forest_20260108_143025.pkl    (backup)
│
├── 🔧 COMPONENTES SEPARADOS (Para uso avanzado)
│   ├── preprocessor.joblib                 (scalers + encoders)
│   └── modelo_ml_random_forest.joblib     (solo ML model)
│
├── 📋 METADATOS
│   ├── metadatos_modelo.json              (info del modelo)
│   ├── categorias_mapping.json            (valores válidos)
│   └── ejemplo_uso_modelo.py              (código ejemplo)
│
```

---

## 🎯 OPCIÓN 1: USO SIMPLE (RECOMENDADO)

### Para aplicación web

```python
# 1. CARGAR (una sola vez al inicio)
import joblib

modelo = joblib.load('modelos_exportados/modelo_final_random_forest.joblib')

# 2. USAR
import pandas as pd

datos = {
    'year': 2020,
    'mileage': 50000,
    'engineSize': 2.0,
    'transmission': 'Automatic',
    'fuelType': 'Petrol',
    'brand': 'BMW',
    'model': 'Series 5'
}

df = pd.DataFrame([datos])
precio = modelo.predict(df)[0]

print(f"Precio predicho: ${precio:,.2f}")  # Precio predicho: $28,456.75
```

**Ventajas:**
- ✓ Una línea de código
- ✓ No necesitas entender preprocesamiento
- ✓ Automático y seguro

---

## 🔧 OPCIÓN 2: USO CON COMPONENTES (Para casos especiales)

### Cuando necesitas máximo control

```python
import joblib
import pandas as pd

# 1. CARGAR COMPONENTES
preprocessor = joblib.load('modelos_exportados/preprocessor.joblib')
modelo_ml = joblib.load('modelos_exportados/modelo_ml_random_forest.joblib')

# 2. DATOS NUEVOS
datos = pd.DataFrame([{
    'year': 2020,
    'mileage': 50000,
    'engineSize': 2.0,
    'transmission': 'Automatic',
    'fuelType': 'Petrol',
    'brand': 'BMW',
    'model': 'Series 5'
}])

# 3. TRANSFORMAR
X_procesados = preprocessor.transform(datos)

# 4. PREDICCIÓN
precio = modelo_ml.predict(X_procesados)[0]
```

**Casos de uso:**
- Batch predictions (muchos registros)
- Pipeline personalizado
- Monitoreo de transformaciones

---

## 📊 FLUJO DE DATOS

### Modelo Completo (Opción 1)
```
Datos Nuevos
    ↓
┌─────────────────────┐
│ modelo_final.joblib │  ← Pipeline COMPLETO
│  ┌───────────────┐  │
│  │ Preprocessor  │  │
│  │  ├─ Scaler    │  │
│  │  └─ Encoder   │  │
│  ├───────────────┤  │
│  │  ML Model     │  │
│  │  (Random      │  │
│  │   Forest)     │  │
│  └───────────────┘  │
└─────────────────────┘
    ↓
Precio Predicho
```

### Componentes Separados (Opción 2)
```
Datos Nuevos
    ↓
┌──────────────────┐
│  Preprocessor    │
│  ├─ Scaler       │
│  └─ Encoder      │
└──────────────────┘
    ↓ (Datos transformados)
┌──────────────────┐
│  ML Model        │
│  (Random Forest) │
└──────────────────┘
    ↓
Precio Predicho
```

---

## 📋 VARIABLES DE ENTRADA (Features)

### Numéricas (se escalan automáticamente)
- `year` → Año del vehículo (ej: 2020)
- `mileage` → Kilómetros (ej: 50000)
- `engineSize` → Tamaño motor (ej: 2.0)

### Categóricas (se codifican automáticamente)
- `transmission` → "Automatic" o "Manual"
- `fuelType` → "Petrol", "Diesel", "Hybrid"
- `brand` → Marca del vehículo (ej: "BMW")
- `model` → Modelo (ej: "Series 5")

Ver valores válidos en `categorias_mapping.json`

---

## 🔍 METADATOS DEL MODELO

Archivo: `metadatos_modelo.json`

```json
{
  "timestamp": "20260108_143025",
  "nombre_modelo": "Random Forest",
  "r2_score": 0.8543,
  "rmse": 5234.50,
  "mae": 3456.75
}
```

**Úsalo para:**
- Registrar cuándo fue entrenado
- Verificar rendimiento
- Decidir cuándo reentrenar
- Documentar versiones

---

## ✅ CHECKLIST DE VALIDACIÓN

Antes de usar en producción:

```
[ ] 1. ¿Se creó la carpeta modelos_exportados/?
[ ] 2. ¿Existen los 6 archivos esperados?
[ ] 3. ¿Probaste a cargar el modelo sin errores?
[ ] 4. ¿Hiciste una predicción de prueba?
[ ] 5. ¿El precio predicho es razonable?
[ ] 6. ¿Verificaste los metadatos del modelo?
[ ] 7. ¿Documentaste los cambios en tu repo?
```

---

## 🚀 INTEGRACIÓN EN FLASK (EJEMPLO)

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# 1. CARGAR MODELO AL INICIAR
modelo = joblib.load('modelos_exportados/modelo_final_random_forest.joblib')
with open('modelos_exportados/metadatos_modelo.json') as f:
    metadatos = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predicción de precios"""
    try:
        # 2. OBTENER DATOS DEL USUARIO
        datos = request.json
        
        # 3. VALIDAR (opcional pero recomendado)
        campos_requeridos = ['year', 'mileage', 'engineSize', 
                            'transmission', 'fuelType', 'brand', 'model']
        if not all(campo in datos for campo in campos_requeridos):
            return {'error': 'Faltan campos requeridos'}, 400
        
        # 4. PREPARAR PARA PREDICCIÓN
        df = pd.DataFrame([datos])
        
        # 5. PREDICCIÓN
        precio = float(modelo.predict(df)[0])
        
        # 6. RESPONDER
        return {
            'success': True,
            'precio_predicho': precio,
            'formato': f'${precio:,.2f}',
            'modelo': metadatos['nombre_modelo'],
            'confianza': f"{metadatos['r2_score']*100:.1f}%"
        }
    
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📚 Recursos Útiles

- [Documentación joblib](https://joblib.readthedocs.io/)
- [Pipelines en sklearn](https://scikit-learn.org/stable/modules/compose.html)
- [Persistencia de modelos](https://scikit-learn.org/stable/modules/model_persistence.html)

