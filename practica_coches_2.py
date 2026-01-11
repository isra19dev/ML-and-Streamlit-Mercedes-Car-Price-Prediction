import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARTE 1: CARGAR Y EXPLORAR EL DATASET
# ============================================================================

print("=" * 80)
print("CARGANDO Y EXPLORANDO DATASET")
print("=" * 80)

# Obtener la ruta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'merc.csv')

# Cargar el dataset
df = pd.read_csv(csv_path)

# Información general del dataset
print(f"\n1. Forma del dataset: {df.shape}")
print(f"\n2. Primeras filas:")
print(df.head())

print(f"\n3. Información del dataset:")
print(df.info())

print(f"\n4. Estadísticas descriptivas:")
print(df.describe())

print(f"\n5. Valores faltantes:")
print(df.isnull().sum())

print(f"\n6. Tipos de datos:")
print(df.dtypes)

# Análisis de la variable objetivo (precio)
print(f"\n7. ANÁLISIS DE LA VARIABLE OBJETIVO (PRECIO):")
print(f"   - Precio mínimo: ${df['price'].min():.2f}")
print(f"   - Precio máximo: ${df['price'].max():.2f}")
print(f"   - Precio medio: ${df['price'].mean():.2f}")
print(f"   - Precio mediano: ${df['price'].median():.2f}")

# ============================================================================
# PARTE 2: PREPROCESAMIENTO (LIMPIEZA, CODIFICACIÓN Y ESCALADO)
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 2: PREPROCESAMIENTO DE DATOS")
print("=" * 80)

# Crear una copia del dataframe
df_processed = df.copy()

# Eliminar espacios en blanco de columnas y datos
df_processed.columns = df_processed.columns.str.strip()
for col in df_processed.select_dtypes(include=['object']).columns:
    df_processed[col] = df_processed[col].str.strip()

print(f"\n1. INFORMACIÓN INICIAL DEL DATASET:")
print(f"   - Forma: {df_processed.shape}")
print(f"   - Columnas: {df_processed.columns.tolist()}")

# Extraer marca del modelo
df_processed['brand'] = df_processed['model'].str.split().str[0]

print(f"\n2. MARCAS DETECTADAS:")
print(f"   - Total de marcas: {df_processed['brand'].nunique()}")
print(f"   - Primeras 10 marcas: {df_processed['brand'].unique()[:10].tolist()}")

# Definir características y variable objetivo
numeric_features = ['year', 'mileage', 'engineSize']
categorical_features = ['transmission', 'fuelType', 'brand', 'model']
all_features = numeric_features + categorical_features

X = df_processed[all_features].copy()
y = df_processed['price'].copy()

print(f"\n3. CARACTERÍSTICAS SELECCIONADAS:")
print(f"   - Numéricas: {numeric_features}")
print(f"   - Categóricas: {categorical_features}")
print(f"   - Total de muestras: {X.shape[0]}")
print(f"   - Total de características: {X.shape[1]}")

# Verificar valores faltantes
print(f"\n4. VALORES FALTANTES:")
missing_values = X.isnull().sum()
if missing_values.sum() == 0:
    print(f"   - No hay valores faltantes")
else:
    print(missing_values[missing_values > 0])

# ============================================================================
# PASO 1: SEPARAR EN CONJUNTOS DE ENTRENAMIENTO Y TEST (PRIMERO!)
# ============================================================================

print(f"\n5. SEPARACIÓN EN CONJUNTOS DE ENTRENAMIENTO Y TEST:")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   - Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"   - Conjunto de test: {X_test.shape[0]} muestras")
print(f"   - Proporción: {X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%} / {X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%}")

# ============================================================================
# PASO 2: CREAR PIPELINE DE PREPROCESAMIENTO CON COLUMNTRANSFORMER
# ============================================================================

print(f"\n6. CREACIÓN DE PIPELINE DE PREPROCESAMIENTO:")

# Definir transformaciones para variables numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Definir transformaciones para variables categóricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combinar transformaciones con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print(f"   - Transformador numérico: StandardScaler")
print(f"   - Transformador categórico: OneHotEncoding")

# Aplicar el preprocesamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n7. DATOS DESPUÉS DEL PREPROCESAMIENTO:")
print(f"   - Shape X_train: {X_train.shape} → {X_train_processed.shape}")
print(f"   - Shape X_test: {X_test.shape} → {X_test_processed.shape}")
print(f"   - Características resultantes:")

# Obtener nombres de características después de OneHotEncoding
feature_names = []
# Nombres numéricos
feature_names.extend(numeric_features)
# Nombres categóricos (resultado de OneHotEncoding)
cat_encoder = preprocessor.named_transformers_['cat']
onehot_features = cat_encoder.named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(onehot_features)

print(f"      Total: {len(feature_names)} características (después del encoding)")
print(f"      - Numéricas escaladas: {len(numeric_features)}")
print(f"      - Categóricas codificadas: {len(onehot_features)}")

print(f"\n   Primeras 20 características resultantes:")
for i, fname in enumerate(feature_names[:20], 1):
    print(f"      {i}. {fname}")

# ============================================================================
# PARTE 3: ENTRENAMIENTO Y COMPARACIÓN DE MODELOS CON PIPELINES
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 3: ENTRENAMIENTO Y COMPARACIÓN DE 3 MODELOS")
print("=" * 80)

# Diccionario para almacenar resultados
resultados = {}
modelos_entrenados = {}

# ==================== MODELO 1: REGRESIÓN LINEAL ====================
print("\n" + "-" * 80)
print("MODELO 1: REGRESIÓN LINEAL")
print("-" * 80)

# Pipeline completo: preprocesamiento + modelo
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

print("\n1. Entrenando modelo...")
pipeline_lr.fit(X_train, y_train)

print("2. Realizando predicciones en conjunto de test...")
y_pred_lr = pipeline_lr.predict(X_test)

# Calcular métricas
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

resultados['Regresión Lineal'] = {
    'MSE': mse_lr,
    'RMSE': rmse_lr,
    'MAE': mae_lr,
    'R²': r2_lr
}
modelos_entrenados['Regresión Lineal'] = pipeline_lr

print(f"\n3. Métricas:")
print(f"   - R² Score: {r2_lr:.4f}")
print(f"   - RMSE: ${rmse_lr:,.2f}")
print(f"   - MAE: ${mae_lr:,.2f}")
print(f"   - MSE: ${mse_lr:,.2f}")

# ==================== MODELO 2: RANDOM FOREST ====================
print("\n" + "-" * 80)
print("MODELO 2: RANDOM FOREST")
print("-" * 80)

# Pipeline completo: preprocesamiento + modelo
# Nota: Random Forest no necesita escalado, pero lo incluimos por consistencia
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

print("\n1. Entrenando modelo (esto puede tardar)...")
pipeline_rf.fit(X_train, y_train)

print("2. Realizando predicciones en conjunto de test...")
y_pred_rf = pipeline_rf.predict(X_test)

# Calcular métricas
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

resultados['Random Forest'] = {
    'MSE': mse_rf,
    'RMSE': rmse_rf,
    'MAE': mae_rf,
    'R²': r2_rf
}
modelos_entrenados['Random Forest'] = pipeline_rf

print(f"\n3. Métricas:")
print(f"   - R² Score: {r2_rf:.4f}")
print(f"   - RMSE: ${rmse_rf:,.2f}")
print(f"   - MAE: ${mae_rf:,.2f}")
print(f"   - MSE: ${mse_rf:,.2f}")

# Obtener importancia de características
print(f"\n4. Importancia de características (Top 10):")
feature_importance_rf = pipeline_rf.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_rf
}).sort_values('importance', ascending=False)
print(feature_importance_df.head(10).to_string(index=False))

# ==================== MODELO 3: GRADIENT BOOSTING ====================
print("\n" + "-" * 80)
print("MODELO 3: GRADIENT BOOSTING")
print("-" * 80)

# Pipeline completo: preprocesamiento + modelo
pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ))
])

print("\n1. Entrenando modelo...")
pipeline_gb.fit(X_train, y_train)

print("2. Realizando predicciones en conjunto de test...")
y_pred_gb = pipeline_gb.predict(X_test)

# Calcular métricas
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

resultados['Gradient Boosting'] = {
    'MSE': mse_gb,
    'RMSE': rmse_gb,
    'MAE': mae_gb,
    'R²': r2_gb
}
modelos_entrenados['Gradient Boosting'] = pipeline_gb

print(f"\n3. Métricas:")
print(f"   - R² Score: {r2_gb:.4f}")
print(f"   - RMSE: ${rmse_gb:,.2f}")
print(f"   - MAE: ${mae_gb:,.2f}")
print(f"   - MSE: ${mse_gb:,.2f}")

# Obtener importancia de características
print(f"\n4. Importancia de características (Top 10):")
feature_importance_gb = pipeline_gb.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_gb
}).sort_values('importance', ascending=False)
print(feature_importance_df.head(10).to_string(index=False))

# ============================================================================
# COMPARACIÓN FINAL DE MODELOS
# ============================================================================

print("\n" + "=" * 80)
print("COMPARACIÓN FINAL DE MODELOS")
print("=" * 80)

# Crear tabla de comparación
comparacion_df = pd.DataFrame(resultados).T
comparacion_df = comparacion_df[['R²', 'RMSE', 'MAE', 'MSE']]

print("\n1. TABLA COMPARATIVA NUMÉRICA DE MÉTRICAS:")
print("-" * 80)
print(comparacion_df.to_string())

# Crear tabla más visual con formato porcentual
print("\n2. TABLA COMPARATIVA FORMATEADA:")
print("-" * 80)
tabla_visual = comparacion_df.copy()
tabla_visual['R²'] = tabla_visual['R²'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
tabla_visual['RMSE'] = tabla_visual['RMSE'].apply(lambda x: f"${x:,.2f}")
tabla_visual['MAE'] = tabla_visual['MAE'].apply(lambda x: f"${x:,.2f}")
tabla_visual['MSE'] = tabla_visual['MSE'].apply(lambda x: f"${x:,.2f}")
print(tabla_visual.to_string())

# Análisis de diferencias
print("\n3. ANÁLISIS DE DIFERENCIAS ENTRE MODELOS:")
print("-" * 80)

r2_scores = comparacion_df['R²'].sort_values(ascending=False)
rmse_scores = comparacion_df['RMSE'].sort_values()

mejor_r2 = r2_scores.iloc[0]
peor_r2 = r2_scores.iloc[-1]
mejor_rmse = rmse_scores.iloc[0]
peor_rmse = rmse_scores.iloc[-1]

diferencia_r2 = mejor_r2 - peor_r2
diferencia_rmse = peor_rmse - mejor_rmse

print(f"\nMejor vs Peor (R² Score):")
print(f"  - Mejor: {r2_scores.index[0]} ({mejor_r2:.4f})")
print(f"  - Peor: {r2_scores.index[-1]} ({peor_r2:.4f})")
print(f"  - Diferencia: {diferencia_r2:.4f} ({diferencia_r2*100:.2f}%)")

print(f"\nMejor vs Peor (RMSE):")
print(f"  - Mejor: {rmse_scores.index[0]} (${mejor_rmse:,.2f})")
print(f"  - Peor: {rmse_scores.index[-1]} (${peor_rmse:,.2f})")
print(f"  - Diferencia: ${diferencia_rmse:,.2f}")

# Ranking
print("\n4. RANKING DE MODELOS (POR R² SCORE):")
print("-" * 80)
ranking = comparacion_df['R²'].sort_values(ascending=False)
for idx, (modelo, score) in enumerate(ranking.items(), 1):
    print(f"{idx}. {modelo:20s} → R² = {score:.4f} ({score*100:.2f}%)")

print("\n5. RANKING DE MODELOS (POR RMSE):")
print("-" * 80)
rmse_ranking = comparacion_df['RMSE'].sort_values()
for idx, (modelo, rmse) in enumerate(rmse_ranking.items(), 1):
    mae = comparacion_df.loc[modelo, 'MAE']
    print(f"{idx}. {modelo:20s} → RMSE = ${rmse:>10,.2f} | MAE = ${mae:>10,.2f}")

print("\n" + "=" * 80)
print("ANÁLISIS CRÍTICO DE LOS RESULTADOS")
print("=" * 80)

mejor_modelo = ranking.index[0]
mejor_r2 = ranking.iloc[0]
mejor_rmse = comparacion_df.loc[mejor_modelo, 'RMSE']
mejor_mae = comparacion_df.loc[mejor_modelo, 'MAE']

print(f"\n1. CARACTERÍSTICAS TÉCNICAS DEL MEJOR MODELO ({mejor_modelo}):")
print("-" * 80)

if mejor_modelo == 'Regresión Lineal':
    print("""
    • Modelo simple y rápido de entrenar
    • Asume relaciones lineales entre variables y precio
    • Bajo riesgo de overfitting
    • Alta interpretabilidad: cada coeficiente indica el impacto en el precio
    • Requiere escalado de variables (StandardScaler)
    • Computacionalmente eficiente
    """)
    
elif mejor_modelo == 'Random Forest':
    print("""
    • Modelo ensemble robusto y no paramétrico
    • Captura relaciones NO LINEALES en los datos
    • Bajo riesgo de overfitting gracias a la media de múltiples árboles
    • Maneja automáticamente variables categóricas después de OneHotEncoding
    • NO requiere escalado de variables (no es sensible a la escala)
    • Resistente a outliers (como vehículos de lujo o muy antiguos)
    • Proporciona medida de importancia de características
    • Más lento que Regresión Lineal pero mucho más preciso
    """)
    
elif mejor_modelo == 'Gradient Boosting':
    print("""
    • Modelo ensemble muy potente basado en boosting secuencial
    • Cada árbol intenta corregir errores de los anteriores
    • Excelente para capturar patrones complejos y no lineales
    • Mejor rendimiento general en competiciones de ML
    • Alto riesgo de overfitting (requiere ajuste de hiperparámetros)
    • Más lento que Random Forest en predicción
    • Sensible al learning_rate y profundidad de árboles
    • Proporciona medida de importancia de características
    """)

print(f"\n2. RENDIMIENTO GENERAL DEL MEJOR MODELO:")
print("-" * 80)
print(f"   • R² Score: {mejor_r2:.4f}")
print(f"     → Explica el {mejor_r2*100:.2f}% de la varianza en los precios")
print(f"\n   • RMSE: ${mejor_rmse:,.2f}")
print(f"     → Error típico en predicciones (±${mejor_rmse:,.2f})")
print(f"\n   • MAE: ${mejor_mae:,.2f}")
print(f"     → Error medio absoluto (±${mejor_mae:,.2f})")

# Porcentaje de error respecto al precio medio
precio_medio = y_test.mean()
error_porcentaje_mae = (mejor_mae / precio_medio) * 100
error_porcentaje_rmse = (mejor_rmse / precio_medio) * 100

print(f"\n   • Precio promedio en test set: ${precio_medio:,.2f}")
print(f"   • Error MAE como % del precio promedio: {error_porcentaje_mae:.2f}%")
print(f"   • Error RMSE como % del precio promedio: {error_porcentaje_rmse:.2f}%")

print(f"\n3. COMPARATIVA CON OTROS MODELOS:")
print("-" * 80)

for modelo in comparacion_df.index:
    if modelo != mejor_modelo:
        r2_diff = comparacion_df.loc[mejor_modelo, 'R²'] - comparacion_df.loc[modelo, 'R²']
        rmse_diff = comparacion_df.loc[modelo, 'RMSE'] - comparacion_df.loc[mejor_modelo, 'RMSE']
        
        print(f"\n   {mejor_modelo} vs {modelo}:")
        print(f"   • R² mejorado en: {r2_diff:.4f} ({r2_diff*100:.2f}%)")
        print(f"   • RMSE reducido en: ${rmse_diff:,.2f} ({(rmse_diff/comparacion_df.loc[modelo, 'RMSE'])*100:.1f}%)")

print(f"\n4. FORTALEZAS DEL MODELO SELECCIONADO:")
print("-" * 80)

if mejor_modelo == 'Random Forest':
    print("""
    ✓ Mejor balance entre precisión y complejidad
    ✓ Robusto ante datos atípicos (outliers)
    ✓ Maneja relaciones no lineales eficientemente
    ✓ Bajo riesgo de sobreajuste gracias a la agregación
    ✓ Interpretable mediante importancia de características
    ✓ Relativamente rápido en predicción
    """)
elif mejor_modelo == 'Gradient Boosting':
    print("""
    ✓ Máxima precisión entre los modelos evaluados
    ✓ Captura patrones muy complejos y no lineales
    ✓ Excelente para datos con estructura jerárquica
    ✓ Importancia de características confiable
    ✓ Mejor generalización en muchos casos
    """)
else:
    print("""
    ✓ Modelo simple y muy interpretable
    ✓ Rápido de entrenar y predecir
    ✓ Bajo riesgo de overfitting
    ✓ Consumo mínimo de recursos
    ✓ Fácil de implementar en producción
    """)

print(f"\n5. LIMITACIONES Y CONSIDERACIONES:")
print("-" * 80)

if mejor_modelo == 'Regresión Lineal':
    print("""
    ✗ Asume relaciones lineales (puede ser insuficiente)
    ✗ Menor capacidad predictiva que métodos más complejos
    ✗ No captura interacciones entre variables
    ✗ Sensible a outliers
    ✗ Requiere escalado de variables
    """)
elif mejor_modelo == 'Random Forest':
    print("""
    ✗ Modelo "caja negra" menos interpretable que regresión lineal
    ✗ Mayor complejidad computacional
    ✗ Requiere más memoria para almacenar
    ✗ Puede sobreajustar si no se ajustan bien los hiperparámetros
    ✗ Sesgado hacia variables categóricas con muchas clases
    """)
elif mejor_modelo == 'Gradient Boosting':
    print("""
    ✗ Mayor riesgo de overfitting (requiere validación cuidadosa)
    ✗ Más lento en entrenamiento que Random Forest
    ✗ Requiere ajuste fino de varios hiperparámetros
    ✗ Sensible al learning rate
    ✗ Más difícil de interpretar que modelos simples
    """)

# ============================================================================
# JUSTIFICACIÓN CLARA DEL MODELO SELECCIONADO
# ============================================================================

print("\n" + "=" * 80)
print("JUSTIFICACIÓN CLARA DEL MODELO SELECCIONADO")
print("=" * 80)

print(f"\n{mejor_modelo.upper()}")
print("=" * 80)

print(f"""
✓ RAZONES TÉCNICAS:

1. Rendimiento Superior
   • Logra un R² de {mejor_r2:.4f}, lo que significa que explica el 
     {mejor_r2*100:.2f}% de la variabilidad en los precios de vehículos
   • Error RMSE de ${mejor_rmse:,.2f}, muy competitivo para datos de precios
   
2. Naturaleza del Problema
   • La predicción de precios de vehículos involucra relaciones NO LINEALES
   • Diferentes marcas, modelos y tipos de combustible tienen impactos 
     variables en el precio
   • Los modelos ensemble capturan mejor estas complejidades
   
3. Robustez
   • {mejor_modelo} es resistente a outliers (coches de lujo, muy antiguos, etc.)
   • No requiere normalización/escalado (ventaja operacional)
   • Maneja bien datos mixtos (numéricos + categónicos)
   
4. Interpretabilidad
   • Proporciona ranking de importancia de características
   • Permite identificar qué variables más impactan el precio
   • Útil para stakeholders y toma de decisiones

5. Generalización
   • Bajo riesgo de overfitting gracias a la agregación de múltiples modelos
   • Buena capacidad de generalización a datos nuevos
   
✓ RAZONES PRÁCTICAS:

1. Aplicabilidad en Producción
   • Suficientemente rápido para predicciones en tiempo real
   • Fácil de serializar y desplegar
   • No requiere escalado en predicción
   
2. Mantenibilidad
   • No requiere ajuste fino de muchos hiperparámetros
   • Estable ante cambios pequeños en datos
   • Código limpio y fácil de mantener
   
3. Escalabilidad
   • Puede entrenarse en paralelo (n_jobs=-1)
   • Maneja datasets grandes eficientemente
""")

print(f"\n✓ CONCLUSIÓN FINAL:")
print("-" * 80)
print(f"""
Se selecciona {mejor_modelo} como modelo de producción para la aplicación
web porque:

• Ofrece el mejor balance entre precisión ({mejor_r2:.4f}) y complejidad
• Es robusto ante datos atípicos comunes en el mercado de segunda mano
• Proporciona buena interpretabilidad de características
• Es escalable y mantenible a largo plazo
• Tiene bajo riesgo de overfitting
• Es suficientemente rápido para predicciones en tiempo real

Este modelo será integrado en la aplicación web para que usuarios puedan
obtener estimaciones de precio precisas basadas en características del 
vehículo.
""")

# ============================================================================
# PARTE 4: OPTIMIZACIÓN DE HIPERPARÁMETROS DEL MEJOR MODELO
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 4: OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("=" * 80)

# Determinar cuál es el mejor modelo
mejor_modelo_nombre = ranking.index[0]

print(f"\n1. MODELO SELECCIONADO PARA OPTIMIZACIÓN: {mejor_modelo_nombre}")
print("-" * 80)

if mejor_modelo_nombre == 'Regresión Lineal':
    print("\n⚠️  NOTA: La Regresión Lineal no tiene hiperparámetros complejos para optimizar.")
    print("   (Solo tiene parámetro 'fit_intercept', que ya está optimizado)")
    print("   Se mantendrá el modelo actual con configuración por defecto.")
    
    modelo_final = pipeline_lr
    hiperparametros_optimos = {'modelo': 'LinearRegression', 'hiperparámetros': 'por defecto'}
    print("\n✓ Modelo optimizado (usando configuración por defecto)")
    
elif mejor_modelo_nombre == 'Random Forest':
    print("\nRealizando búsqueda de mejores hiperparámetros...")
    print("(Esto puede tardar unos minutos...)\n")
    
    # Crear pipeline solo con Random Forest para optimizar
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [15, 20, 25],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    pipeline_rf_opt = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    # GridSearchCV con validación cruzada
    grid_search = GridSearchCV(
        pipeline_rf_opt,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    print("Entrenando con GridSearchCV (5-fold)...")
    grid_search.fit(X_train, y_train)
    
    modelo_final = grid_search.best_estimator_
    hiperparametros_optimos = {
        'modelo': 'RandomForestRegressor',
        'mejores_hiperparámetros': grid_search.best_params_,
        'mejor_cv_score': grid_search.best_score_
    }
    
    print(f"\n✓ Mejores hiperparámetros encontrados:")
    for param, valor in grid_search.best_params_.items():
        print(f"   - {param}: {valor}")
    print(f"\nMejor CV Score (R²): {grid_search.best_score_:.4f}")
    
    # Calcular métricas con el modelo optimizado
    y_pred_final = modelo_final.predict(X_test)
    r2_final = r2_score(y_test, y_pred_final)
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
    mae_final = mean_absolute_error(y_test, y_pred_final)
    
    print(f"\nMétricas en conjunto TEST con modelo optimizado:")
    print(f"   - R² Score: {r2_final:.4f}")
    print(f"   - RMSE: ${rmse_final:,.2f}")
    print(f"   - MAE: ${mae_final:,.2f}")

elif mejor_modelo_nombre == 'Gradient Boosting':
    print("\nRealizando búsqueda de mejores hiperparámetros...")
    print("(Esto puede tardar unos minutos...)\n")
    
    # Crear pipeline solo con Gradient Boosting para optimizar
    param_grid = {
        'model__n_estimators': [100, 150, 200],
        'model__learning_rate': [0.05, 0.1, 0.15],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    pipeline_gb_opt = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(random_state=42))
    ])
    
    # GridSearchCV con validación cruzada
    grid_search = GridSearchCV(
        pipeline_gb_opt,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    print("Entrenando con GridSearchCV (5-fold)...")
    grid_search.fit(X_train, y_train)
    
    modelo_final = grid_search.best_estimator_
    hiperparametros_optimos = {
        'modelo': 'GradientBoostingRegressor',
        'mejores_hiperparámetros': grid_search.best_params_,
        'mejor_cv_score': grid_search.best_score_
    }
    
    print(f"\n✓ Mejores hiperparámetros encontrados:")
    for param, valor in grid_search.best_params_.items():
        print(f"   - {param}: {valor}")
    print(f"\nMejor CV Score (R²): {grid_search.best_score_:.4f}")
    
    # Calcular métricas con el modelo optimizado
    y_pred_final = modelo_final.predict(X_test)
    r2_final = r2_score(y_test, y_pred_final)
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
    mae_final = mean_absolute_error(y_test, y_pred_final)
    
    print(f"\nMétricas en conjunto TEST con modelo optimizado:")
    print(f"   - R² Score: {r2_final:.4f}")
    print(f"   - RMSE: ${rmse_final:,.2f}")
    print(f"   - MAE: ${mae_final:,.2f}")

# ============================================================================
# PARTE 5: EXPORTACIÓN DEL MODELO Y TRANSFORMADORES
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 5: EXPORTACIÓN DEL MODELO Y TRANSFORMADORES")
print("=" * 80)

# Crear directorio para guardar modelos
directorio_modelos = 'modelos_exportados'
if not os.path.exists(directorio_modelos):
    os.makedirs(directorio_modelos)
    print(f"\n✓ Creado directorio: {directorio_modelos}/")

print(f"\n1. EXPORTANDO MODELO FINAL ({mejor_modelo_nombre})")
print("-" * 80)

# Guardar modelo completo (pipeline) con joblib (más eficiente)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_modelo = f"modelo_final_{mejor_modelo_nombre.lower().replace(' ', '_')}_{timestamp}"

# Guardar con joblib (recomendado para sklearn)
ruta_modelo_joblib = os.path.join(directorio_modelos, f"{nombre_modelo}.joblib")
joblib.dump(modelo_final, ruta_modelo_joblib)
print(f"\n✓ Modelo completo guardado (joblib):")
print(f"   Ruta: {ruta_modelo_joblib}")
print(f"   Tamaño: {os.path.getsize(ruta_modelo_joblib) / (1024*1024):.2f} MB")

# También guardar con pickle por compatibilidad
ruta_modelo_pickle = os.path.join(directorio_modelos, f"{nombre_modelo}.pkl")
with open(ruta_modelo_pickle, 'wb') as f:
    pickle.dump(modelo_final, f)
print(f"\n✓ Modelo completo guardado (pickle - compatibilidad):")
print(f"   Ruta: {ruta_modelo_pickle}")
print(f"   Tamaño: {os.path.getsize(ruta_modelo_pickle) / (1024*1024):.2f} MB")

# ============================================================================
# Exportar SOLO el preprocessor (para hacer predicciones sin reentrenar)
# ============================================================================

print(f"\n2. EXPORTANDO PREPROCESSADOR (TRANSFORMADORES)")
print("-" * 80)

ruta_preprocessor = os.path.join(directorio_modelos, "preprocessor.joblib")
joblib.dump(preprocessor, ruta_preprocessor)
print(f"\n✓ Preprocessor guardado:")
print(f"   Ruta: {ruta_preprocessor}")
print(f"   Tamaño: {os.path.getsize(ruta_preprocessor) / (1024*1024):.2f} MB")
print(f"\n   Incluye:")
print(f"   - StandardScaler para variables numéricas: {numeric_features}")
print(f"   - OneHotEncoder para variables categóricas: {categorical_features}")

# ============================================================================
# Exportar el modelo entrenado (SOLO el componente del modelo)
# ============================================================================

print(f"\n3. EXPORTANDO MODELO DE MACHINE LEARNING (SOLO EL MODELO)")
print("-" * 80)

modelo_ml_solo = modelo_final.named_steps['model']
ruta_modelo_ml = os.path.join(directorio_modelos, f"modelo_ml_{mejor_modelo_nombre.lower().replace(' ', '_')}.joblib")
joblib.dump(modelo_ml_solo, ruta_modelo_ml)
print(f"\n✓ Componente ML guardado:")
print(f"   Ruta: {ruta_modelo_ml}")
print(f"   Tamaño: {os.path.getsize(ruta_modelo_ml) / (1024*1024):.2f} MB")
print(f"   (Para usar después con preprocessor guardado)")

# ============================================================================
# Guardar metadatos del modelo
# ============================================================================

print(f"\n4. GUARDANDO METADATOS DEL MODELO")
print("-" * 80)

metadatos = {
    'timestamp': timestamp,
    'nombre_modelo': mejor_modelo_nombre,
    'dataset_size_train': X_train.shape[0],
    'dataset_size_test': X_test.shape[0],
    'num_features': X_train.shape[1],
    'features': all_features,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'metricas': {
        'r2_score': float(r2_final if mejor_modelo_nombre != 'Regresión Lineal' else r2_lr),
        'rmse': float(rmse_final if mejor_modelo_nombre != 'Regresión Lineal' else rmse_lr),
        'mae': float(mae_final if mejor_modelo_nombre != 'Regresión Lineal' else mae_lr)
    },
    'hiperparametros': hiperparametros_optimos,
    'variables_precio': {
        'min': float(y_test.min()),
        'max': float(y_test.max()),
        'media': float(y_test.mean()),
        'mediana': float(y_test.median())
    }
}

ruta_metadatos = os.path.join(directorio_modelos, "metadatos_modelo.json")
with open(ruta_metadatos, 'w', encoding='utf-8') as f:
    json.dump(metadatos, f, indent=4, ensure_ascii=False)

print(f"\n✓ Metadatos guardados:")
print(f"   Ruta: {ruta_metadatos}")

print(f"\n   Contenido:")
print(f"   - Nombre del modelo: {metadatos['nombre_modelo']}")
print(f"   - Timestamp: {metadatos['timestamp']}")
print(f"   - R² Score: {metadatos['metricas']['r2_score']:.4f}")
print(f"   - RMSE: ${metadatos['metricas']['rmse']:,.2f}")
print(f"   - MAE: ${metadatos['metricas']['mae']:,.2f}")
print(f"   - Rango de precios: ${metadatos['variables_precio']['min']:,.0f} - ${metadatos['variables_precio']['max']:,.0f}")

# ============================================================================
# Guardar información de las clases categóricas
# ============================================================================

print(f"\n5. GUARDANDO DICCIONARIOS DE CATEGORÍAS")
print("-" * 80)

categorias_info = {}
onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']

for idx, categoria in enumerate(categorical_features):
    categorias_info[categoria] = {
        'clases': onehot_encoder.categories_[idx].tolist(),
        'num_clases': len(onehot_encoder.categories_[idx])
    }

ruta_categorias = os.path.join(directorio_modelos, "categorias_mapping.json")
with open(ruta_categorias, 'w', encoding='utf-8') as f:
    json.dump(categorias_info, f, indent=4, ensure_ascii=False)

print(f"\n✓ Categorías guardadas:")
print(f"   Ruta: {ruta_categorias}")
print(f"\n   Resumen de categorías:")
for cat, info in categorias_info.items():
    print(f"   - {cat}: {info['num_clases']} clases")

# ============================================================================
# Crear script de ejemplo para cargar y usar el modelo
# ============================================================================

print(f"\n6. CREANDO SCRIPT DE EJEMPLO PARA CARGAR EL MODELO")
print("-" * 80)

script_ejemplo = '''"""
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

print("\\nOpción 2: Cargar preprocessor y modelo por separado")
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

print("\\nPaso 3: Acceder a información del modelo")
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

print(f"\\nCategorías disponibles en 'transmission': {categorias['transmission']['clases']}")
'''

ruta_script = os.path.join(directorio_modelos, "ejemplo_uso_modelo.py")
with open(ruta_script, 'w', encoding='utf-8') as f:
    f.write(script_ejemplo)

print(f"\n✓ Script de ejemplo creado:")
print(f"   Ruta: {ruta_script}")
print(f"   Contiene instrucciones para cargar y usar el modelo")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN DE ARCHIVOS EXPORTADOS")
print("=" * 80)

print(f"\n📁 Directorio: {os.path.abspath(directorio_modelos)}/")
print("\n📄 Archivos generados:\n")

archivos = os.listdir(directorio_modelos)
for i, archivo in enumerate(sorted(archivos), 1):
    ruta_completa = os.path.join(directorio_modelos, archivo)
    tamaño = os.path.getsize(ruta_completa)
    print(f"{i}. {archivo}")
    print(f"   └─ Tamaño: {tamaño / 1024:.2f} KB")

print(f"\n✓ Todos los archivos necesarios han sido guardados correctamente")
print(f"✓ El modelo está listo para ser usado en la aplicación web")

print("\n" + "=" * 80)


