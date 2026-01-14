import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

#! CONFIGURACIÓN DE ESTILO

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#! CARGAR DATOS Y MODELO


print("=" * 80)
print("ANÁLISIS GRÁFICO DEL MODELO DE PREDICCIÓN DE PRECIOS")
print("=" * 80)

#! Obtener rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'merc.csv')
modelos_dir = os.path.join(script_dir, 'modelos_exportados')

#! Cargar datos
print("\n Cargando datos...")
df = pd.read_csv(csv_path)

#! Cargar modelo
print(" Cargando modelo entrenado...")
modelo_files = [f for f in os.listdir(modelos_dir)
                if f.startswith('modelo_final_') and f.endswith('.joblib')]

if not modelo_files:
    print(" Error: No se encontró el modelo. Ejecuta primero practica_coches_2.py")
    exit()

modelo_path = os.path.join(modelos_dir, modelo_files[0])
modelo = joblib.load(modelo_path)

preprocessor_path = os.path.join(modelos_dir, 'preprocessor.joblib')
preprocessor = joblib.load(preprocessor_path)

print(f" Modelo cargado: {modelo_files[0]}")

print("\n Preparando datos...")

#! Informacion sobre las columnas.
print(f"   Columnas disponibles: {list(df.columns)}")
available_cols = df.columns.tolist()

#! Crear la columna 'brand' a partir del 'model' si no existe, para consistencia
if 'brand' not in df.columns:
    print("   Creando columna 'brand' a partir de 'model'...")
    df['brand'] = df['model'].str.split().str[0]
    print("   Columna 'brand' creada")

#! Características necesarias en el entrenamiento del modelo
features = ['year', 'mileage', 'engineSize', 'transmission', 'fuelType', 'brand', 'model']

#! Verificación de que estas existan
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"   Características faltantes: {missing_features}")
    features = [f for f in features if f in df.columns]

#! Cargar datos
X = df[features].copy()
y = df['price'].copy()

print(f"   Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"   Características: {features}")

#! Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

try:
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    print("Datos preprocesados correctamente")
except Exception as e:
    print(f"Error en preprocesamiento: {str(e)}")
#! Si falla, usar los datos directamente    
    print("Usando los datos sin preprocesar")
   
    X_train_prep = X_train
    X_test_prep = X_test

#! Predicciones
print("🔮 Realizando predicciones...")
y_pred_train = modelo.predict(X_train) 
y_pred_test = modelo.predict(X_test)    

#! Cálculo de métricas
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n Predicciones completadas")
print(f"\n MÉTRICAS DE ENTRENAMIENTO:")
print(f" MAE Train:  ${mae_train:,.2f}")
print(f" RMSE Train: ${rmse_train:,.2f}")
print(f" R² Train:   {r2_train:.4f}")

print(f" MÉTRICAS DE TEST:")
print(f" MAE Test:   ${mae_test:,.2f}")
print(f" RMSE Test:  ${rmse_test:,.2f}")
print(f" R² Test:    {r2_test:.4f}")


print("Generando gráficas...")

fig = plt.figure(figsize=(18, 12))

#! 1. GRÁFICA 1: Predicciones vs Valores Reales (Test)

ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='steelblue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Predicción Perfecta')
ax1.set_xlabel('Precio Real ($)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Precio Predicho ($)', fontsize=11, fontweight='bold')
ax1.set_title('Predicciones vs Valores Reales (Test Set)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

#! 2. GRÁFICA 2: Residuos (Errores)

ax2 = plt.subplot(2, 3, 2)
residuos_test = y_test - y_pred_test
ax2.scatter(y_pred_test, residuos_test, alpha=0.5, s=30, color='coral')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Precio Predicho ($)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuo ($)', fontsize=11, fontweight='bold')
ax2.set_title('Análisis de Residuos (Test Set)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

#! 3. GRÁFICA 3: Distribución de Errores

ax3 = plt.subplot(2, 3, 3)
ax3.hist(residuos_test, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='Error = 0')
ax3.set_xlabel('Residuo ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax3.set_title('Distribución de Errores (Test Set)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

#! 4. GRÁFICA 4: Error Absoluto (MAE) por precio

ax4 = plt.subplot(2, 3, 4)
errores_abs = np.abs(residuos_test)
rangos_precio = pd.cut(y_test, bins=10)
errores_por_rango = pd.DataFrame({
    'rango': rangos_precio,
    'error': errores_abs
}).groupby('rango')['error'].agg(['mean', 'std'])

rangos_labels = [f"${int(interval.left/1000)}k-${int(interval.right/1000)}k" 
                 for interval in errores_por_rango.index]

ax4.bar(range(len(errores_por_rango)), errores_por_rango['mean'], 
        yerr=errores_por_rango['std'], capsize=5, color='skyblue', 
        edgecolor='black', alpha=0.7)
ax4.set_xticks(range(len(errores_por_rango)))
ax4.set_xticklabels(rangos_labels, rotation=45, ha='right')
ax4.set_ylabel('Error Absoluto Medio ($)', fontsize=11, fontweight='bold')
ax4.set_title('MAE por Rango de Precio', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

#! 5. GRÁFICA 5: Comparación Train vs Test

ax5 = plt.subplot(2, 3, 5)
metricas = ['MAE', 'RMSE', 'R²']
valores_train = [mae_train, rmse_train, r2_train]
valores_test = [mae_test, rmse_test, r2_test]

x = np.arange(len(metricas))
width = 0.35

ax5.bar(x - width/2, valores_train, width, label='Train', color='lightcoral', alpha=0.8)
ax5.bar(x + width/2, valores_test, width, label='Test', color='lightgreen', alpha=0.8)
ax5.set_ylabel('Valor', fontsize=11, fontweight='bold')
ax5.set_title('Comparación de Métricas: Train vs Test', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metricas)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

#! 6. GRÁFICA 6: Q-Q Plot (Normalidad de residuos)

ax6 = plt.subplot(2, 3, 6)
residuos_normalizados = (residuos_test - residuos_test.mean()) / residuos_test.std()
cuantiles_teoricos = np.quantile(np.random.normal(0, 1, 10000), 
                                 np.linspace(0.01, 0.99, len(residuos_normalizados)))
cuantiles_empiricos = np.sort(residuos_normalizados)

ax6.scatter(cuantiles_teoricos, cuantiles_empiricos, alpha=0.5, s=30, color='purple')
min_val = min(cuantiles_teoricos.min(), cuantiles_empiricos.min())
max_val = max(cuantiles_teoricos.max(), cuantiles_empiricos.max())
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Línea Teórica')
ax6.set_xlabel('Cuantiles Teóricos', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cuantiles Empíricos', fontsize=11, fontweight='bold')
ax6.set_title('Q-Q Plot (Normalidad de Residuos)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()


fig2 = plt.figure(figsize=(16, 10))

#! 7. GRÁFICA 7: Box Plot de Residuos
ax7 = plt.subplot(2, 2, 1)
deciles = pd.qcut(y_test, q=5, labels=['Muy Bajos', 'Bajos', 'Medios', 'Altos', 'Muy Altos'])
datos_boxplot = [residuos_test[deciles == cat].values for cat in deciles.unique()]

bp = ax7.boxplot(datos_boxplot, labels=['Muy Bajos', 'Bajos', 'Medios', 'Altos', 'Muy Altos'],
                 patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(range(5))):
    patch.set_facecolor(color)
ax7.axhline(y=0, color='r', linestyle='--', lw=2)
ax7.set_ylabel('Residuo ($)', fontsize=11, fontweight='bold')
ax7.set_xlabel('Categoría de Precio', fontsize=11, fontweight='bold')
ax7.set_title('Distribución de Residuos por Categoría de Precio', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')


#! 8. GRÁFICA 8: Histograma 2D de Densidad

ax8 = plt.subplot(2, 2, 2)
h = ax8.hist2d(y_test, y_pred_test, bins=30, cmap='YlOrRd')
plt.colorbar(h[3], ax=ax8, label='Frecuencia')
ax8.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'b--', lw=2, label='Predicción Perfecta')
ax8.set_xlabel('Precio Real ($)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Precio Predicho ($)', fontsize=11, fontweight='bold')
ax8.set_title('Densidad: Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
ax8.legend()

#! 9. GRÁFICA 9: Error Porcentual

ax9 = plt.subplot(2, 2, 3)
error_porcentual = np.abs((y_test - y_pred_test) / y_test * 100)
ax9.hist(error_porcentual, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
ax9.axvline(x=error_porcentual.mean(), color='r', linestyle='--', lw=2, 
            label=f'Media: {error_porcentual.mean():.2f}%')
ax9.set_xlabel('Error Porcentual (%)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax9.set_title('Distribución de Error Porcentual', fontsize=12, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

#! 10. GRÁFICA 10: Resumen de Métricas

ax10 = plt.subplot(2, 2, 4)
ax10.axis('off')

resumen_text = f"""
╔════════════════════════════════════════════════════════════╗
║          RESUMEN DE MÉTRICAS DEL MODELO                   ║
╚════════════════════════════════════════════════════════════╝

 CONJUNTO DE ENTRENAMIENTO:
   • MAE (Error Absoluto Medio):  ${mae_train:>12,.2f}
   • RMSE (Error Cuadrático Medio):  ${rmse_train:>10,.2f}
   • R² (Coeficiente de Determinación): {r2_train:>6.4f}

 CONJUNTO DE TEST:
   • MAE (Error Absoluto Medio):  ${mae_test:>12,.2f}
   • RMSE (Error Cuadrático Medio):  ${rmse_test:>10,.2f}
   • R² (Coeficiente de Determinación): {r2_test:>6.4f}

 ESTADÍSTICAS DE RESIDUOS (TEST):
   • Media de Residuos:  ${residuos_test.mean():>12,.2f}
   • Std Dev Residuos:   ${residuos_test.std():>12,.2f}
   • Error % Promedio:   {error_porcentual.mean():>12.2f}%

 ANÁLISIS:
   • Diferencia MAE (Train-Test): ${mae_test - mae_train:>8,.2f}
   • Diferencia RMSE (Train-Test): ${rmse_test - rmse_train:>8,.2f}
   • Diferencia R² (Train-Test): {r2_test - r2_train:>11.4f}
   
 Observaciones:
   - Residuos centrados en cero: {'SÍ' if abs(residuos_test.mean()) < 100 else 'NO'}
   - Modelo sin overfitting: {'SÍ' if abs(r2_test - r2_train) < 0.1 else 'POSIBLE'}
   - Distribución aproximadamente normal: {'SÍ' if abs(error_porcentual.skew()) < 1 else 'NO'}
"""

ax10.text(0.05, 0.95, resumen_text, transform=ax10.transAxes, 
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

#! MOSTRAR GRÁFICAS


print("\n Gráficas generadas correctamente")
print("\n Mostrando visualizaciones...")
print("\n   - Página 1: Análisis Principal (6 gráficas)")
print("   - Página 2: Análisis Complementario (4 gráficas)")

plt.show()

#! ESTADÍSTICAS ADICIONALES

print("\n" + "=" * 80)
print("ANÁLISIS ESTADÍSTICO COMPLETO")
print("=" * 80)

print(f"\n DISTRIBUCIÓN DE PRECIOS:")
print(f"   Precio Mínimo:    ${y.min():>12,.2f}")
print(f"   Precio Máximo:    ${y.max():>12,.2f}")
print(f"   Precio Promedio:  ${y.mean():>12,.2f}")
print(f"   Desv. Estándar:   ${y.std():>12,.2f}")

print(f"\n DISTRIBUCIÓN DE PREDICCIONES (TEST):")
print(f"   Min Predicción:   ${y_pred_test.min():>12,.2f}")
print(f"   Max Predicción:   ${y_pred_test.max():>12,.2f}")
print(f"   Promedio Pred:    ${y_pred_test.mean():>12,.2f}")
print(f"   Desv. Estándar:   ${y_pred_test.std():>12,.2f}")

print(f"\n ANÁLISIS DE ERRORES:")
print(f"   Error Mín:        ${errores_abs.min():>12,.2f}")
print(f"   Error Máx:        ${errores_abs.max():>12,.2f}")
print(f"   Error Med:        ${errores_abs.median():>12,.2f}")
print(f"   Error % Mín:      {(np.abs(y_test - y_pred_test) / y_test * 100).min():>12.2f}%")
print(f"   Error % Máx:      {(np.abs(y_test - y_pred_test) / y_test * 100).max():>12.2f}%")

print(f"\n Análisis completado exitosamente\n")
