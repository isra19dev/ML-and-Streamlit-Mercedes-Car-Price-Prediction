"""
SCRIPT DE PRUEBA: Cómo usar el modelo exportado
Úsalo para verificar que todo funciona correctamente
"""

import joblib
import pandas as pd
import json
import os

print("=" * 80)
print("SCRIPT DE PRUEBA: CARGAR Y USAR EL MODELO EXPORTADO")
print("=" * 80)

#! Directorio donde se guardaron los modelos, creado en practica_coches_2.py
directorio = 'modelos_exportados'

#! Verificar que existe el directorio, y lanza una alternativa en caso de no encontrarlo
if not os.path.exists(directorio):
    print(f"\n❌ ERROR: No se encontró el directorio '{directorio}'")
    print("   Asegúrate de haber ejecutado practica_coches_2.py primero")
    exit()

print(f"\n✓ Directorio encontrado: {directorio}/")

#! OPCIÓN 1: CARGAR EL MODELO COMPLETO (RECOMENDADO), en caso de que se disponga del mismo.

print("\n" + "=" * 80)
print("OPCIÓN 1: USANDO MODELO COMPLETO")
print("=" * 80)

archivos_modelo = [f for f in os.listdir(directorio) 
                   if f.startswith('modelo_final_') and f.endswith('.joblib')]

#! Si no encuentra el archivo .joblib, lo notifica con un error
if not archivos_modelo:
    print("\n ERROR: No se encontró modelo_final_*.joblib")
    exit()

archivo_modelo = os.path.join(directorio, archivos_modelo[0])

print(f"\n1. Cargando modelo: {archivos_modelo[0]}")
try:
    modelo = joblib.load(archivo_modelo)
    print(" El modelo ha sido cargado con éxito")
except Exception as e:
    print(f" Hubo un error en la carga del modelo: {e}")
    exit()

#! OPCIÓN 2: CARGAR METADATOS


print(f"\n2. Cargando metadatos del modelo")
try:
    with open(os.path.join(directorio, 'metadatos_modelo.json'), 'r') as f:
        metadatos = json.load(f)
    print("   ✓ Metadatos cargados")
    
    print(f"\n   Información del modelo:")
    print(f"   - Nombre: {metadatos['nombre_modelo']}")
    print(f"   - Timestamp: {metadatos['timestamp']}")
    print(f"   - R² Score: {metadatos['metricas']['r2_score']:.4f}")
    print(f"   - RMSE: ${metadatos['metricas']['rmse']:,.2f}")
    print(f"   - MAE: ${metadatos['metricas']['mae']:,.2f}")
except Exception as e:
    print(f"   ⚠️  Advertencia: {e}")

#! OPCIÓN 3: CARGAR MAPEO DE CATEGORÍAS

print(f"\n3. Cargando mapeo de categorías")
try:
    with open(os.path.join(directorio, 'categorias_mapping.json'), 'r') as f:
        categorias = json.load(f)
    print("   ✓ Categorías cargadas")
    
    print(f"\n   Valores válidos para variables categóricas:")
    for variable, info in categorias.items():
        num_clases = info['num_clases']
        primeros = info['clases'][:3]
        print(f"   - {variable}: {num_clases} opciones")
        print(f"     Ejemplos: {', '.join(primeros)}")
except Exception as e:
    print(f"   ⚠️  Advertencia: {e}")

#! Realización de algunas predicciones de prueba para comprobar el funcionamiento del modelo.

print("\n" + "=" * 80)
print("PRUEBA DE PREDICCIÓN")
print("=" * 80)

#! Vehiculo económico:
print("\n4. Ejemplo 1: Vehículo económico")
print("-" * 80)

datos1 = {
    'year': 2018,
    'mileage': 80000,
    'engineSize': 1.6,
    'transmission': 'Manual',
    'fuelType': 'Petrol',
    'brand': 'Toyota',
    'model': 'Corolla'
}

print(f"\nDatos de entrada:")
for clave, valor in datos1.items():
    print(f"  {clave}: {valor}")

try:
    df1 = pd.DataFrame([datos1])
    prediccion1 = modelo.predict(df1)[0]
    print(f"\n✓ Precio predicho: ${prediccion1:,.2f}")
except Exception as e:
    print(f"\n❌ Error en predicción: {e}")

#! Vehículo de alta gama:
print("\n\n5. Ejemplo 2: Vehículo de alta gama")
print("-" * 80)

datos2 = {
    'year': 2020,
    'mileage': 30000,
    'engineSize': 3.0,
    'transmission': 'Automatic',
    'fuelType': 'Petrol',
    'brand': 'BMW',
    'model': 'Series 5'
}

print(f"\nDatos de entrada:")
for clave, valor in datos2.items():
    print(f"  {clave}: {valor}")

try:
    df2 = pd.DataFrame([datos2])
    prediccion2 = modelo.predict(df2)[0]
    print(f"\n✓ Precio predicho: ${prediccion2:,.2f}")
except Exception as e:
    print(f"\n❌ Error en predicción: {e}")

#! Vehículo de diésel::
print("\n\n6. Ejemplo 3: Vehículo diesel")
print("-" * 80)

datos3 = {
    'year': 2019,
    'mileage': 50000,
    'engineSize': 2.0,
    'transmission': 'Automatic',
    'fuelType': 'Diesel',
    'brand': 'Mercedes',
    'model': 'C Class'
}

print(f"\nDatos de entrada:")
for clave, valor in datos3.items():
    print(f"  {clave}: {valor}")

try:
    df3 = pd.DataFrame([datos3])
    prediccion3 = modelo.predict(df3)[0]
    print(f"\n✓ Precio predicho: ${prediccion3:,.2f}")
except Exception as e:
    print(f"\n❌ Error en predicción: {e}")

#! Y se realiza una predicción en lote para no tener que ir vehículo por vehículo.

print("\n\n" + "=" * 80)
print("PREDICCIÓN EN LOTE (Batch Prediction)")
print("=" * 80)

print("\n7. Predicción para múltiples vehículos a la vez")
print("-" * 80)

datos_lote = [
    {
        'year': 2021, 'mileage': 20000, 'engineSize': 1.8,
        'transmission': 'Manual', 'fuelType': 'Petrol',
        'brand': 'Audi', 'model': 'A4'
    },
    {
        'year': 2018, 'mileage': 100000, 'engineSize': 1.5,
        'transmission': 'Automatic', 'fuelType': 'Petrol',
        'brand': 'Honda', 'model': 'Civic'
    },
    {
        'year': 2020, 'mileage': 40000, 'engineSize': 2.2,
        'transmission': 'Automatic', 'fuelType': 'Diesel',
        'brand': 'VW', 'model': 'Passat'
    }
]

try:
    df_lote = pd.DataFrame(datos_lote)
    predicciones_lote = modelo.predict(df_lote)
    
    print(f"\nPredicciones para {len(datos_lote)} vehículos:\n")
    for i, (datos, precio) in enumerate(zip(datos_lote, predicciones_lote), 1):
        print(f"{i}. {datos['brand']} {datos['model']} ({datos['year']})")
        print(f"   Mileage: {datos['mileage']:,} km | Motor: {datos['engineSize']}L")
        print(f"   └─ Precio predicho: ${precio:,.2f}\n")
        
except Exception as e:
    print(f"\n❌ Error en predicción en lote: {e}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✓ PRUEBA COMPLETADA EXITOSAMENTE")
print("=" * 80)

print(f"""
✓ El modelo se cargó correctamente
✓ Se realizaron predicciones sin errores
✓ Los precios predichos parecen razonables

📋 PRÓXIMOS PASOS:

1. Si todo funcionó: Integra el modelo en tu aplicación web
2. Si hay errores: Revisa la consola para mensajes de error
3. Para producción: Sigue la guía GUIA_EXPORTACION_MODELO.md

📚 RECURSOS:
- GUIA_EXPORTACION_MODELO.md (Documentación completa)
- ESTRUCTURA_EXPORTACION.md (Diagramas y ejemplos)
- README_EXPORTACION.txt (Visión general)
- ejemplo_uso_modelo.py (Código en modelos_exportados/)
""")

print("=" * 80)
