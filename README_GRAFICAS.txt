# 📊 Análisis Gráfico del Modelo

Este script genera visualizaciones completas del desempeño del modelo de predicción de precios.

## ¿Qué muestra?

### Página 1: Análisis Principal (6 gráficas)
1. **Predicciones vs Valores Reales**: Dispersión de predicciones comparadas con precios reales
2. **Análisis de Residuos**: Distribución de errores
3. **Distribución de Errores**: Histograma de residuos
4. **MAE por Rango de Precio**: Cómo varia el error según el precio
5. **Comparación Train vs Test**: Métricas lado a lado
6. **Q-Q Plot**: Verificación de normalidad de residuos

### Página 2: Análisis Complementario (4 gráficas)
7. **Box Plot por Categoría**: Residuos clasificados por rango de precio
8. **Densidad 2D**: Mapa de calor predicciones vs reales
9. **Error Porcentual**: Distribución de errores en porcentaje
10. **Resumen de Métricas**: Tabla completa de resultados

## Cómo Ejecutar

```bash
cd "Practica 2 - Aplicacion web con ML"
python analisis_graficas_modelo.py
```

**Requisitos previos:**
- Ya debe haber ejecutado `python practica_coches_2.py` para generar el modelo

## Salida

- **Consola**: Métricas detalladas (MAE, RMSE, R², análisis estadístico)
- **Gráficas**: 2 ventanas con 10 gráficas totales

## Librerías Usadas

- matplotlib
- seaborn  
- pandas
- numpy
- scikit-learn
- joblib

Todas incluidas en requisitos estándar.

---

**¡Ejecuta para ver el análisis completo del modelo! 📈**
