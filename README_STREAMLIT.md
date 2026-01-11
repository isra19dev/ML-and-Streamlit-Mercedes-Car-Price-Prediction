# 🚗 Interfaz Streamlit - Predictor de Precios Mercedes

## ¿Qué es esto?

Una **interfaz web profesional** creada con Streamlit que permite a los usuarios predecir precios de vehículos Mercedes de forma intuitiva.

---

## 📋 Requisitos Previos

Antes de ejecutar la aplicación Streamlit, **DEBES** completar estos pasos:

### 1. ✅ Ejecutar el script principal
```bash
cd "Practica 2 - Aplicacion web con ML"
python practica_coches_2.py
```

**Esto genera:**
- Carpeta `modelos_exportados/`
- Archivos del modelo entrenado
- Metadatos y categorías en JSON

**Espera 15-20 minutos** hasta que termine.

### 2. ✅ Verificar que se crearon los archivos
Debes ver estos archivos en `modelos_exportados/`:
- `modelo_final_random_forest_*.joblib`
- `metadatos_modelo.json`
- `categorias_mapping.json`
- `preprocessor.joblib`

---

## 🚀 Cómo Ejecutar la Interfaz

### Instalación de Streamlit (solo la primera vez)

```bash
pip install streamlit
```

### Ejecutar la aplicación

**Opción 1: Desde el directorio del proyecto**
```bash
cd "Practica 2 - Aplicacion web con ML"
streamlit run app_streamlit.py
```

**Opción 2: Desde cualquier lugar**
```bash
streamlit run "Practica 2 - Aplicacion web con ML\app_streamlit.py"
```

### ¿Qué verás?

Se abrirá automáticamente en tu navegador en:
```
http://localhost:8501
```

---

## 🎨 Características de la Interfaz

### ✨ Diseño Visual
- Interfaz limpia y profesional
- Colores personalizados
- Responsive (funciona en móvil)
- Indicadores de carga

### 🎯 Funcionalidades

#### **Formulario de Entrada**
- Slider para año (1990-2024)
- Input para kilometraje (0-500,000 km)
- Decimal para tamaño motor (0.5-10 L)
- Dropdowns para categorías (transmisión, combustible, marca, modelo)

#### **Validación**
- Validación automática de datos
- Mensajes de error claros
- Rangos realistas

#### **Resultados**
- Precio predicho formateado
- Métricas del modelo (R², Error típico)
- Datos confirmados de entrada
- Información sobre la predicción

#### **Barra Lateral**
- Información del modelo ML
- Métricas de precisión
- Fecha de entrenamiento
- Metadata completa

---

## 📊 Campos del Formulario

| Campo | Tipo | Rango | Ejemplo |
|-------|------|-------|---------|
| Año | Slider | 1990-2024 | 2020 |
| Kilometraje | Input | 0-500,000 | 50,000 |
| Tamaño motor | Decimal | 0.5-10 | 2.0 |
| Transmisión | Dropdown | Auto/Manual | Automatic |
| Combustible | Dropdown | Petrol/Diesel/Hybrid | Petrol |
| Marca | Dropdown | 45+ marcas | BMW |
| Modelo | Dropdown | Depende marca | Series 5 |

---

## 🔧 Troubleshooting

### Problema: "El modelo no está disponible"
**Solución:** Ejecuta `python practica_coches_2.py` primero para generar los archivos del modelo.

### Problema: "Module not found: streamlit"
**Solución:** Instala Streamlit con `pip install streamlit`

### Problema: "FileNotFoundError"
**Solución:** 
- Asegúrate de ejecutar desde la carpeta correcta
- O usa la ruta absoluta completa

### Problema: La app se ve lenta
**Solución:** 
- Streamlit recarga en tiempo real
- Espera a que termine la predicción
- Recarga la página si es necesario

---

## 💡 Cómo Funciona

### Flujo de la Aplicación

```
1. Usuario ingresa datos
   ↓
2. Validación de entrada
   ↓
3. Si hay errores → mostrar mensajes
   Si todo es OK → continuar
   ↓
4. Crear DataFrame con los datos
   ↓
5. Usar el modelo para predecir
   ↓
6. Mostrar resultado formateado
   ↓
7. Mostrar métricas de confianza
```

### Cargas en Caché
- El modelo se carga **UNA SOLA VEZ** al iniciar
- Predicciones son muy rápidas (<100ms)
- No se recarga el modelo con cada predicción

---

## 📝 Personalización

### Cambiar colores
Edita la sección CSS en el archivo (líneas 20-60):
```python
color: #1f77b4;  # Cambia este color
```

### Cambiar título
Línea 15:
```python
page_title="Tu título aquí"
```

### Agregar más información
Añade más `st.markdown()` en cualquier parte del archivo.

---

## 🌐 Despliegue en Producción

Streamlit puede desplegarse en:
- **Streamlit Cloud** (gratis, línea de comando: `streamlit deploy`)
- **Heroku** (con requirements.txt)
- **AWS / Google Cloud / Azure**
- **Tu propio servidor**

---

## 📦 Archivos Necesarios

Para que funcione, necesitas:

```
Practica 2 - Aplicacion web con ML/
├── app_streamlit.py                    ← Este archivo
├── practica_coches_2.py                ← El script que genera el modelo
├── merc.csv                             ← Datos de entrenamiento
└── modelos_exportados/                 ← Se crea automáticamente
    ├── modelo_final_random_forest_*.joblib
    ├── metadatos_modelo.json
    ├── categorias_mapping.json
    └── preprocessor.joblib
```

---

## 🎓 Conceptos Técnicos

### Caché de Streamlit
```python
@st.cache_resource
def cargar_modelo_y_componentes():
    # Esta función solo se ejecuta una vez
    # Luego se guardan los resultados en caché
```

### State Management
- Streamlit maneja automáticamente el estado
- Los valores persisten durante la sesión
- Se reinician al hacer F5

### Validación
- Se valida en el cliente (inmediato)
- Se valida nuevamente en la predicción (seguridad)

---

## ⚙️ Requisitos del Sistema

- Python 3.9+
- 500 MB de RAM
- Conexión a internet (opcional)
- Navegador moderno

---

## 📞 Soporte

Si algo no funciona:

1. **Verifica** que ejecutaste `python practica_coches_2.py`
2. **Comprueba** que existe `modelos_exportados/`
3. **Instala** Streamlit: `pip install streamlit`
4. **Lee** el PDF original para requisitos
5. **Revisa** los logs de error en la consola

---

## 🚀 Próximos Pasos

- ✅ Interfaz web creada
- ⏳ Ejecutar `python practica_coches_2.py` (si no lo hiciste)
- ⏳ Ejecutar `streamlit run app_streamlit.py`
- ⏳ Probar predicciones
- ⏳ Personalizar si lo deseas

---

## 📋 Checklist

- [ ] Ejecuté `python practica_coches_2.py`
- [ ] Se creó `modelos_exportados/`
- [ ] Instalé Streamlit (`pip install streamlit`)
- [ ] Ejecuté `streamlit run app_streamlit.py`
- [ ] La app se abrió en el navegador
- [ ] Hice una predicción de prueba
- [ ] El resultado se ve correctamente

---

**¡Listo para usar! 🎉**

Ahora tienes una interfaz web profesional para tu modelo de Machine Learning.
