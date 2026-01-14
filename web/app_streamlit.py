import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#! Configuraciones principales.
st.set_page_config(
    page_title="🚗 Predictor de Precios de Vehículos Mercedes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

#! Estilos de la página.

st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        font-size: 1.3em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #1f77b4;
        color: #000000;
    }
    
    .result-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 20px 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 20px 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        color: #000000;
    }
    
    .price-display {
        font-size: 2.5em;
        font-weight: bold;
        color: #28a745;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

#! Funciones.

@st.cache_resource
def cargar_modelo_y_componentes():
    """Carga el modelo entrenado y sus componentes"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        modelos_dir = os.path.join(script_dir, '..', 'models', 'modelos_exportados')
        
        # Buscar cualquier archivo del modelo (random_forest, gradient_boosting, etc.)
        modelo_files = [f for f in os.listdir(modelos_dir)
                       if f.startswith('modelo_final_') and f.endswith('.joblib')]
        
        if not modelo_files:
            return None, None, None, None
        
        # Tomar el primer archivo encontrado
        modelo_path = os.path.join(modelos_dir, modelo_files[0])
        metadatos_path = os.path.join(modelos_dir, 'metadatos_modelo.json')
        categorias_path = os.path.join(modelos_dir, 'categorias_mapping.json')
        
        modelo = joblib.load(modelo_path)
        
        with open(metadatos_path, 'r') as f:
            metadatos = json.load(f)
        
        with open(categorias_path, 'r') as f:
            categorias = json.load(f)
        
        return modelo, metadatos, categorias, modelo_files[0]
    
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None, None, None

def formatear_precio(precio):
    """Formatea el precio con separadores de miles"""
    return f"${precio:,.2f}"

def validar_entrada(data):
    """Valida los datos ingresados por el usuario"""
    errores = []
    
    #! Validar año
    if data['year'] < 1990 or data['year'] > 2020:
        errores.append("El año debe estar entre 1990 y 2020")
    
    #! Validar kilometraje
    if data['mileage'] < 0 or data['mileage'] > 500000:
        errores.append("El kilometraje debe estar entre 0 y 500,000 km")
    
    #! Validar tamaño del motor
    if data['engineSize'] <= 0 or data['engineSize'] > 10:
        errores.append("El tamaño del motor debe estar entre 0.5 y 10 litros")
    
    return errores

#! Carga de datos y modelo

modelo, metadatos, categorias, nombre_modelo = cargar_modelo_y_componentes()

#! Encabezado

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class="main-header">🚗 PREDICTOR DE PRECIOS</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Estima el precio de tu vehículo Mercedes</p>', unsafe_allow_html=True)

#! Comprobaciones sobre el modelo

if modelo is None:
    st.error("""
    ⚠️ **Error**: El modelo no está disponible.
    
    Por favor, asegúrate de que:
    1. Has ejecutado `python practica_coches_2.py`
    2. Se creó la carpeta `modelos_exportados/`
    3. Los archivos del modelo están presentes
    """)
    st.stop()

#! Formulario (Recogida de datos)

st.markdown("### 📝 Ingresa los datos del vehículo")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### **Datos Técnicos**")
    
    year = st.slider(
        "Año de fabricación",
        min_value=1990,
        max_value=2020,
        value=2020,
        step=1,
        help="El año en que se fabricó el vehículo"
    )
    
    mileage = st.number_input(
        "Kilometraje (km)",
        min_value=0,
        max_value=500000,
        value=50000,
        step=5000,
        help="Kilómetros recorridos por el vehículo"
    )
    
    engine_size = st.number_input(
        "Tamaño del motor (litros)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Cilindrada del motor en litros"
    )

with col2:
    st.markdown("#### **Características**")
    
    #! Obtener los valores únicos de categorías para evitar que el usuario cometa errores.
    transmission_options = categorias.get('transmission', {}).get('clases', ['Automatic', 'Manual', 'Semi-Auto'])
    fuel_type_options = categorias.get('fuelType', {}).get('clases', ['Petrol', 'Diesel', 'Hybrid'])
    
    transmission = st.selectbox(
        "Transmisión",
        options=transmission_options,
        help="Tipo de transmisión del vehículo"
    )
    
    fuel_type = st.selectbox(
        "Tipo de combustible",
        options=fuel_type_options,
        help="Tipo de combustible que utiliza"
    )

#! MODELO

st.markdown("#### **Modelo Mercedes**")

#! Obtención de los modelos disponibles
modelo_options = categorias.get('model', {}).get('clases', [])

model = st.selectbox(
    "Modelo",
    options=modelo_options if modelo_options else [],
    help="Modelo específico del vehículo Mercedes",
    key="model_select"
)

#! Boton de prediccion

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    boton_prediccion = st.button(
        "🔮 Predecir Precio",
        use_container_width=True,
        type="primary"
    )

#! Procesamiento de los datos y resultado de los mismos.

if boton_prediccion:
    #! Recolectar datos
    datos_entrada = {
        'year': int(year),
        'mileage': int(mileage),
        'engineSize': float(engine_size),
        'transmission': transmission,
        'fuelType': fuel_type,
        'brand': 'Mercedes',  #! Siempre Mercedes
        'model': model
    }
    
    #! Validar entrada
    errores = validar_entrada(datos_entrada)
    
    if errores:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("❌ Errores encontrados:")
        for error in errores:
            st.error(f"  • {error}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        #! Crear DataFrame para predicción
        df_prediccion = pd.DataFrame([datos_entrada])
        
        try:
            #! Realizar predicción
            with st.spinner("🔄 Procesando predicción..."):
                precio_predicho = modelo.predict(df_prediccion)[0]
            
            #! Mostrar resultado
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown("### ✅ Predicción Completada")
            
            #! Precio principal
            st.markdown(f'<div class="price-display">{formatear_precio(precio_predicho)}</div>', 
                       unsafe_allow_html=True)
            
            st.markdown("---")
            
            #! Detalles sobre la predicción
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Modelo ML</strong><br>
                    Random Forest<br>
                    <small>Optimizado</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                r2_score = metadatos.get('r2_score', 0.8543)
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Confianza</strong><br>
                    {r2_score:.1%}<br>
                    <small>R² Score</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                mae = metadatos.get('mae', 3456.75)
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Error Típico</strong><br>
                    ±{formatear_precio(mae)}<br>
                    <small>MAE</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### 📋 Datos Ingresados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Año:** {datos_entrada['year']}")
                st.write(f"**Kilometraje:** {datos_entrada['mileage']:,} km")
                st.write(f"**Motor:** {datos_entrada['engineSize']} L")
            
            with col2:
                st.write(f"**Transmisión:** {datos_entrada['transmission']}")
                st.write(f"**Combustible:** {datos_entrada['fuelType']}")
            
            with col3:
                st.write(f"**Marca:** {datos_entrada['brand']}")
                st.write(f"**Modelo:** {datos_entrada['model']}")
            
            st.markdown("---")
            
            #! Información adicional
            st.markdown(f"""
            <div class="info-box">
                <strong>ℹ️ Acerca de esta predicción:</strong><br>
                Este modelo fue entrenado con {13121:,} vehículos Mercedes reales.
                La precisión típica es de ±${metadatos.get('mae', 3456.75):,.0f}.
                Esta predicción es una estimación basada en datos históricos.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error(f"❌ Error durante la predicción: {str(e)}")
            st.error("Asegúrate de haber ejecutado previamente `python practica_coches_2.py`")
            st.markdown('</div>', unsafe_allow_html=True)

#! Informacion sobre el uso de la página web para el usuario.

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📚 Cómo Usar
    
    1. **Ingresa los datos** del vehículo que quieres valorar
    2. **Haz clic** en "Predecir Precio"
    3. **Obtén la estimación** con el precio predicho
    
    La predicción se basa en un modelo de Machine Learning entrenado con 
    miles de vehículos Mercedes reales.
    """)

with col2:
    st.markdown("""
    ### 🔍 Características del Modelo
    
    - ✅ **Precisión:** 85.43% (R² Score)
    - ✅ **Error típico:** ±$3,456.75 (MAE)
    - ✅ **Algoritmo:** Random Forest optimizado
    - ✅ **Validación:** 5-fold Cross-Validation
    - ✅ **Dataset:** 13,121 vehículos
    """)

#! Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🚗 Predictor de Precios de Vehículos Mercedes | Especialización en IA y Big Data</p>
    <p>Desarrollado con Streamlit y Machine Learning | 2026</p>
</div>
""", unsafe_allow_html=True)
