import streamlit as st
import numpy as np
import cv2
from detector import procesar_imagen_array, ZONAS

st.set_page_config(page_title="AutoTraffic · Ojos inteligentes", layout="centered")
st.title("AutoTraffic · Ojos inteligentes")
st.write("Sube una foto del carril confinado para detectar riesgos viales.")

camaras_disponibles = list(ZONAS["camaras"].keys())
camara_elegida = st.selectbox("¿De qué cámara es esta foto?", camaras_disponibles)

archivo_subido = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    bytes_imagen = np.frombuffer(archivo_subido.read(), dtype=np.uint8)
    imagen_cv = cv2.imdecode(bytes_imagen, cv2.IMREAD_COLOR)
    with st.spinner("Analizando imagen..."):
        imagen_marcada, veredicto, explicaciones = procesar_imagen_array(imagen_cv, camara_elegida)  # <<< CAMBIADO

    imagen_rgb = cv2.cvtColor(imagen_marcada, cv2.COLOR_BGR2RGB)
    st.image(imagen_rgb, caption="Resultado", use_container_width=True)

    if veredicto == "riesgo":
        st.error(" RIESGO detectado — Estado: pendiente de revisión")
    else:
        st.success(" Normal — sin riesgo detectado")

    # <<< NUEVO: mostrar el razonamiento
    st.subheader("¿Por qué esta decisión?")
    for razon in explicaciones:
        st.markdown(f"- {razon}")