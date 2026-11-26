import streamlit as st
import numpy as np
import cv2
from detector import procesar_imagen_array, ZONAS

st.set_page_config(page_title="AutoTraffic · Ojos inteligentes", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #fff5f3;
}

h1 {
    color: #c62828;
    font-weight: 800;
}

div[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff;
    border: 2px dashed #e57373;
    border-radius: 12px;
}

div[data-testid="stSelectbox"] label {
    color: #b71c1c;
    font-weight: 600;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-top: 2px solid #e57373;
}
</style>
""", unsafe_allow_html=True)

st.title("🚨 AutoTraffic · Ojos inteligentes")
st.markdown("Sube una foto del carril confinado para **detectar riesgos viales**.")
st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    camaras_disponibles = list(ZONAS["camaras"].keys())
    camara_elegida = st.selectbox("📷 ¿De qué cámara es esta foto?", camaras_disponibles)
with col2:
    archivo_subido = st.file_uploader("📤 Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    bytes_imagen = np.frombuffer(archivo_subido.read(), dtype=np.uint8)
    imagen_cv = cv2.imdecode(bytes_imagen, cv2.IMREAD_COLOR)
    with st.spinner("Analizando imagen..."):
        imagen_marcada, veredicto, explicaciones = procesar_imagen_array(imagen_cv, camara_elegida)

    imagen_rgb = cv2.cvtColor(imagen_marcada, cv2.COLOR_BGR2RGB)
    st.image(imagen_rgb, caption="Resultado", use_container_width=True)

    if veredicto == "riesgo":
        st.error("⚠️ RIESGO detectado — Estado: pendiente de revisión")
    else:
        st.success("✅ Normal — sin riesgo detectado")

    st.subheader("¿Por qué esta decisión?")
    for razon in explicaciones:
        st.markdown(f"- {razon}")