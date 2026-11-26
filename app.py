import streamlit as st
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from detector import procesar_imagen_array, ZONAS

ARCHIVO_HISTORIAL = Path(__file__).resolve().parent / "historial.csv"

if "historial" not in st.session_state:
    if ARCHIVO_HISTORIAL.exists():
        st.session_state.historial = pd.read_csv(ARCHIVO_HISTORIAL).to_dict("records")
    else:
        st.session_state.historial = []


def registrar_incidencia(camara, veredicto):
    registro = {
        "hora": datetime.now().strftime("%H:%M:%S"),
        "camara": camara,
        "resultado": veredicto,
    }
    st.session_state.historial.insert(0, registro)
    pd.DataFrame(st.session_state.historial).to_csv(ARCHIVO_HISTORIAL, index=False)


st.set_page_config(page_title="AutoTraffic · Ojos inteligentes", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #0b0f17;
    color: #e6eaf0;
}

h1, h2, h3, p, span, label, li {
    color: #e6eaf0 !important;
}

h1 {
    font-weight: 800;
}

h1 span.marca-auto { color: #e6eaf0; }
h1 span.marca-traffic { color: #3b82f6; }

div[data-testid="stFileUploaderDropzone"] {
    background-color: #141a24;
    border: 2px dashed #2a3444;
    border-radius: 12px;
}

div[data-testid="stFileUploaderDropzone"] * {
    color: #93a1b4 !important;
}

div[data-testid="stSelectbox"] label {
    color: #93a1b4 !important;
    font-weight: 600;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #141a24;
    border-color: #2a3444 !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #e6eaf0 !important;
}

.stButton>button {
    background-color: #1f2937;
    color: #3b82f6 !important;
    border: 1px solid #2a3444;
    border-radius: 8px;
    font-weight: 600;
}

.stButton>button:hover {
    border-color: #3b82f6;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stAlertContentSuccess"] {
    background-color: #16311f !important;
}

div[data-testid="stAlertContentError"] {
    background-color: #3a1414 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #c5cdd8 !important;
}

[data-testid="stDataFrame"] {
    background-color: #141a24;
    border-radius: 12px;
}

hr {
    border-top: 1px solid #2a3444;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🚨 <span class="marca-auto">Auto</span><span class="marca-traffic">Traffic</span> · Ojos inteligentes</h1>', unsafe_allow_html=True)
st.markdown("Sube una foto del carril confinado para **detectar riesgos viales**.")
st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    camaras_disponibles = list(ZONAS["camaras"].keys()) + ["automatico"]
    camara_elegida = st.selectbox("📷 ¿De qué cámara es esta foto?", camaras_disponibles)
with col2:
    archivo_subido = st.file_uploader("📤 Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    bytes_imagen = np.frombuffer(archivo_subido.read(), dtype=np.uint8)
    imagen_cv = cv2.imdecode(bytes_imagen, cv2.IMREAD_COLOR)
    with st.spinner("Analizando imagen..."):
        imagen_marcada, veredicto, explicaciones = procesar_imagen_array(imagen_cv, camara_elegida)

    registrar_incidencia(camara_elegida, veredicto)

    imagen_rgb = cv2.cvtColor(imagen_marcada, cv2.COLOR_BGR2RGB)
    st.image(imagen_rgb, caption="Resultado", use_container_width=True)

    if veredicto == "riesgo":
        st.error("⚠️ RIESGO detectado — Estado: pendiente de revisión")
    else:
        st.success("✅ Normal — sin riesgo detectado")

    st.subheader("¿Por qué esta decisión?")
    for razon in explicaciones:
        st.markdown(f"- {razon}")

st.markdown("---")
st.subheader("📋 Historial de alertas")
if st.session_state.historial:
    st.dataframe(pd.DataFrame(st.session_state.historial), use_container_width=True, hide_index=True)
else:
    st.write("Sin alertas todavía.")