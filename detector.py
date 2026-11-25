from pathlib import Path
import json
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "zonas.json", encoding="utf-8") as f:
    ZONAS = json.load(f)

modelo = YOLO(str(BASE_DIR / "yolo26n.pt"))

NOMBRES_ES = {
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "carro",
    "motorcycle": "moto",
    "bus": "autobús",
    "truck": "camión",
}


def procesar_imagen_array(imagen_cv, nombre_camara: str):
    poligono = np.array(ZONAS["camaras"][nombre_camara], dtype=np.int32)

    overlay = imagen_cv.copy()
    cv2.fillPoly(overlay, [poligono], color=(60, 60, 220))
    imagen_cv = cv2.addWeighted(overlay, 0.25, imagen_cv, 0.75, 0)
    cv2.polylines(imagen_cv, [poligono], isClosed=True, color=(0, 0, 255), thickness=2)

    resultados = modelo.predict(source=imagen_cv, conf=0.15, save=False, verbose=False)

    huvo_alerta = False
    explicaciones = []

    for resultado in resultados:
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            confianza = float(caja.conf[0])
            nombre = modelo.names[clase]
            nombre_es = NOMBRES_ES.get(nombre, nombre)

            x1, y1, x2, y2 = caja.xyxy[0]
            punto_base =(float((x1 + x2) / 2), float((y1 + y2) / 2))