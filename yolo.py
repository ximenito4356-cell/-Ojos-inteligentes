from ultralytics import YOLO
from pathlib import Path
import json
import csv
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
CARPETA_FOTOS = BASE_DIR / "fotos"
CARPETA_SALIDA = BASE_DIR / "salidas"
CARPETA_SALIDA.mkdir(exist_ok=True)

# Cargar zonas una sola vez
with open(BASE_DIR / "zonas.json", encoding="utf-8") as f:
    zonas = json.load(f)

# Cargar modelo una sola vez
print("Cargando YOLO...")
modelo = YOLO("yolo26n.pt")


def procesar_imagen(ruta_imagen: Path, nombre_camara: str):
    poligono = np.array(zonas["camaras"][nombre_camara], dtype=np.int32)
    imagen_cv = cv2.imread(str(ruta_imagen))

    # Dibujar la zona restringida
    overlay = imagen_cv.copy()
    cv2.fillPoly(overlay, [poligono], color=(60, 60, 220))
    imagen_cv = cv2.addWeighted(overlay, 0.25, imagen_cv, 0.75, 0)
    cv2.polylines(imagen_cv, [poligono], isClosed=True, color=(0, 0, 255), thickness=2)

    resultados = modelo.predict(source=str(ruta_imagen), conf=0.25, save=False, verbose=False)

    huvo_alerta = False

    for resultado in resultados:
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            confianza = float(caja.conf[0])
            nombre = modelo.names[clase]

            x1, y1, x2, y2 = caja.xyxy[0]
            punto_base = (float((x1 + x2) / 2), float(y2))

            dentro = cv2.pointPolygonTest(poligono, punto_base, False) >= 0
            es_bus = nombre == "bus"
            es_alerta = dentro and not es_bus
            if es_alerta:
                huvo_alerta = True

            color = (0, 0, 255) if es_alerta else (0, 255, 0)
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(imagen_cv, (x1i, y1i), (x2i, y2i), color, 2)

            etiqueta = (
                f"ALERTA: invasion de zona ({nombre} {confianza:.2f})"
                if es_alerta
                else f"{nombre} {confianza:.2f} - sin riesgo"
            )
            cv2.putText(imagen_cv, etiqueta, (x1i, max(y1i - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if huvo_alerta:
        cv2.putText(imagen_cv, "Estado: pendiente de revision", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    salida = CARPETA_SALIDA / f"marcada_{ruta_imagen.name}"
    cv2.imwrite(str(salida), imagen_cv)
    return "riesgo" if huvo_alerta else "normal"


# Leer etiquetas.csv para saber la cámara de cada foto, y recorrer todas
with open(BASE_DIR / "etiquetas.csv", encoding="utf-8") as f:
    lector = csv.DictReader(f)
    for fila in lector:
        ruta = CARPETA_FOTOS / fila["archivo"]
        if not ruta.exists():
            print(f"No encontrada: {ruta}")
            continue

        prediccion = procesar_imagen(ruta, fila["camara"])
        print(f"{fila['archivo']} ({fila['camara']}) -> predicho: {prediccion} | real: {fila['etiqueta']}")

print("Proceso terminado.")