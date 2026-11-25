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

    resultados = modelo.predict(
        source=imagen_cv,
        conf=0.15,
        classes=[0, 1, 2, 3, 5, 7],  # persona, bicicleta, carro, moto, autobús, camión
        save=False,
        verbose=False,
    )

    huvo_alerta = False
    explicaciones = []

    for resultado in resultados:
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            confianza = float(caja.conf[0])
            nombre = modelo.names[clase]
            nombre_es = NOMBRES_ES.get(nombre, nombre)

            x1, y1, x2, y2 = caja.xyxy[0]
            punto_base = (float((x1 + x2) / 2), float(y2))

            dentro = cv2.pointPolygonTest(poligono, punto_base, False) >= 0
            es_bus = nombre == "bus"
            es_alerta = dentro and not es_bus
            if es_alerta:
                huvo_alerta = True

            if es_alerta:
                razon = (
                    f"Se detectó un/a **{nombre_es}** (confianza {confianza:.0%}) "
                    f"cuyo punto de apoyo cae **dentro** de la zona restringida, "
                    f"y no es el autobús autorizado, entonces cuenta como riesgo."
                )
            elif dentro and es_bus:
                razon = (
                    f"Se detectó un **autobús** (confianza {confianza:.0%}) dentro de la zona, "
                    f"pero es el vehículo autorizado, entonces no cuenta como riesgo."
                )
            else:
                razon = (
                    f"Se detectó un/a **{nombre_es}** (confianza {confianza:.0%}), "
                    f"pero su punto de apoyo cae **fuera** de la zona restringida, entonces se considera sin riesgo."
                )
            explicaciones.append(razon)

            color = (0, 0, 255) if es_alerta else (0, 255, 0)
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(imagen_cv, (x1i, y1i), (x2i, y2i), color, 2)

            etiqueta = (
                f"ALERTA: invasion de zona ({nombre_es} {confianza:.2f})"
                if es_alerta
                else f"{nombre_es} {confianza:.2f} - sin riesgo"
            )
            pos_y_etiqueta = y1i - 10 if y1i > 40 else y2i + 20
            cv2.putText(imagen_cv, etiqueta, (x1i, pos_y_etiqueta),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resumen de conteo por tipo de objeto
    conteo = {}
    for resultado in resultados:
        for caja in resultado.boxes:
            nombre_obj = modelo.names[int(caja.cls[0])]
            nombre_obj_es = NOMBRES_ES.get(nombre_obj, nombre_obj)
            conteo[nombre_obj_es] = conteo.get(nombre_obj_es, 0) + 1

    if conteo:
        partes_resumen = [f"{cantidad} {tipo}" for tipo, cantidad in conteo.items()]
        resumen = "Se detectaron en total: " + ", ".join(partes_resumen) + "."
        explicaciones.insert(0, resumen)

    if not explicaciones:
        explicaciones.append("No se detectó ningún vehículo ni peatón en la imagen.")

    if huvo_alerta:
        cv2.putText(imagen_cv, "Estado: pendiente de revision", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return imagen_cv, ("riesgo" if huvo_alerta else "normal"), explicaciones