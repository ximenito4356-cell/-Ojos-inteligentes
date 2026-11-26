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


def detectar_carretera_automatica(imagen_cv):
    """Traza un polígono aproximado de la zona de carretera/carril,
    usando bordes y líneas detectadas, sin depender de zonas.json."""
    alto, ancho = imagen_cv.shape[:2]

    gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    difuminado = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(difuminado, 50, 150)

    mascara = np.zeros_like(bordes)
    region = np.array([[
        (0, alto),
        (ancho, alto),
        (int(ancho * 0.9), int(alto * 0.45)),
        (int(ancho * 0.1), int(alto * 0.45)),
    ]], dtype=np.int32)
    cv2.fillPoly(mascara, region, 255)
    bordes_region = cv2.bitwise_and(bordes, mascara)

    lineas = cv2.HoughLinesP(bordes_region, 1, np.pi / 180, threshold=40,
                              minLineLength=40, maxLineGap=100)

    izquierdas, derechas = [], []
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea.flatten()
            if x2 == x1:
                continue
            pendiente = (y2 - y1) / (x2 - x1)
            if abs(pendiente) < 0.3:
                continue
            if pendiente < 0:
                izquierdas.append((x1, y1, x2, y2))
            else:
                derechas.append((x1, y1, x2, y2))

    def promediar(lista, lado):
        if not lista:
            if lado == "izq":
                return (int(ancho * 0.15), alto, int(ancho * 0.45), int(alto * 0.5))
            else:
                return (int(ancho * 0.85), alto, int(ancho * 0.55), int(alto * 0.5))
        xs = [p[0] for p in lista] + [p[2] for p in lista]
        ys = [p[1] for p in lista] + [p[3] for p in lista]
        indice_max_y = ys.index(max(ys))
        indice_min_y = ys.index(min(ys))
        return (xs[indice_max_y], max(ys), xs[indice_min_y], min(ys))

    xi1, yi1, xi2, yi2 = promediar(izquierdas, "izq")
    xd1, yd1, xd2, yd2 = promediar(derechas, "der")

    poligono_auto = np.array([
        [xi1, yi1],
        [xi2, yi2],
        [xd2, yd2],
        [xd1, yd1],
    ], dtype=np.int32)

    return poligono_auto


def procesar_imagen_array(imagen_cv, nombre_camara: str):
    if nombre_camara == "automatico":
        poligono = detectar_carretera_automatica(imagen_cv)
    else:
        poligono = np.array(ZONAS["camaras"][nombre_camara], dtype=np.int32)

    overlay = imagen_cv.copy()
    cv2.fillPoly(overlay, [poligono], color=(60, 60, 220))
    imagen_cv = cv2.addWeighted(overlay, 0.25, imagen_cv, 0.75, 0)
    cv2.polylines(imagen_cv, [poligono], isClosed=True, color=(0, 0, 255), thickness=2)

    resultados = modelo.predict(
        source=imagen_cv,
        conf=0.15,
        classes=[0, 1, 2, 3, 5, 7],
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