# Reto 2 · Ojos inteligentes: dataset para participantes

Fotos reales de cámaras de vigilancia de un **carril confinado de transporte público** (BRT). Solo el autobús autorizado puede circular por ese carril. Las fotos están anonimizadas: las placas y los rostros están difuminados, no hay datos impresos ni metadatos, y los nombres de archivo son neutros.

## Contenido

| Archivo | Qué es |
|---|---|
| `fotos/foto_001.jpg` … `foto_030.jpg` | 30 fotos de 1024 × 768 px: **15 normales** y **15 de riesgo** |
| `etiquetas.csv` | `archivo, camara, etiqueta` con `etiqueta` = `normal` o `riesgo` |
| `zonas.json` | Polígono de la **zona restringida** de cada cámara, en píxeles |
| `zonas/camara_X.jpg` | La misma zona dibujada en rojo sobre una foto de esa cámara |

Hay 4 cámaras (`camara_A` a `camara_D`), de día y de noche, con lluvia y sombras.

## Qué cuenta como riesgo

- **Riesgo**: dentro de la zona restringida hay un vehículo que no es el autobús autorizado (auto, moto, camioneta o camión) o un peatón.
- **Normal**: la zona está libre o solo circula por ella el autobús autorizado. Los vehículos que van por fuera del polígono, por ejemplo en el carril contiguo, no cuentan.

## Qué se espera

Por cada foto, una salida **normal / riesgo**. Ante un riesgo, una **alerta** con:

- la foto marcada (la zona y el objeto que la invade);
- el estado **«pendiente de revisión»**, porque una persona confirma la alerta.

## Cómo se evalúa

El jurado guarda **10 fotos balanceadas** (5 normales y 5 de riesgo) de las mismas cámaras, que no se entregan. Se cuentan **aciertos, falsos positivos y falsos negativos** sobre esas 10.

## Coordenadas de `zonas.json`

`[x, y]` en píxeles. El origen está en la esquina superior izquierda; `x` crece hacia la derecha y `y` hacia abajo. Una forma sencilla de decidir si algo está dentro de la zona es tomar el punto central de la base de su recuadro, que es donde toca el piso.

## Uso permitido

Las fotos se usan solo durante el hackathon y se borran al terminar. No se publican ni se intenta revertir la anonimización.
