# ParkVision

MVP para detectar y reportar la ocupacion de plazas de aparcamiento usando YOLOv8 + vision por computadora.

Flujo: video de camara → deteccion de vehiculos → evaluacion de plazas → estado JSON → API → dashboard.

---

## Estado del proyecto

- [x] Paso 1: estructura del repositorio.
- [x] Paso 2: entorno virtual y dependencias.
- [x] Paso 3: pipeline CV + batch offline + video anotado + CSV timeline + estadisticas por plaza.
- [ ] Paso 4: API FastAPI (`GET /occupancy`, `GET /health`, servicio de background).
- [ ] Paso 5: dashboard Streamlit (mapa de plazas, metricas en tiempo real).
- [ ] Paso 6: persistencia SQLite + graficas de tendencia + API key.

---

## Archivos cruciales

Estos archivos son el nucleo del proyecto. **No borrar nunca.**

| Archivo / Carpeta | Para que sirve |
|---|---|
| `cv_pipeline/pipeline.py` | Orquestador principal: une ingesta + YOLO + plazas |
| `cv_pipeline/detector/yolo_detector.py` | Inferencia YOLOv8, filtra solo vehiculos |
| `cv_pipeline/ingestion/video_source.py` | Lee frames de camara o archivo MP4 |
| `cv_pipeline/parking/spot_manager.py` | Evalua si cada plaza esta ocupada por poligono |
| `scripts/run_offline_batch.py` | Script principal de batch: procesa videos y genera salidas |
| `scripts/select_yolo_roi.py` | Editor interactivo para dibujar las zonas de parking |
| `scripts/run_pipeline_demo.py` | Debug rapido: imprime JSON por frame en consola |
| `config/yolo_roi_test1.json` | Zonas dibujadas para test_1.mp4 (tarda en rehacerse) |
| `config/yolo_roi_test2.json` | Zonas dibujadas para test_2.mp4 (tarda en rehacerse) |
| `data/videos/test_1.mp4` | Video fuente de prueba 1 |
| `data/videos/test_2.mp4` | Video fuente de prueba 2 |
| `yolov8n.pt` | Pesos del modelo YOLO (se descarga automaticamente si falta) |
| `requirements.txt` | Dependencias del proyecto |

---

## Que puedes borrar entre test y test

Todos los archivos de `outputs/offline/` son **regenerados automaticamente** en cada ejecucion de batch. Puedes borrarlos libremente para empezar limpio.

```
outputs/offline/
├── *_frames.jsonl       <- borrable (regenerado)
├── *_summary.json       <- borrable (regenerado)
├── *_timeline.csv       <- borrable (regenerado)
├── *_annotated.mp4      <- borrable (regenerado, el mas pesado)
└── batch_summary.json   <- borrable (regenerado)
```

**Nunca borres** `config/yolo_roi_test1.json` ni `config/yolo_roi_test2.json`. Dibujar las zonas lleva tiempo y no se regeneran solas.

Comando para limpiar salidas antes de un test nuevo:

```powershell
Remove-Item outputs\offline\* -Recurse -Force
```

---

## Guia rapida de uso

### 1) Preparar entorno (solo la primera vez)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Dibujar zonas de parking (solo si no existen ya)

Cada video/angulo necesita su propio archivo de zonas. Si `config/yolo_roi_test1.json` ya existe, salta este paso.

```powershell
# Zonas para test_1
python scripts\select_yolo_roi.py --video ".\data\videos\test_1.mp4" --output "config\yolo_roi_test1.json"

# Zonas para test_2
python scripts\select_yolo_roi.py --video ".\data\videos\test_2.mp4" --output "config\yolo_roi_test2.json"
```

Opcionalmente puedes abrir el editor en otro frame:

```powershell
python scripts\select_yolo_roi.py --video ".\data\videos\test_1.mp4" --output "config\yolo_roi_test1.json" --start-frame 250
```

Controles del editor:

| Accion | Tecla / Gesto |
|---|---|
| Crear zona | click + drag en espacio vacio |
| Ajustar vertice | arrastrar vertice |
| Reproducir / pausar | `Espacio` |
| Mover video atras / delante | flecha izquierda / derecha o `n` / `m` |
| Salto grande | `PgUp` / `PgDn` |
| Ir al inicio / final | `Inicio` / `Fin` |
| Deshacer | `u` o `Ctrl+Z` |
| Rehacer | `r` |
| Borrar zona activa | `d` |
| Guardar | `s` o `Enter` |
| Salir sin guardar | `q` o `Esc` |

### 3) Ejecutar batch

```powershell
# Solo test_1
python scripts\run_offline_batch.py --videos ".\data\videos\test_1.mp4" --spots "config\yolo_roi_test1.json"

# Solo test_2
python scripts\run_offline_batch.py --videos ".\data\videos\test_2.mp4" --spots "config\yolo_roi_test2.json"
```

Opciones utiles:

| Flag | Efecto |
|---|---|
| `--max-frames 300` | Procesa solo los primeros N frames (util para pruebas rapidas) |
| `--confidence 0.4` | Cambia el umbral de confianza YOLO (default: 0.35) |
| `--model yolov8s.pt` | Usa un modelo YOLO mas preciso (mas lento) |
| `--no-render-video` | No genera el MP4 anotado (mucho mas rapido) |

### 4) Salidas generadas

En `outputs/offline/` por cada video:

| Archivo | Contenido |
|---|---|
| `*_frames.jsonl` | Estado completo por frame: timestamp, ocupadas, libres, detecciones |
| `*_summary.json` | Resumen: avg/max/min globales + `occupancy_rate_pct` y `state_changes` por plaza |
| `*_timeline.csv` | Matriz `frame x plaza` con 0/1 — abre directamente en Excel |
| `*_annotated.mp4` | Video con bounding boxes YOLO, zonas verde/rojo y contador de ocupacion |
| `batch_summary.json` | Resumen global de todos los videos procesados |

### 5) Ver resultados rapido

```powershell
# Resumen global
Get-Content outputs\offline\batch_summary.json

# Solo metricas clave
$data = Get-Content outputs\offline\batch_summary.json | ConvertFrom-Json
$data | Select-Object video, frames_processed, avg_occupied_spots, max_occupied_spots
```

---

## Arquitectura

```
cv_pipeline/        Pipeline de vision: ingesta -> deteccion -> ocupacion
backend/            API FastAPI (pendiente Paso 4)
dashboard/          Interfaz Streamlit (pendiente Paso 5)
config/             Zonas de parking por video (.json)
scripts/            Scripts de ejecucion y utilidades
data/videos/        Videos fuente de prueba
outputs/offline/    Resultados generados por batch (todos borrables)
```
