# ParkVision

Repositorio base para MVP de deteccion de plazas de parking con vision por computadora.

## Enfoque actual

Antes de tiempo real, vamos en modo offline con videos de prueba.

Objetivo inmediato:

- procesar 2 videos ya grabados
- validar deteccion y logica de ocupacion
- generar resultados por frame y resumen por video

## Que es ParkVision

ParkVision es un prototipo para ciudades y municipios que permite estimar en tiempo real la ocupacion de plazas de aparcamiento a partir de camaras publicas.

La idea principal es transformar un feed de video en informacion util para operacion urbana:

- cuantas plazas estan ocupadas
- cuantas plazas estan libres
- estado individual por plaza

Con esto se puede construir una base para cuadros de mando, alertas y futuras integraciones con apps ciudadanas.

## Problema que resuelve

En muchos aparcamientos publicos no existe una vision centralizada del nivel de ocupacion. Eso provoca:

- mas tiempo buscando plaza
- mas trafico innecesario en zonas de alta demanda
- menor capacidad de reaccion operativa

ParkVision propone una solucion ligera para empezar:

- usar deteccion de vehiculos con YOLOv8
- definir plazas fijas por poligonos
- exponer el estado por API
- visualizarlo en un dashboard simple

## Alcance del MVP

El MVP esta pensado para validar el enfoque tecnico, no para cubrir todos los casos complejos de produccion.

Incluye:

- ingesta desde camara o archivo de video
- deteccion de vehiculos por frame
- logica de ocupacion por plaza
- API REST para consultar estado actual
- dashboard para seguimiento basico

No incluye aun:

- persistencia historica de datos
- autenticacion/autorizacion
- multi-camara distribuida
- calibracion automatica de plazas

## Funcionalidades previstas

1. Ingesta de stream de video (`camera index` o fichero).
2. Deteccion de objetos con YOLOv8.
3. Filtrado de clases de vehiculo (car, truck, bus, motorcycle).
4. Evaluacion de ocupacion por plaza a partir de poligonos configurables.
5. Publicacion del estado en endpoint HTTP.
6. Visualizacion en dashboard con refresco periodico.

## Arquitectura (alto nivel)

- `cv_pipeline/`: procesamiento de video y logica de ocupacion.
- `backend/`: servicio FastAPI y contratos de datos.
- `dashboard/`: interfaz Streamlit para operacion MVP.
- `config/`: parametros y definicion de plazas.
- `scripts/`: comandos de arranque.

## Flujo de datos

1. Se captura un frame del stream.
2. YOLO detecta bounding boxes de vehiculos.
3. Se cruza cada deteccion con los poligonos de plazas.
4. Se calcula el estado ocupado/libre por plaza.
5. El backend guarda el ultimo estado en memoria.
6. API y dashboard consumen ese estado.

## Casos de uso iniciales

- Monitor de ocupacion para personal municipal.
- Demo tecnica para presentar viabilidad del sistema.
- Base de prueba para evolucionar a multi-camara.

## Roadmap corto

- Paso 3: pipeline CV basico.
- Paso 4: API FastAPI con endpoint de ocupacion.
- Paso 5: dashboard Streamlit.
- Paso 6: mejoras de robustez y observabilidad.

## Paso 3 - Pipeline CV Basico (implementado)

En este paso construimos el nucleo tecnico que transforma video en estado de ocupacion.

### Modulos creados en Paso 3

- `cv_pipeline/ingestion/video_source.py`
	- Se encarga de abrir la fuente de video (camara o archivo).
	- Va devolviendo `(frame, frame_index)` para que el resto del sistema procese secuencialmente.

- `cv_pipeline/detector/yolo_detector.py`
	- Carga el modelo YOLOv8 (`ultralytics`).
	- Ejecuta inferencia por frame.
	- Filtra solo clases de vehiculo: `car`, `truck`, `bus`, `motorcycle`.

- `cv_pipeline/parking/spot_manager.py`
	- Carga plazas desde `config/parking_spots.example.json`.
	- Evalua plaza por plaza si esta ocupada usando el centro del bounding box dentro del poligono.

- `cv_pipeline/pipeline.py`
	- Orquesta los 3 bloques anteriores.
	- Devuelve un estado por frame con:
		- timestamp
		- total/ocupadas/libres
		- estado por plaza
		- detecciones

- `scripts/run_pipeline_demo.py`
	- Script CLI para prueba rapida de una sola fuente.
	- Imprime resultados JSON por frame en consola.

- `scripts/run_offline_batch.py`
	- Script CLI para procesar uno o varios videos (incluyendo tus 2 videos de test).
	- Genera ficheros de salida en `outputs/offline/`:
		- `*_frames.jsonl` (estado por frame)
		- `*_summary.json` (resumen por video)
		- `batch_summary.json` (resumen global del lote)

- `scripts/select_yolo_roi.py`
	- Script interactivo para extraer el primer frame y seleccionar con raton la zona donde quieres aplicar YOLO.
	- Guarda la ROI en `config/yolo_roi.json`.

### Configuracion de plazas de ejemplo

Se actualizo `config/parking_spots.example.json` con 3 plazas (`A1`, `A2`, `A3`) para pruebas iniciales.

## Flujo completo y comandos concretos

Este flujo cubre todo el Paso 3 de punta a punta en Windows PowerShell: preparar entorno, definir zonas, ejecutar test y generar video anotado con YOLO.

### 1) Preparar entorno (solo primera vez)

Desde la raiz del repo (`C:\Users\Hugo\Downloads\ParkVision`):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Colocar videos de prueba

Ubica los videos en:

- `data/videos/test_1.mp4`
- `data/videos/test_2.mp4`

### 3) Dibujar plazas para cada video

Importante: cada camara/angulo necesita su propio archivo de zonas.

Para `test_1.mp4`:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\select_yolo_roi.py --video ".\data\videos\test_1.mp4" --output "config\yolo_roi_test1.json"
```

Para `test_2.mp4`:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\select_yolo_roi.py --video ".\data\videos\test_2.mp4" --output "config\yolo_roi_test2.json"
```

Controles del editor de zonas:

- click+drag en espacio vacio: crear nueva zona
- arrastrar vertice: ajustar zona
- `u` o `Ctrl+Z`: deshacer
- `r`: rehacer
- `d`: borrar zona activa
- `s` o `Enter`: guardar
- `q` o `Esc`: salir sin guardar

### 4) Ejecutar test offline + generar video YOLO anotado

`run_offline_batch.py` ahora genera tambien un MP4 anotado por video (`*_annotated.mp4`) con:

- bounding boxes YOLO
- confianza por deteccion
- zonas en verde/rojo (libre/ocupada)
- contador `Occ: ocupadas/total`

Ejecuta `test_1` con sus zonas:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_offline_batch.py --videos ".\data\videos\test_1.mp4" --spots "config\yolo_roi_test1.json" --max-frames 0
```

Comando directo para ejecutar solo `test1` (si ya tienes el entorno activo):

```powershell
python scripts\run_offline_batch.py --videos ".\data\videos\test_1.mp4" --spots "config\yolo_roi_test1.json" --max-frames 0
```

Ejecuta `test_2` con sus zonas:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_offline_batch.py --videos ".\data\videos\test_2.mp4" --spots "config\yolo_roi_test2.json" --max-frames 0
```

Si quieres correr ambos en una sola ejecucion (solo valido si comparten exactamente el mismo mapa de plazas):

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_offline_batch.py --videos ".\data\videos\test_1.mp4" ".\data\videos\test_2.mp4" --spots "config\yolo_roi_test1.json" --max-frames 0
```

### 5) Salidas generadas

En `outputs/offline/`:

- `test_1_frames.jsonl` o `test_2_frames.jsonl`: estado por frame
- `test_1_summary.json` o `test_2_summary.json`: resumen por video
- `batch_summary.json`: resumen global de la corrida
- `test_1_annotated.mp4` o `test_2_annotated.mp4`: video final con YOLO aplicado

### 6) Comandos de validacion rapida

Ver resumen global:

```powershell
Get-Content outputs\offline\batch_summary.json
```

Ver solo metricas clave (promedio y maximo ocupadas):

```powershell
$data = Get-Content outputs\offline\batch_summary.json | ConvertFrom-Json
$data | Select-Object video, frames_processed, avg_occupied_spots, max_occupied_spots
```

### 7) Opciones utiles

- Procesar solo una parte del video: `--max-frames 300`
- Cambiar modelo: `--model yolov8s.pt`
- Cambiar confianza: `--confidence 0.35`
- Desactivar render de video anotado: `--no-render-video`

Ejemplo corto (100 frames, sin video anotado):

```powershell
python scripts\run_offline_batch.py --videos ".\data\videos\test_1.mp4" --spots "config\yolo_roi_test1.json" --max-frames 100 --no-render-video
```

## Estado actual

- Paso 1 completado: estructura inicial del repositorio.
- Paso 2 completado: entorno virtual y dependencias instaladas.
- Paso 3 completado: pipeline CV + batch offline + export de video YOLO anotado.
- Paso 4 pendiente: API FastAPI.
- Paso 5 pendiente: dashboard Streamlit.
