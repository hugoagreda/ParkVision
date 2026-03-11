# ParkVision

MVP de vision por computadora para detectar ocupacion de plazas de aparcamiento desde camaras publicas.

## Objetivo

Detectar vehiculos con YOLOv8, evaluar plazas fijas y exponer disponibilidad por API y dashboard.

## Estructura del Repositorio

```text
ParkVision/
	backend/
		api/
			routes/
				occupancy.py
			main.py
		core/
			config.py
		models/
			schemas.py
		services/
			occupancy_service.py
	config/
		parking_spots.example.json
		settings.yaml
	cv_pipeline/
		detector/
			yolo_detector.py
		ingestion/
			video_source.py
		parking/
			spot_manager.py
		pipeline.py
	dashboard/
		app.py
	scripts/
		run_api.py
		run_dashboard.py
	utils/
		logging.py
	.env.example
	.gitignore
	Makefile
	pyproject.toml
	requirements.txt
	README.md
```

## Modulos

- `cv_pipeline/ingestion/video_source.py`: lectura de stream desde camara o archivo.
- `cv_pipeline/detector/yolo_detector.py`: carga de YOLOv8 (Ultralytics) e inferencia por frame.
- `cv_pipeline/parking/spot_manager.py`: logica de ocupacion por plaza basada en poligonos fijos.
- `cv_pipeline/pipeline.py`: orquestacion de ingest, deteccion y evaluacion de plazas.
- `backend/services/occupancy_service.py`: loop de procesamiento en segundo plano y estado actual.
- `backend/api/routes/occupancy.py`: endpoint para consultar estado de ocupacion.
- `backend/api/main.py`: app FastAPI y arranque del servicio.
- `dashboard/app.py`: interfaz Streamlit para visualizar plazas ocupadas/libres.
- `backend/core/config.py`: configuracion central via variables de entorno.
- `backend/models/schemas.py`: contratos de respuesta para API.
- `config/parking_spots.example.json`: ejemplo de plazas definidas por poligonos.

## Flujo MVP

1. `VideoSource` captura frames.
2. `YoloDetector` detecta vehiculos (`car`, `truck`, `bus`, `motorcycle`).
3. `SpotManager` marca cada plaza como ocupada/libre segun el centro del bounding box.
4. `OccupancyService` mantiene el ultimo estado en memoria.
5. FastAPI expone `GET /occupancy/latest`.
6. Streamlit consulta el endpoint y muestra disponibilidad.

## Instalacion

Comandos para crear y activar el entorno virtual en Windows.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ritmo Global (Comandos End-to-End)

PowerShell (flujo recomendado):

```powershell
cd C:\Users\Hugo\Downloads\ParkVision
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\run_api.py
```

En una segunda terminal PowerShell:

```powershell
cd C:\Users\Hugo\Downloads\ParkVision
.\.venv\Scripts\Activate.ps1
python scripts\run_dashboard.py
```

CMD (equivalente):

```bat
cd /d C:\Users\Hugo\Downloads\ParkVision
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
python scripts\run_api.py
```

En una segunda terminal CMD:

```bat
cd /d C:\Users\Hugo\Downloads\ParkVision
.venv\Scripts\activate.bat
python scripts\run_dashboard.py
```

## Configuracion

1. Copiar `.env.example` a `.env`.
2. Ajustar `VIDEO_SOURCE` (camara `0` o ruta de video).
3. Ajustar `SPOT_CONFIG_PATH` si usas otro archivo de plazas.

Variables relevantes:

- `VIDEO_SOURCE`
- `YOLO_MODEL_PATH`
- `YOLO_CONFIDENCE`
- `SPOT_CONFIG_PATH`
- `API_HOST`
- `API_PORT`
- `DASHBOARD_API_BASE_URL`

## Ejecucion

API:

```bash
python scripts/run_api.py
```

Dashboard:

```bash
python scripts/run_dashboard.py
```

Tambien puedes usar:

```bash
make api
make dashboard
```

## Endpoints

- `GET /health`
- `GET /occupancy/latest`

Respuesta ejemplo de `GET /occupancy/latest`:

```json
{
	"timestamp": "2026-03-11T10:00:00+00:00",
	"frame_index": 42,
	"total_spots": 3,
	"occupied_spots": 2,
	"free_spots": 1,
	"spots": [
		{"spot_id": "A1", "occupied": true},
		{"spot_id": "A2", "occupied": false},
		{"spot_id": "A3", "occupied": true}
	],
	"detections": []
}
```

## Notas de Produccion (Siguiente Etapa)

- Persistencia de historico (PostgreSQL/Timescale).
- Cola/event bus para multiples camaras.
- Calibracion robusta de plazas con herramientas de anotacion.
- Metrics y observabilidad (Prometheus + Grafana).
