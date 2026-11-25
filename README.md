# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Sistema completo de búsqueda por similitud, clasificación y detección de razas de perros: embeddings, base vectorial (PostgreSQL + pgvector), dos modelos entrenados, detector YOLO, una API asincrónica (FastAPI) y una interfaz web (Gradio) con una pestaña por etapa.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

**Integrantes**

| Alumna | Legajo |
|---|---|
| Jauregui, Lucía | J-0627/1 |
| Nebbia, María Sofía | N-1301/3 |

## Resumen del sistema

```text
Etapa 1  Imagen → ResNet18 (embedding de 512 dim., normalizado L2)
         → similitud coseno contra la galería (PostgreSQL + pgvector)
         → voto ponderado de los 10 vecinos → raza, o "unknown" si la similitud < 0,83

Etapa 2  Imagen → ResNet18 con fine-tuning (Modelo A) → raza y score (70 razas)

Etapa 3  Imagen → YOLOv8n (clase "dog", confianza ≥ 0,25) → recorte de cada perro
         → Modelo A → raza y score por perro
```

- **Embeddings de búsqueda:** Modelo A (`resnet18_finetuned`), 512 dimensiones. El baseline (ResNet18 de ImageNet) es el punto de comparación.
- **Clasificador:** Modelo A, ResNet18 con fine-tuning. El Modelo B (CNN propia) se compara y se descarta.
- **Detector:** YOLOv8n (`yolov8n.pt`), clase `dog` de COCO.
- **Umbral de similitud:** `SIMILARITY_THRESHOLD=0.83` (coseno), calibrado en `valid`.

## Modelos (link público de descarga)

Los modelos no se versionan en el repositorio (ver `.gitignore`). Se descargan desde este link público de solo lectura:

**https://drive.google.com/drive/folders/1jrQeT_6peakkLWBmGW-w4zAJCCtaJCTb**

Archivos que hay que copiar a la carpeta `models/` del repositorio:

- `resnet18_finetuned.pth` (Modelo A, usado para clasificar y para los embeddings)
- `cnn_custom.pth` (Modelo B, CNN propia)

La carpeta también incluye los historiales de entrenamiento (`*_history.json`), que no son necesarios para ejecutar la app. El detector YOLO (`yolov8n.pt`) se descarga solo en el primer uso.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── preprocessing.py     # Preprocesamiento compartido entre notebooks y app (PadToSquare, resize, normalización)
│       ├── schemas.py
│       ├── models/              # Arquitecturas de los modelos entrenados (Modelo A y Modelo B)
│       │   ├── cnn_custom.py
│       │   └── resnet18_classifier.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   #  Etapa 1
│       │   ├── classifier_service.py   #  Etapa 2
│       │   ├── detection_service.py    #  Etapa 3
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Descarga del dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configuración del modelo

1. Descargá `resnet18_finetuned.pth` y `cnn_custom.pth` desde el [link público](https://drive.google.com/drive/folders/1jrQeT_6peakkLWBmGW-w4zAJCCtaJCTb) y copialos a la carpeta `models/`.
2. Los `.env.*.example` ya vienen configurados para estos modelos (`EMBEDDING_MODEL=resnet18_finetuned`, `RESNET18_MODEL_NAME=resnet18_finetuned.pth`, `CNN_CUSTOM_MODEL_NAME=cnn_custom.pth`, `EMBEDDING_DIM=512`, `IMAGE_SIZE=224`, `SIMILARITY_THRESHOLD=0.83`, `YOLO_CONF_THRESHOLD=0.25`).

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 y la búsqueda con el modelo entrenado fallan hasta que existan.

La base vectorial guarda los embeddings de un solo modelo. La que se indexa con las instrucciones de abajo usa `resnet18_finetuned`: el selector de modelo de la pestaña de la Etapa 1 solo da resultados válidos con ese modelo. Para usar otro (baseline: umbral 0.84; `cnn_custom`), cambiá `EMBEDDING_MODEL` y `SIMILARITY_THRESHOLD` en el `.env` y reindexá.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

**Notas:**

- **La primera construcción (`docker compose build`) puede tardar mucho** (en nuestra máquina, más de una hora): descarga PyTorch con CUDA. Las siguientes son mucho más rápidas gracias a la caché de Docker.
- **Orden recomendado:** descargar el dataset, copiar los modelos a `models/`, `docker compose build`, `docker compose up -d` e indexar la galería. Indexar las 7.946 imágenes de `train` con CPU tarda unos 25 minutos.
- Si `docker compose up` falla con `container name "/backend" is already in use`, hay contenedores de otro proyecto con el mismo nombre: renombralos con `docker rename backend <otro-nombre>` (y lo mismo con `frontend`) y volvé a levantar.
- Si el backend falla en el primer arranque porque PostgreSQL todavía se está inicializando, volvé a correr `docker compose up -d`.
- Para comprobar el estado: `http://localhost:8000/health` debe mostrar `"embedding_model": "resnet18_finetuned"` y los dos checkpoints en `true`.
- Las credenciales de base de datos de `.env.docker.example` son **de ejemplo, solo para desarrollo local**.

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend Gradio con una pestaña por etapa.

### Implementado por el grupo

**Etapa 1 (`src/lib/services/similarity_service.py`):**
1. `extract_embedding`: convierte BGR a RGB, aplica el preprocesamiento compartido (padding a cuadrado, resize a 224, normalización de ImageNet), ejecuta ResNet18 de ImageNet sin la capa final y normaliza el embedding de 512 dimensiones con L2.
2. `search_similar_images`: con pgvector y métrica coseno busca en la base; en los otros casos calcula la similitud contra todos los registros y devuelve los `top_k` más parecidos, ordenados de mayor a menor.
3. `predict_breed_from_neighbors`: voto ponderado por la similitud de los vecinos. Si no hay vecinos o la mejor similitud es menor que `SIMILARITY_THRESHOLD`, devuelve `unknown`.

**Etapa 2 (`src/lib/services/classifier_service.py`):**
1. `train_classifier`: entrena el modelo activo con los splits de `data/splits.csv`, el mismo preprocesamiento que la app y data augmentation solo en `train`; guarda el checkpoint completo (con la lista de razas) y el historial por época.
2. `evaluate_classifier`: devuelve accuracy, precision, recall, specificity y F1 (macro) sobre `test`, y guarda la matriz de confusión y las predicciones para el análisis de errores. Acepta `split` y un conjunto de imágenes propio (se usa para el conjunto independiente).
3. `extract_custom_embedding`: embedding de la penúltima capa del modelo activo, con el mismo preprocesamiento.

**Etapa 3 (`src/lib/services/detection_service.py`):**
1. `detect_dogs`: YOLOv8n filtrando la clase `dog` con `YOLO_CONF_THRESHOLD`; devuelve las cajas ordenadas por confianza.
2. `classify_detected_dog`: clasifica cada recorte con el modelo activo (PyTorch u ONNX) y el mismo preprocesamiento.

**Archivos nuevos:** `src/lib/preprocessing.py` (preprocesamiento compartido entre las notebooks y la app) y `src/lib/models/` (arquitecturas del Modelo A y del Modelo B).


## Modelos y entrenamiento

Se compararon tres embeddings (baseline, Modelo A y Modelo B) y dos clasificadores (A y B) sobre los mismos conjuntos de datos (ver `2-train.ipynb`). Los checkpoints se pueden descargar desde el [link público de Drive](https://drive.google.com/drive/folders/1jrQeT_6peakkLWBmGW-w4zAJCCtaJCTb).

- **Baseline (Etapa 1):** ResNet18 preentrenada en ImageNet sin la capa final, embedding de 512 dimensiones. Es el punto de partida del fine-tuning.
- **Modelo A: ResNet18 con fine-tuning** (11,21 M de parámetros, embedding de 512 dimensiones). Se compararon *feature extraction* (solo la cabeza) y *fine-tuning* (toda la red, con tasa de aprendizaje diferencial: 1e-4 en el backbone y 1e-3 en la cabeza), y se eligió el fine-tuning por validación. AdamW (weight decay 1e-4), scheduler cosine, batch 64, dropout 0,3, label smoothing 0,1, pesos por clase inversamente proporcionales a la frecuencia y early stopping sobre la accuracy de `valid`.
- **Modelo B: CNN propia** (2,52 M de parámetros, embedding de 512 dimensiones), entrenada desde cero con lr 1e-3, hasta 40 épocas con early stopping (paciencia 8) y dropout 0,4. Arquitectura en `src/lib/models/cnn_custom.py`.
- **Preprocesamiento** (igual en las notebooks y en la app): padding a cuadrado con la media de ImageNet → resize a 224 × 224 → normalización de ImageNet.
- **Data augmentation** (solo en entrenamiento): recorte aleatorio (escala 0,8 a 1,0), volteo horizontal, rotación de hasta 15°, variaciones de color, desenfoque gaussiano y ruido (los dos últimos con probabilidad 0,3).
- **Detector:** YOLOv8n con `YOLO_CONF_THRESHOLD=0.25`, elegido en la sección 14 de `2-train.ipynb`.

**Etapa 1: búsqueda por similitud** (galería `train`, coseno, TOP_K = 10):

| Embeddings | NDCG@10 test | NDCG@10 independiente | Accuracy raza test / indep. |
|---|---|---|---|
| Baseline (ResNet18 ImageNet) | 0,873 | 0,759 | 93,7% / 81,0% |
| **Modelo A (fine-tuning)** | **0,949** | **0,784** | **96,8%** / 78,0% |
| Modelo B (CNN propia) | 0,714 | 0,398 | 79,7% / 44,0% |

**Etapa 2: clasificación** (precision, recall, specificity y F1 en promedio macro):

| Métrica | A: ResNet18 fine-tuning | B: CNN propia |
|---|---|---|
| Accuracy (test) | **95,99%** | 80,66% |
| Precision / Recall / F1 (test) | **0,963 / 0,960 / 0,959** | 0,820 / 0,807 / 0,805 |
| Specificity (test) | **0,999** | 0,997 |
| Accuracy (conjunto independiente) | **81,0%** | 48,0% |
| Parámetros / tamaño | 11,21 M / 44,9 MB | 2,52 M / 10,1 MB |
| CPU (ms/imagen, Colab de 2 núcleos) | ~62 | ~92 |

La matriz de confusión y las curvas de entrenamiento de cada modelo están en `2-train.ipynb`.

**Modelo elegido: A (ResNet18 con fine-tuning).** Es mejor en todas las métricas, generaliza mucho mejor a imágenes de otras fuentes y no cuesta más que el baseline. El Modelo B es 4,4 veces más chico, pero no es más rápido y rinde bastante peor. También se usa para los embeddings de búsqueda: mejora al baseline en `test` y queda empatado en el conjunto independiente (dentro del margen de error de ~±8 puntos con 100 imágenes).

**Umbral de similitud:** `τ = 0,83` (coseno), el mayor umbral que mantiene al menos 95% de cobertura en `valid` con los embeddings del Modelo A (con el baseline el umbral calibrado es 0,84). Con él, la accuracy cuando el sistema responde es de 96,8% y responde al 96,6% de las consultas de `valid`. Se calibró con `valid` y no con `test`.

**Comparación entre enfoques:** la clasificación es más simple y rápida (una sola pasada, sin galería) pero está atada a las 70 razas; la búsqueda por similitud permite agregar imágenes sin reentrenar, muestra ejemplos parecidos y puede responder "unknown" con el umbral. Con el conjunto independiente, la accuracy de A baja de 95,5% a 81,0% (misma comparación con las mismas 20 razas): es el efecto del cambio de dominio.

**Análisis de errores (FP/FN):** los errores de A se concentran en razas visualmente casi idénticas (Cockapoo/Labradoodle, Mex Hairless/American Hairless, Malinois/German Shepherd) y en posibles errores de etiquetado (Bulldog/French Bulldog). B comete muchos más errores y los concentra en razas con el mismo pelaje. Los ejemplos con imagen y score están en las secciones 8 y 10 de `2-train.ipynb`.

**Etapa 3:** se usa **YOLOv8n** preentrenado para detectar perros en la imagen (umbral de confianza **0,25**). Cada caja detectada se recorta y se clasifica con el modelo A (`resnet18_finetuned`), el mismo de la Etapa 2. El resultado devuelve las cajas con su raza y score, y la lista de razas detectadas.

Resultados de las pruebas manuales en la app:

| Caso | Resultado |
|---|---|
| Un perro | Detecta 1 perro (confianza de detección 0,28) y lo clasifica como American Spaniel |
| Varios perros | Detecta 2 perros de razas distintas y clasifica Golden Retriever y Dalmatian |
| Escena compleja (3 perros) | Detecta y clasifica Labrador y French Bulldog, pero **omite un perro** |
| Sin perros (paisaje) | Lista de detecciones vacía, no dibuja cajas |

**Limitación:** YOLOv8n es el modelo más chico de la familia y puede omitir perros en escenas con varios animales juntos. Subir el umbral perdería detecciones de baja confianza, y bajarlo podría agregar falsos positivos, por eso se dejó en 0,25.

## Dataset

- **Fuente:** [70 Dog Breeds Image Dataset](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set) (Kaggle, autor gpiosenka), 70 razas, en carpetas por raza y divididas en `train`, `valid` y `test`. La licencia está en la página del dataset.
- **Cantidad por raza:** 9.346 imágenes originales; después del filtrado quedan **8.869**: `train` 7.487 (entre 61 y 187 por raza, mediana 106,5), `valid` 684 (entre 8 y 10) y `test` 698 (entre 9 y 10). Las 70 razas están en los tres splits.
- **Variaciones:** pose, tamaño del perro en la imagen, iluminación, fondos y accesorios.
- **Reglas de filtrado y calidad** (`1-pre-processing.ipynb`, sección 7): se descartó el 5,1% del dataset (477 imágenes), contando cada imagen una sola vez.

  | Regla | Qué detecta | train | valid | test | Total |
  |---|---|---|---|---|---|
  | 1. Etiquetas inconsistentes | Una raza escrita de dos formas (71 → 70 razas) | 0 | 0 | 0 | 0 (se corrige) |
  | 2. Imágenes borrosas | Marca 46 / 7 / 6; se descartan solo las de `train` | 44 | 0 | 0 | 44 |
  | 3. Exposición extrema | Marca 16 / 0 / 1; falsos positivos (fondo blanco) | 0 | 0 | 0 | 0 (no se aplica) |
  | 4. Duplicados y fuga de datos | Copias entre splits y dentro de `train` (Hamming ≤ 8) | 394 | 14 | 0 | 408 |
  | 5. Etiqueta ambigua | La misma foto con dos razas distintas | 21 | 2 | 2 | 25 |
  | **Total descartadas** | | **459** | **16** | **2** | **477** |

  Después del filtrado no quedan casi-duplicados entre splits.
- **Splits:** los del dataset (`train`/`valid`/`test`), filtrados y guardados en `data/splits.csv` con rutas y razas. `train` es la galería de la Etapa 1 y los datos de entrenamiento; `valid` se usa para elegir hiperparámetros y umbrales; `test` se usa una sola vez al final.
- **Conjunto independiente de evaluación:** 110 imágenes tomadas de Wikimedia Commons (autor, licencia y link de origen de cada foto en `fuentes.csv`): 100 de un solo perro (20 razas, 5 por raza, incluidas razas fáciles de confundir) para las Etapas 1 y 2, y 10 para la Etapa 3 (5 con varios perros y 5 sin perros). Ninguna está en el dataset ni en la galería. Link de descarga: https://drive.google.com/drive/folders/1ATQaU1s605lqiCqibdgkFU8x7vDQJz-0?usp=sharing
- Los datos no se versionan (`data/dataset/` está en `.gitignore`).

## Problemas encontrados y soluciones

- **Umbral de similitud por modelo:** el valor por defecto (0.55) no sirve para ningún embedding; se calibró con `valid` para el baseline (0.84) y para el Modelo A (0.83).
- **Cambio de dominio:** las imágenes del conjunto independiente (otras fuentes, resoluciones y encuadres) hacen bajar las métricas; se cuantificó comparando con `test` restringido a las mismas 20 razas.
- **Ruido de etiquetas** (por ejemplo, Bulldog vs. French Bulldog): detectado en el análisis de errores.
- **Galería indexada:** `scripts/build_index.py` indexa toda la carpeta `train` (7.946 imágenes), mientras que la evaluación de `2-train.ipynb` usa la galería filtrada de `splits.csv` (7.487, sin las 459 descartadas). Afecta a pocas imágenes por raza y no cambia las conclusiones, pero las métricas del notebook corresponden a la galería filtrada.
- **Docker:** contenedores de otro proyecto con los mismos nombres (`backend`, `frontend`) impiden `docker compose up`; se resuelve renombrándolos. La primera construcción de la imagen es lenta.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido (512). Si lo cambian, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Se descargan del [link público](https://drive.google.com/drive/folders/1jrQeT_6peakkLWBmGW-w4zAJCCtaJCTb) y se colocan en `models/`.
- Los pesos preentrenados (torchvision y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.
- Notebooks: `1-pre-processing.ipynb` (dataset, filtrado, conjunto independiente, preprocesamiento y splits) y `2-train.ipynb` (evaluación de la búsqueda, entrenamiento de los modelos A y B, métricas, análisis de errores, comparación y checkpoints), ejecutadas y con sus salidas. 
- Notebooks ejecutadas y con sus salidas (Colab):
  - [1-pre-processing.ipynb](https://colab.research.google.com/github/luciajaureguiheisterborg-eng/tuia-dog-recognition-app/blob/feature%2Ftp2-razas-perros/1-pre-processing.ipynb)
  - [2-train.ipynb](https://colab.research.google.com/github/luciajaureguiheisterborg-eng/tuia-dog-recognition-app/blob/feature%2Ftp2-razas-perros/2-train.ipynb)