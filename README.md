# Monitoreo de aves urbanas de Lima con visión computacional

Aplicación web que reconoce a partir de una foto 12 especies de aves urbanas de Lima
y permite registrar observaciones por parque junto con las características del espacio
(tema 22 del Anexo 01: monitoreo de aves urbanas).

## Cómo funciona

1. `01_descargar_dataset.py` descarga fotos de iNaturalist (grado de investigación, licencias
   Creative Commons) de cada especie, primero de Lima y, si faltan, del resto del Perú.
   Guarda `metadatos.csv` con autor, licencia y origen de cada foto.
2. `02_entrenar_modelo.py` entrena un clasificador con transfer learning (MobileNetV2 preentrenado
   en ImageNet): primero la cabeza clasificadora y luego ajuste fino de las últimas 30 capas.
   Evalúa en un conjunto de prueba separado (70/15/15 estratificado) y genera precisión, top-3,
   reporte por clase, matriz de confusión y curvas de entrenamiento.
3. `web/` contiene la página que carga el modelo en el navegador con TensorFlow.js
   (las fotos no se envían a ningún servidor).

## Requisitos

- Python 3.10+ y una cuenta de Google para Colab (GPU gratuita).
- Para la descarga local: `pip install requests pillow`

## Pasos de ejecución

### 1. Descargar el dataset (en tu computadora o en Colab)
```bash
python 01_descargar_dataset.py --por-especie 300 --salida data
```
Revisa en la consola cuántas fotos salieron de Lima y cuántas del resto del Perú, y revisa los
avisos de nombres científicos. Si una especie queda con muy pocas fotos, redúcela o cámbiala en
la lista `ESPECIES`. Borra manualmente fotos claramente erróneas (sin ave, jaulas, etc.).

### 2. Entrenar (Google Colab, con GPU)
Sube `data/` (o descárgala dentro de Colab) y ejecuta:
```
!pip install -q tf_keras tensorflowjs scikit-learn matplotlib
!python 02_entrenar_modelo.py --datos data --salida salida
```

### 3. Convertir a TensorFlow.js
```bash
tensorflowjs_converter --input_format=keras salida/modelo_aves_lima.h5 web/model
cp salida/class_names.json web/model/
```
Si la conversión falla por versiones de Keras, usa `pip install tf_keras` y
`export TF_USE_LEGACY_KERAS=1` antes de entrenar y convertir.

### 4. Abrir la página
```bash
cd web
python -m http.server 8000
```
Abre http://localhost:8000. No funciona con doble clic porque el navegador bloquea la carga
del modelo desde archivos locales. Para publicarla gratis, sube la carpeta `web/` (con `model/`)
a GitHub Pages o Netlify.

## Estructura
```
01_descargar_dataset.py
02_entrenar_modelo.py
web/
  index.html
  especies.json        fichas de cada especie (edítalas y cita fuentes en el informe)
  model/               (se genera en el paso 3)
```

## Nota sobre las especies
Las fichas de `especies.json` son un punto de partida general. Verifica cada dato con fuentes
confiables (por ejemplo eBird, guías de aves del Perú o artículos de Scopus) antes de usarlos
en el informe y cítalos en APA 7 o IEEE.

## Declaración de uso de IA (exigida por la rúbrica)
El código de la aplicación fue desarrollado con asistencia de Claude (Anthropic). Esto debe
indicarse expresamente en la introducción del informe.
