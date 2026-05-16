# Trabajo Práctico 3 - Machine Learning & Deep Learning

Este repositorio contiene la resolución del Trabajo Práctico 3 de la materia *Machine Learning & Deep Learning* de UdeSA. El objetivo es resolver un problema de clasificación multiclase sobre **EMNIST ByMerge** y comparar dos enfoques:

- una red neuronal implementada desde cero con `NumPy`
- una versión equivalente construida con `PyTorch`

Además del entrenamiento base, el proyecto incluye búsqueda aleatoria de hiperparámetros, métricas de evaluación, visualizaciones y un análisis de robustez frente a ruido gaussiano.

## Estructura del proyecto

```text
Kneipp_Mathias_TP3/
├── data/
│   ├── X_images.npy
│   └── y_images.npy
├── doc/
│   └── I302__Machine_Learning_TP03.pdf
├── notebooks/
│   └── Entrega_TP3.ipynb
├── src/
│   ├── evaluation/
│   │   ├── grid_search.py
│   │   ├── metrics.py
│   │   ├── noise_robustness.py
│   │   └── predictions.py
│   ├── models/
│   │   ├── custom/
│   │   │   ├── activations.py
│   │   │   ├── layers.py
│   │   │   ├── loss.py
│   │   │   ├── neural_network.py
│   │   │   └── optimizers.py
│   │   └── torch/
│   │       └── mlp.py
│   ├── training/
│   │   └── train.py
│   └── utils/
│       ├── plotting.py
│       ├── preprocessing.py
│       └── utils.py
├── requirements.txt
└── README.md
```

## Descripción de carpetas

- `data/`: archivos `NumPy` con imágenes y etiquetas.
- `doc/`: enunciado del trabajo práctico.
- `notebooks/Entrega_TP3.ipynb`: notebook principal de la entrega.
- `src/models/custom/`: implementación propia de la red neuronal, activaciones, capas, pérdida y optimizador.
- `src/models/torch/`: implementación del MLP en `PyTorch`.
- `src/training/`: loops de entrenamiento y validación para modelos `PyTorch`.
- `src/evaluation/`: métricas, predicciones, búsqueda aleatoria y robustez frente a ruido.
- `src/utils/`: preprocesamiento, selección de configuraciones y visualización.

## Funcionalidades principales

- Red neuronal secuencial implementada desde cero con `NumPy`.
- MLP equivalente implementado con `PyTorch`.
- Entrenamiento con mini-batches, `ADAM`, early stopping y learning rate scheduling.
- Regularización L2 y label smoothing.
- Búsqueda aleatoria de hiperparámetros para modelos custom y `PyTorch`.
- Evaluación con accuracy, macro F1 one-vs-all y matriz de confusión.
- Análisis de robustez ante ruido gaussiano.
- Visualizaciones de imágenes, curvas de entrenamiento y comparación de modelos.

## Requisitos

- Python 3.11 o superior
- `pip`
- Jupyter Notebook o JupyterLab

Dependencias principales:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `tqdm`
- `torch`

## Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd Kneipp_Mathias_TP3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

El proyecto espera encontrar estos archivos dentro de `data/`:

- `data/X_images.npy`
- `data/y_images.npy`

En esta entrega, el dataset se usa en formato `NumPy`. Las imágenes son de `28x28` píxeles y el problema tiene **47 clases** correspondientes a `EMNIST ByMerge`.

## Uso

### Flujo principal

La forma principal de ejecutar el trabajo es abrir la notebook:

```bash
jupyter notebook notebooks/Entrega_TP3.ipynb
```

La notebook recorre el flujo completo:

1. carga de datos
2. escalado de píxeles a `[0, 1]`
3. partición estratificada en train, validation y test
4. entrenamiento de modelos base
5. búsqueda de hiperparámetros
6. evaluación comparativa
7. análisis de robustez con ruido gaussiano

### Modelos trabajados

- `M0`: baseline custom implementado desde cero.
- `M1`: modelo custom ajustado con búsqueda aleatoria.
- `M2`: versión en `PyTorch` basada en la mejor configuración custom.
- `M3`: modelo `PyTorch` ajustado con búsqueda propia.

## Uso desde módulos

### Preprocesamiento

```python
import numpy as np

from src.utils.preprocessing import scaler, stratified_split

X = np.load("data/X_images.npy")
y = np.load("data/y_images.npy")

X = scaler(X)

X_dev, X_test, y_dev, y_test = stratified_split(X, y, frac=0.8, seed=42)
X_train, X_val, y_train, y_val = stratified_split(
    X_dev, y_dev, frac=7/9, seed=42
)
```

### Búsqueda aleatoria para el modelo custom

```python
from src.evaluation.grid_search import random_grid_search_custom

models, configs = random_grid_search_custom(
    input_dim=28 * 28,
    output_dim=47,
    X_train=X_train.reshape(len(X_train), -1),
    y_train=y_train,
    X_val=X_val.reshape(len(X_val), -1),
    y_val=y_val,
    epochs=50,
    K_models=5,
    possible_configs=possible_configs,
    early_stopping=3,
)
```

### Evaluación de un modelo

```python
from src.utils.plotting import evaluate_model

evaluate_model(
    model=models[0],
    X_train=X_train.reshape(len(X_train), -1),
    y_train=y_train,
    X_val=X_val.reshape(len(X_val), -1),
    y_val=y_val,
    dataset_name="emnist_bymerge",
    val_name="Validation",
)
```

## Observaciones

- El punto de entrada principal del proyecto es la notebook; no hay una CLI dedicada.
- Los modelos densos esperan inputs aplanados al momento de entrenar o evaluar.
- Algunos experimentos pueden tardar bastante si se ejecutan sobre el dataset completo.
