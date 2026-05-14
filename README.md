# Trabajo Práctico 3 - Machine Learning & Deep Learning (UdeSA)

Este repositorio contiene la implementación del Trabajo Práctico 3 de la materia *Machine Learning & Deep Learning* de UdeSA. El proyecto estudia clasificación multiclase sobre el dataset **EMNIST ByMerge**, comparando una red neuronal implementada desde cero con variantes construidas en **PyTorch**, además de evaluar búsqueda de hiperparámetros, regularización y robustez frente a ruido.

## Estructura

```text
.
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

### Descripción de carpetas y archivos

- `data/`: contiene los arreglos `NumPy` con las imágenes y etiquetas del dataset.
- `doc/`: incluye el enunciado del trabajo práctico.
- `notebooks/Entrega_TP3.ipynb`: notebook principal donde se ejecuta el flujo completo del TP, desde carga de datos hasta evaluación final.
- `src/models/custom/`: implementación propia de la red neuronal, capas densas, funciones de activación, función de pérdida y optimizadores.
- `src/models/torch/`: implementación del perceptrón multicapa equivalente usando PyTorch.
- `src/training/`: loops de entrenamiento y validación para modelos PyTorch.
- `src/evaluation/`: métricas, predicciones, búsqueda aleatoria de hiperparámetros y evaluación de robustez ante ruido gaussiano.
- `src/utils/`: utilidades de preprocesamiento, selección de configuraciones y visualización de resultados.

## Características Principales

- Implementación de una red neuronal secuencial desde cero con `NumPy`.
- Implementación equivalente de un MLP con `PyTorch`.
- Entrenamiento con:
  - gradient descent
  - mini-batch training
  - optimizador `ADAM`
  - early stopping
  - learning rate scheduling lineal y exponencial
  - regularización L2
  - label smoothing
- Búsqueda aleatoria de hiperparámetros para modelos custom y PyTorch.
- Evaluación con métricas de clasificación multiclase:
  - accuracy
  - macro F1 one-vs-all
  - matriz de confusión
- Análisis de robustez frente a ruido gaussiano en el conjunto de test.
- Visualización de curvas de entrenamiento, comparación de modelos e inspección de imágenes del dataset.

## Requisitos e Instalación

### Requisitos

- Python 3.11 o superior recomendado
- `pip`
- Jupyter Notebook o JupyterLab para ejecutar la entrega

Dependencias principales:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `tqdm`
- `torch`

### Instalación

1. Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd Kneipp_Mathias_TP3
```

2. Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

### Datos

El proyecto espera encontrar los siguientes archivos dentro de `data/`:

- `data/X_images.npy`
- `data/y_images.npy`

Por cuestiones de tamaño, el dataset **no fue subido a GitHub** junto con el repositorio. Para ejecutar la notebook o reutilizar los módulos, es necesario contar localmente con esos archivos dentro de la carpeta `data/`.

En esta entrega, el dataset ya está preparado en formato `NumPy`. Las imágenes tienen forma `28x28` y el problema contempla **47 clases** de `EMNIST ByMerge`.

## Uso

### Flujo principal

La forma principal de usar el proyecto es abrir y ejecutar la notebook:

```bash
jupyter notebook notebooks/Entrega_TP3.ipynb
```

La notebook realiza, en orden:

1. Carga de `X_images.npy` e `y_images.npy`.
2. Normalización de píxeles al rango `[0, 1]`.
3. Split de datos en train, validation y test.
4. Entrenamiento de modelos base y avanzados.
5. Búsqueda de hiperparámetros.
6. Evaluación comparativa de performance.
7. Análisis de robustez con ruido gaussiano.

### Modelos trabajados en la notebook

- `M0`: red neuronal básica implementada desde cero, usada como baseline.
- `M1`: red custom optimizada a partir de búsqueda aleatoria de hiperparámetros.
- `M2`: versión en PyTorch construida con la mejor configuración de `M1`.
- `M3`: modelo PyTorch optimizado con búsqueda propia de hiperparámetros.

### Uso desde módulos de Python

Si querés reutilizar componentes fuera de la notebook, podés importar directamente desde `src/`.

Ejemplo de preprocesamiento:

```python
import numpy as np
from src.utils.preprocessing import scaler, data_split

X = np.load("data/X_images.npy")
y = np.load("data/y_images.npy")

X = scaler(X)
X_dev, y_dev, X_test, y_test = data_split(X, y, frac=0.8)
X_train, y_train, X_val, y_val = data_split(X_dev, y_dev, frac=7/9)
```

Ejemplo de búsqueda aleatoria para el modelo custom:

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

### Observaciones

- El repositorio no expone actualmente una CLI propia; el punto de entrada real es la notebook.
- Para entrenar los modelos correctamente, las imágenes deben aplanarse antes de ingresar a las capas densas.
- Algunos experimentos pueden demandar bastante tiempo y memoria si se ejecutan sobre el dataset completo.
