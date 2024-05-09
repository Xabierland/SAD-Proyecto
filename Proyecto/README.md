<!-- markdownlint-disable MD024 -->
<!-- markdownlint-disable MD029 -->
# Manual del Uso

## Requerimientos

- Python 3.8
- pip
- anaconda
  - conda
  - anaconda-navigator
    - tablau

## Instalación

1. Instalar [anaconda](https://github.com/Xabierland/SAD/blob/main/INSTALACIONES/Instalacion.md#anaconda)
2. Crear un entorno virtual con conda

```bash
conda create -n sad python=3.8
```

3. Activar el entorno virtual

```bash
conda activate sad
```

4. Instalar las dependencias con pip

```bash
pip install -r requirements.txt
```

## Ayuda

Clasificador

```bash
python clasificador.py --help
=== Clasificador ===
usage: clasificador.py [-h] -m MODE -f FILE -a ALGORITHM -p PREDICTION [-e ESTIMATOR] [-c CPU]
                       [-v] [--debug]

Practica de algoritmos de clasificación de datos.

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Modo de ejecución (train o test)
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
  -a ALGORITHM, --algorithm ALGORITHM
                        Algoritmo a utilizar (kNN, decision_tree, random_forest, naive_bayes)
  -p PREDICTION, --prediction PREDICTION
                        Columna a predecir (Nombre de la columna)
  -e ESTIMATOR, --estimator ESTIMATOR
                        Estimador a utilizar para elegir el mejor modelo https://scikit-
                        learn.org/stable/modules/model_evaluation.html#scoring-parameter
  -c CPU, --cpu CPU     Número de CPUs a utilizar [-1 para usar todos]
  -v, --verbose         Muestra las metricas por la terminal
  --debug               Modo debug [Muestra informacion extra del preprocesado y almacena el
                        resultado del mismo en un .csv]
```

Clustering

```bash
=== Clustering ===
usage: clustering.py [-h] -f FILE [-v] [--debug]

Practica de algoritmos de clustering de datos.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
  -v, --verbose         Muestra las metricas por la terminal
  --debug               Modo debug [Muestra informacion extra del preprocesado y almacena el
                        resultado del mismo en un .csv]
```

## Uso

### Datos

Preprocesado y union de los datos

```bash
python Datos/datos.py
```

### Clasificador

Basico

```bash
python clasificador.py -m train -a naive_bayes -f Datos.csv -p "Overall Rating"
```

Avanzado

```bash
python clasificador.py -m train -a naive_bayes -f Datos.csv -p "Overall Rating" -e accuracy -c 4 -v --debug
```

### Clustering

Basico

```bash
python clustering.py -f Datos.csv
```

Avanzado

```bash
python clustering.py -f AirlinesTrain.csv -v --debug
```

## JSON

### Clasificador

```json
{
    "preprocessing": {
        "unique_category_threshold": 51,
        "drop_features": [],
        "missing_values": "impute",
        "impute_strategy": "mean",
        "scaling": "minmax",
        "text_process": "tf-idf",
        "sampling": "SMOTE"
    },
    "kNN": {
        "n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2]
    },
    "decision_tree": {
        "criterion": ["gini"],
        "max_depth": [5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "splitter": ["best"]
    },
    "random_forest": {
        "n_estimators": [50],
        "criterion": ["gini"],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [false]
    },
    "naive_bayes": {
        "alpha": [0.00000001,0.0001,0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
        "fit_prior": [true, false]
        }
}
```

### Clustering

```json
{
    "preprocessing": {
        "unique_category_threshold": 51,
        "drop_features": ["Name", "Airline", "Verified", "Date", "Type of Traveller", "Route", "Class", "Seat Comfort","Staff Service","Food & Beverages", "Inflight Entertainment", "Value For Money", "Overall Rating", "Numerical Overall Rating"],
        "missing_values": "drop",
        "impute_strategy": "",
        "text_process": "tf-idf"
    },
    "lda":{
        "num_topics" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        "passes" : [50],
        "iterations" : [100]
    },
    "nmf":{
        "num_topics" : [5, 10, 15, 20],
        "max_iter" : [1000],
        "alpha" : [0.1],
        "l1_ratio" : [0.5]
    }
}
```
