# Manual de Uso

## Requerimientos

- Python 3.8
- pip
- conda

## Instalación

1. Clonar el repositorio
2. Crear un entorno virtual con conda
3. Instalar las dependencias con pip

```bash
git clone https://github.com/Xabierland/SAD-Proyecto.git
cd SAD-Proyecto/Clasificador
conda create -n sad python=3.8
conda activate sad
pip install -r requirements.txt
```

## Ayuda

```bash
python clasificador.py --help
=== Clasificador ===
usage: clasificador.py [-h] -m MODE -f FILE -a ALGORITHM -p PREDICTION [-e ESTIMATOR] [-c CPU] [-v] [--debug]

Practica de algoritmos de clasificación de datos.

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Modo de ejecución (train o test)
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
  -a ALGORITHM, --algorithm ALGORITHM
                        Algoritmo a ejecutar (kNN, decision_tree o random_forest)
  -p PREDICTION, --prediction PREDICTION
                        Columna a predecir (Nombre de la columna)
  -e ESTIMATOR, --estimator ESTIMATOR
                        Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
  -c CPU, --cpu CPU     Número de CPUs a utilizar [-1 para usar todos]
  -v, --verbose         Muestra las metricas por la termina
  --debug               Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]
```

## Uso

Basico

```bash
python clasificador.py -m train -a kNN -f iris.csv -p Especie
```

Avanzado

```bash
python clasificador.py -m train -a kNN -f iris.csv -p Especie -e accuracy -c 4 -v --debug
```

## JSON

```json
{
    "preprocessing": {
        "unique_category_threshold": 50,      // Numero de apariciones unicas para considerar una columna como categorica (int)
        "drop_features": [],                  // Columnas a eliminar (lista de strings)
        "missing_values": "impute",           // Estrategia para tratar los valores nulos (impute, drop)
        "impute_strategy": "mean",            // Estrategia para imputar los valores nulos (mean, median, most_frequent)
        "scaling": "minmax",                  // Estrategia para escalar los valores (minmax, normalizer, maxabs, standard)
        "text_process": "tf-idf",             // Estrategia para procesar el texto (tf-idf, bow)
        "sampling": "oversampling"            // Estrategia para tratar el desbalanceo de clases (oversampling, undersampling)
    },
    "kNN": {
        "n_neighbors": [3, 5, 7],             // Numero de vecinos (lista de enteros)
        "weights": ["uniform", "distance"],   // Peso de los vecinos (uniform, distance)
        "algorithm": ["auto"],                // Algoritmo para calcular los vecinos (auto, ball_tree, kd_tree, brute)
        "leaf_size": [30],                    // Tamaño de la hoja (lista de enteros)
        "p": [2]                              // Parametro de la distancia (1 para manhattan, 2 para euclidean)
    },
    "decision_tree": {
        "criterion": ["gini"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max_depth": [5, 10, 20, 30],         // Profundidad maxima del arbol (lista de enteros)
        "min_samples_split": [2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min_samples_leaf": [1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max_features": ["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)
        "splitter": ["best"]                  // Estrategia para elegir la particion (best, random)
    },
    "random_forest": {
        "n_estimators": [50],                 // Numero de arboles (lista de enteros)
        "criterion": ["gini"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max_depth": [5, 10],                 // Profundidad maxima del arbol (lista de enteros)
        "min_samples_split": [2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min_samples_leaf": [1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max_features": ["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)  
        "bootstrap": [false]                  // Si se deben usar muestras bootstrap (true, false)
    }
}
```
