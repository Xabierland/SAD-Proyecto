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

5. Instalar tablau

```bash

```

## Ayuda

```bash
python clustering.py --help
=== Clustering ===
usage: clustering.py [-h] -m MODE -f FILE -a ALGORITHM -p PREDICTION [-e ESTIMATOR] [-c CPU] [-v] [--debug]

Practica de algoritmos de clusterificacion de datos.

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Modo de ejecución (train o test)
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
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
python clustering.py -m train -a kNN -f iris.csv -p Especie
```

Avanzado

```bash
python clustering.py -m train -a kNN -f iris.csv -p Especie -e accuracy -c 4 -v --debug
```

## JSON

```json
{
    "preprocessing": {
        "unique_category_threshold": 51,
        "drop_features": [],
        "missing_values": "impute",
        "impute_strategy": "mean",
        "scaling": "standard",
        "text_process": "tf-idf",
        "sampling": "undersampling"
    },
    "lda": {
        "covariance_estimator": ["svd"],
        "n_components": [10],
        "shrinkage": ["auto"]
    }
}
```
