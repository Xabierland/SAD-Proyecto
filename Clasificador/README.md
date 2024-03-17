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

## Uso

```bash
python clasificador.py --help
=== Clasificador ===
usage: clasificador.py [-h] -m MODE -f FILE -a ALGORITHM -p PREDICTION [-v] [--debug]

Practica de algoritmos de clasificación de datos.

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Modo de ejecución (train o test)
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
  -a ALGORITHM, --algorithm ALGORITHM
                        Algoritmo a ejecutar (kNN, decision_tree o random_forest)
  -p PREDICTION, --prediction PREDICTION
                        Columna a predecir (Nombre de la columna)
  -v, --verbose         Mostrar información adicional
  --debug               Modo debug
```

## Ejemplo

```bash
python clasificador.py -m train -a kNN -f iris.csv -p Especie -v --debug
```
