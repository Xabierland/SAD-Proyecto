# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña y Ibai Sologestoa.
Script para la implementación del algoritmo de clasificación
"""

import sys
import signal
import argparse
import pandas as pd
import string
import pickle
import time
import json
import csv
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Funciones auxiliares

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-v", "--verbose", help="Mostrar información adicional", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug", required=False, default=False, action="store_true")
    
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('clasificador.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print("Datos cargados con éxito")
        return data
    except Exception as e:
        print("Error al cargar los datos")
        print(e)
        sys.exit(1)

# Funciones para calcular métricas

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_classification_report(y_test, y_pred):
    """
    Función para calcular el informe de clasificación
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Informe de clasificación
    """
    report = classification_report(y_test, y_pred, zero_division=0)
    return report

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    cm = confusion_matrix(y_test, y_pred)
    return cm

# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        
        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print("Datos separados con éxito")
        
        if args.debug:
            print("> Columnas numéricas:\n", numerical_feature.columns)
            print("> Columnas de texto:\n", text_feature.columns)
            print("> Columnas categóricas:\n", categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print("Error al separar los datos")
        print(e)
        sys.exit(1)

def process_missing_values():
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    global data
    try:
        if args.preprocessing["missing_values"] == "drop":
            data = data.dropna()
            print("Missing values eliminados con éxito")
        elif args.preprocessing["missing_values"] == "impute":
            if args.preprocessing["impute_strategy"] == "mean":
                data = data.fillna(data.mean())
                print("Missing values imputados con éxito usando la media")
            elif args.preprocessing["impute_strategy"] == "median":
                data = data.fillna(data.median())
                print("Missing values imputados con éxito usando la mediana")
            elif args.preprocessing["impute_strategy"] == "most_frequent":
                data = data.fillna(data.mode().iloc[0])
                print("Missing values imputados con éxito usando la moda")
            else:
                print("No se ha seleccionado ninguna estrategia de imputación")
        else:
            print("No se están tratando los missing values")
    except Exception as e:
        print("Error al tratar los missing values")
        print(e)
        sys.exit(1)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    global data
    try:
        if numerical_feature.columns.size > 0:
            if args.preprocessing["scaling"] == "minmax":
                scaler = MinMaxScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print("Datos reescalados con éxito usando MinMaxScaler")
            elif args.preprocessing["scaling"] == "normalizer":
                scaler = Normalizer()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print("Datos reescalados con éxito usando Normalizer")
            elif args.preprocessing["scaling"] == "maxabs":
                scaler = MaxAbsScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print("Datos reescalados con éxito usando MaxAbsScaler")
            else:
                print("No se están escalando los datos")
        else:
            print("No se han encontrado columnas numéricas")
    except Exception as e:
        print("Error al reescalar los datos")
        print(e)
        sys.exit(1)

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    global data
    try:
        if categorical_feature.columns.size > 0:
            labelencoder = LabelEncoder()
            for col in categorical_feature.columns:
                data[col] = labelencoder.fit_transform(data[col])
            print("Datos categóricos pasados a numéricos con éxito")
        else:
            print("No se han encontrado columnas categóricas")
    except Exception as e:
        print("Error al pasar los datos categóricos a numéricos")
        print(e)
        sys.exit(1)

def simplify_text(text_feature):
    """
    Simplifica el texto en el DataFrame 'data' utilizando técnicas de procesamiento de lenguaje natural.
    
    Parámetros:
    - data: DataFrame que contiene los datos a procesar.
    - text_data: DataFrame que contiene las columnas de texto a simplificar.
    
    Retorna:
    None
    """
    global data
    try:
        if text_feature.columns.size > 0:
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            for col in text_feature.columns:
                data[col] = data[col].apply(lambda x: ' '.join(sorted([stemmer.stem(word) for word in word_tokenize(x.lower()) if word not in stop_words and word not in string.punctuation])))
            print("Texto simplificado con éxito")
        else:
            print("No se han encontrado columnas de texto a simplificar")
    except Exception as e:
        print("Error al simplificar el texto")
        print(e)
        sys.exit(1)

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":
                vectorizer = TfidfVectorizer()
                text_features = vectorizer.fit_transform(data[text_feature.columns])
                text_features_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print("Texto tratado con éxito usando TF-IDF")
            elif args.preprocessing["text_process"] == "bow":
                vectorizer = CountVectorizer()
                text_features = vectorizer.fit_transform(data[text_feature.columns])
                text_features_df = pd.DataFrame(text_features.toarray())
                data = pd.concat([data, text_features_df], axis=1)
                print("Texto tratado con éxito usando BOW")
            else:
                print("No se están tratando los textos")
        else:
            print("No se han encontrado columnas de texto a procesar")
    except Exception as e:
        print("Error al tratar el texto")
        print(e)
        sys.exit(1)

def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    
    global data
    if args.mode != "test":
        try:
            if args.preprocessing["sampling"] == "oversampling":
                ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x, y = ros.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print("Oversampling realizado con éxito")
            elif args.preprocessing["sampling"] == "undersampling":
                rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x, y = rus.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print("Undersampling realizado con éxito")
            else:
                print("No se están realizando oversampling o undersampling")
        except Exception as e:
            print("Error al realizar oversampling o undersampling")
            print(e)
            sys.exit(1)
    else:
        print("No se realiza oversampling o undersampling en modo test")

def drop_features(features):
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=features)
        print("Columnas eliminadas con éxito")
    except Exception as e:
        print("Error al eliminar columnas")
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos 
        TODO 3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        TODO 6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        TODO 8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    if args.algorithm == "kNN":
        # Simplificamos el texto
        simplify_text(text_feature)

        # Pasar los datos a categoriales a numéricos
        cat2num(categorical_feature)

        # Tratamos missing values
        process_missing_values()

        # Reescalamos los datos numéricos
        reescaler(numerical_feature)
        
        # Realizamos Oversampling o Undersampling
        over_under_sampling()
    elif args.algorithm == "decision_tree" or args.algorithm == "random_forest":
        # Simplificamos el texto
        simplify_text(text_feature)

        # Pasar los datos a categoriales a numéricos
        cat2num(categorical_feature)

        # Tratamos el texto
        process_text(text_feature)

        # Realizamos Oversampling o Undersampling
        over_under_sampling()
    else:
        print("Algoritmo no soportado")
        sys.exit(1)

    return data

# Funciones para entrenar un modelo

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    - data: DataFrame que contiene los datos.
    - args: Objeto que contiene los argumentos necesarios para la división de datos.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    # Sacamos la columna a predecir
    y = data[args.prediction]
    x = data.drop(columns=[args.prediction])
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.25, random_state=42)
    return x_train, x_dev, y_train, y_dev

def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print("Modelo guardado con éxito")
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print("Error al guardar el modelo")
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print("> Mejores parametros:\n", gs.best_params_)
        print("> Mejor puntuacion:\n", gs.best_score_)
        print("> F1-score micro:\n", calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print("> F1-score macro:\n", calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print("> Informe de clasificación:\n", calculate_classification_report(y_dev, gs.predict(x_dev)))
        print("> Matriz de confusión:\n", calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    gs = GridSearchCV(KNeighborsClassifier(), args.kNN, cv=5, n_jobs=-1, )
    start_time = time.time()
    gs.fit(x_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:\n", execution_time, "segundos")
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    gs = GridSearchCV(DecisionTreeClassifier(), args.decision_tree, cv=5, n_jobs=-1,)
    start_time = time.time()
    gs.fit(x_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:\n", execution_time, "segundos")
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
        
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    gs = GridSearchCV(RandomForestClassifier(), args.random_forest, cv=5, n_jobs=-1,)
    start_time = time.time()
    gs.fit(x_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:\n", execution_time, "segundos")
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

# Funciones para predecir con un modelo

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print("Modelo cargado con éxito")
            return model
    except Exception as e:
        print("Error al cargar el modelo")
        print(e)
        sys.exit(1)
        
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = model.predict(data)
    
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)
    
    # Guardamos el dataframe con la prediccion
    data.to_csv('output/data-prediction.csv', index=False)
    print("Predicción guardada con éxito")

# Función principal

if __name__ == "__main__":
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    if args.mode == "train":
        # Cargamos los datos
        print("\n- Cargando datos...")
        data = load_data(args.file)
        # Descargamos los recursos necesarios de nltk
        print("\n- Descargando diccionarios...")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        # Preprocesamos los datos
        print("\n- Preprocesando datos...")
        preprocesar_datos()
        data.to_csv('output/data-processed.csv', index=False)
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print("Algoritmo kNN ejecutado con éxito")
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print("Algoritmo árbol de decisión ejecutado con éxito")
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print("Algoritmo random forest ejecutado con éxito")
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print("Algoritmo no soportado")
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos los datos
        print("\n- Cargando datos...")
        data = load_data(args.file)
        
        # Preprocesamos los datos
        print("\n- Preprocesando datos...")
        preprocesar_datos()
        
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        
        # Predecimos
        print("\n- Prediciendo...")
        predict()
    else:
        print("Modo no soportado")
        sys.exit(1)
