# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña y Ibai Sologestoa.
Script para la implementación del algoritmo de clasificación
"""

import random
import sys
import signal
import argparse
import unicodedata
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
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
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

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
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
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
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
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

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """
    global data
    try:
        if args.preprocessing["missing_values"] == "drop":
            data = data.dropna(subset=numerical_feature.columns)
            data = data.dropna(subset=categorical_feature.columns)
            print(Fore.GREEN+"Missing values eliminados con éxito"+Fore.RESET)
        elif args.preprocessing["missing_values"] == "impute":
            if args.preprocessing["impute_strategy"] == "mean":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].mean())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].mean())
                print(Fore.GREEN+"Missing values imputados con éxito usando la media"+Fore.RESET)
            elif args.preprocessing["impute_strategy"] == "median":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].median())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].median())
                print(Fore.GREEN+"Missing values imputados con éxito usando la mediana"+Fore.RESET)
            elif args.preprocessing["impute_strategy"] == "most_frequent":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].mode())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].mode())
                print(Fore.GREEN+"Missing values imputados con éxito usando la moda"+Fore.RESET)
            else:
                print(Fore.GREEN+"No se ha seleccionado ninguna estrategia de imputación"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se están tratando los missing values"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar los missing values"+Fore.RESET)
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
                print(Fore.GREEN+"Datos reescalados con éxito usando MinMaxScaler"+Fore.RESET)
            elif args.preprocessing["scaling"] == "normalizer":
                scaler = Normalizer()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN+"Datos reescalados con éxito usando Normalizer"+Fore.RESET)
            elif args.preprocessing["scaling"] == "maxabs":
                scaler = MaxAbsScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN+"Datos reescalados con éxito usando MaxAbsScaler"+Fore.RESET)
            elif args.preprocessing["scaling"] == "standard":
                scaler = StandardScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN+"Datos reescalados con éxito usando StandardScaler"+Fore.RESET)
            else:
                print(Fore.YELLOW+"No se están escalando los datos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas numéricas"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al reescalar los datos"+Fore.RESET)
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
            print(Fore.GREEN+"Datos categóricos pasados a numéricos con éxito"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas categóricas que pasar a numericas"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al pasar los datos categóricos a numéricos"+Fore.RESET)
        print(e)
        sys.exit(1)

def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame.
    
    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    
    Retorna:
    None
    """
    global data
    try:
        if text_feature.columns.size > 0:
            for col in text_feature.columns:
                # Minúsculas
                data[col] = data[col].apply(lambda x: x.lower())
                # Tokenizamos
                data[col] = data[col].apply(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
                # Borrar numeros
                data[col] = data[col].apply(lambda x: [word for word in x if not word.isnumeric()])
                # Borrar stopwords
                data[col] = data[col].apply(lambda x: [word for word in x if word not in nltk.corpus.stopwords.words('english')])
                # Lemmatizar
                data[col] = data[col].apply(lambda x: [WordNetLemmatizer().lemmatize(word) for word in x])
                # Borrar acentos y otros caracteres especiales
                data[col] = data[col].apply(lambda x: ''.join(patata for patata in unicodedata.normalize('NFD', ' '.join(x)) if unicodedata.category(patata) != 'Mn'))
        else:
            print(Fore.YELLOW+"No hay columnas de texto en el dataset"+Fore.RESET)

    except Exception as e:
               print(Fore.RED+"Error al simplificar el texto"+Fore.RESET)
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
                v = TfidfVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                x = v.fit_transform(text_data)
                df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names_out())
                data.drop(text_feature.columns, axis=1, inplace=True)
                data = pd.concat([data, df1], axis=1)
                #data.drop_duplicates(keep='first', inplace=True)
                print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)            
            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def samplingMetodo():
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
                if isinstance(data[args.prediction], pd.DataFrame):
                    y = data[args.prediction].iloc[:,0]
                else:
                    y = data[args.prediction]
                x, y = ros.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print(Fore.GREEN+"Oversampling realizado con éxito"+Fore.RESET)
            elif args.preprocessing["sampling"] == "undersampling":
                rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
                x = data.drop(columns=[args.prediction])
                if isinstance(data[args.prediction], pd.DataFrame):
                    y = data[args.prediction].iloc[:,0]
                else:
                    y = data[args.prediction]
                x, y = rus.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print(Fore.GREEN+"Undersampling realizado con éxito"+Fore.RESET)
            elif args.preprocessing["sampling"].lower() == "smote":
                sm = SMOTE(sampling_strategy='auto', random_state=42)
                x = data.drop(columns=[args.prediction])
                if isinstance(data[args.prediction], pd.DataFrame):
                    y = data[args.prediction].iloc[:,0]
                else:
                    y = data[args.prediction]
                x, y = sm.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1) 
                print(Fore.GREEN+"SMOTE realizado con éxito"+Fore.RESET)
                
            else:
                print(Fore.YELLOW+"No se están realizando oversampling, undersampling o SMOTE"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al realizar oversampling, undersampling o SMOTE"+Fore.RESET)
            print(e)
            sys.exit(1)
    else:
        print(Fore.GREEN+"No se realiza oversampling, undersampling o SMOTE en modo test"+Fore.RESET)

def drop_features():
    global data
    try:
        for column in args.preprocessing["drop_features"]:
            if column not in data.columns:
                print(Fore.YELLOW+f"La columna {column} no existe en el dataset"+Fore.RESET)
            else:
                data.drop(column, axis=1, inplace=True)
                if args.debug:
                    print(Fore.MAGENTA+f"> Columna {column} eliminada"+Fore.RESET)
        print(Fore.GREEN+f"Columnas eliminadas"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Simplificamos el texto
    simplify_text(text_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    convertirRating()
    # Reescalamos los datos numéricos
    reescaler(numerical_feature)
    
    # Tratamos el texto
    process_text(text_feature)
    
    # Realizamos Oversampling o Undersampling
    samplingMetodo()

    drop_features()

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
            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
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
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def convertirRating():
    global data
    data[args.preprocessing["rating_to_cat"]]= data[args.preprocessing["rating_to_cat"]].map({10: "positiva", 9: "positiva", 8: "positiva", 7: "positiva", 6: "neutra", 5: "neutra", 4: "negativa", 3: "negativa", 2: "negativa", 1: "negativa", 0: "negativa"})
def naiveBayes():
    x_train, x_dev, y_train, y_dev = divide_data()

    """

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

    # Entrenar el modelo
    text_clf.fit(X_train, y_train)

    # Guardar el modelo con pickle
    with open('modelo.pkl', 'wb') as file:
        pickle.dump(text_clf, file)

    predicted = text_clf.predict(X_test)

    accuracy = np.mean(predicted == y_test)
    print(f'Precisión del modelo: {accuracy}') """
    with tqdm(total=100, desc='Procesando naive bayes', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(MultinomialNB(), args.bayes, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        if isinstance(y_train, pd.DataFrame):
           print(y_train.iloc[:,0])
           gs.fit(x_train, y_train.iloc[:,0])
        # print("hola")
        # print("hola, que tal?")
        # print("yo bien y tu?")
        # print("bien tambien")
        # print("y tu?")
        # print("yo tambien")
        # print("cual es el significado de la vida?")
        # print("42")
        # print("gracias")
           y_dev = y_dev.iloc[:,0]
        else:
            print(y_train)
            gs.fit(x_train, y_train)                 
                                    #solo el texto y el overall
                                    #todos y el overrall
                                    #con los ratings menos el texto y cosas que no sean rating
         #       print(x_train.iloc[:, 3])                            
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15)) 
            pbar.update(random.random()*2)  
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    
    
    
    
    data.to_csv('preprocessed_data.csv', index=False)
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

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
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
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
    
# Función principal

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
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
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        try:    
                naiveBayes()
                print(Fore.GREEN+"Algoritmo naive bayes ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
        except Exception as e:
                print(e)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            data.to_csv('output/data-prediction.csv', index=False)
            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
