# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña y Ibai Sologestoa.
Script para la implementación del algoritmo de clasificación
"""

import sys
import signal
import argparse
import numpy
import pandas as pd
import string
import imblearn
import pickle
import json
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler

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
    parse = argparse.ArgumentParser(description="Clasificador")
    parse.add_argument("-f", "--file", help="Fichero csv", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-P", "--prediction", help="Columna a predecir", required=True)
    
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('clasificador.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    print(args)
    
    # Parseamos los argumentos
    return args
    

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    data = pd.read_csv(file, encoding='utf-8')
    return data

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
    report = classification_report(y_test, y_pred)
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

def preprocesar_datos(data, args):
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Tratamos missing values (Eliminar o imputar)
        3. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        4. Pasar los datos a categoriales a numéricos 
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        TODO 7. Realizamos Oversampling o Undersampling
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """

    # 1 Separamos los datos por tipos
    numerical_data = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
    text_data = data[data.columns[data.apply(lambda col: col.astype(str).str.contains(' ', na=False).any())]] # Columnas con texto
    categorical_data = data.select_dtypes(include='object').drop(columns=text_data.columns) # Columnas categóricas
    #numerical_data = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
    #categorical_data = data.select_dtypes(include='category') # Columnas con categorías
    #text_data = data.select_dtypes(include='object').drop(columns=categorical_data.columns) # Columnas con texto

    # 2 Tratamos missing values
    if args.preprocessing["missing_values"] == "drop":
        data = data.dropna()
    elif args.preprocessing["missing_values"] == "impute":
        if args.preprocessing["impute_strategy"] == "mean":
            data = data.fillna(data.mean())
        elif args.preprocessing["impute_strategy"] == "median":
            data = data.fillna(data.median())
        elif args.preprocessing["impute_strategy"] == "most_frequent":
            data = data.fillna(data.mode().iloc[0])
    else:
        print("No se estan tratando los missing values")

    # 3 Reescalamos los datos numéricos
    if numerical_data.columns.size > 0:
        if args.preprocessing["scaling"] == "minmax":
            scaler = MinMaxScaler()
            data[numerical_data.columns] = scaler.fit_transform(data[numerical_data.columns])
        elif args.preprocessing["scaling"] == "normalizer":
            scaler = Normalizer()
            data[numerical_data.columns] = scaler.fit_transform(data[numerical_data.columns])
        elif args.preprocessing["scaling"] == "maxabs":
            scaler = MaxAbsScaler()
            data[numerical_data.columns] = scaler.fit_transform(data[numerical_data.columns])
        else:
            print("No se estan escalando los datos")

    # 4 Pasar los datos a categoriales a numéricos
    if categorical_data.columns.size > 0:
        labelencoder = LabelEncoder()
        for col in categorical_data.columns:
            data[col] = labelencoder.fit_transform(data[col])

    # 5 Simplificamos el texto
    if text_data.columns.size > 0:
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        for col in text_data.columns:
            data[col] = data[col].apply(lambda x: ' '.join(sorted([stemmer.stem(word) for word in word_tokenize(x.lower()) if word not in stop_words and word not in string.punctuation])))

    # 6 Tratamos el texto
    if text_data.columns.size > 0:
        if args.preprocessing["text_process"] == "tf-idf":
            vectorizer = TfidfVectorizer()
            text_features = vectorizer.fit_transform(data[text_data.columns].values.astype('U').flatten())
            text_features_df = pd.DataFrame(text_features.toarray())
            data = pd.concat([data, text_features_df], axis=1)
        elif args.preprocessing["text_process"] == "bow":
            vectorizer = CountVectorizer()
            text_features = vectorizer.fit_transform(data[text_data.columns].values.astype('U').flatten())
            text_features_df = pd.DataFrame(text_features.toarray())
            data = pd.concat([data, text_features_df], axis=1)
        else:
            print("No se estan tratando los textos")
    
    # 7 Realizamos Oversampling o Undersampling
    
    
    # ! Just for testing
    data.to_csv('datos/data.csv', index=False)
    return data

def kNN(data):
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param train: Conjunto de datos de entrenamiento.
    :type train: pandas.DataFrame
    :param dev: Conjunto de datos de desarrollo.
    :type dev: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Sacamos la columna a predecir
    y = data[args.prediction]
    x = data.drop(columns=[args.prediction])
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.25)
    
    # Hacemos un barrido de hiperparametros
    f1_score_micro = 0
    f1_score_macro = 0
    k_optimo = 0
    w_optimo = ''
    p_optimo = 0
    i=0
    for k in range(int(args.kNN["k-min"]), int(args.kNN["k-max"])):
        for w in ['uniform', 'distance']:
            for p in range(1, 3):
                knn = KNeighborsClassifier(n_neighbors=k, weights=w, p=p)
                knn.fit(x_train, y_train)
                y_pred = knn.predict(x_dev)
                i+=1
                if f1_score(y_dev, y_pred, average='micro') > f1_score_micro:
                    if f1_score(y_dev, y_pred, average='macro') > f1_score_macro:
                        f1_score_micro = f1_score(y_dev, y_pred, average='micro')
                        f1_score_macro = f1_score(y_dev, y_pred, average='macro')
                        k_optimo = k
                        w_optimo = w
                        p_optimo = p
                        # Guardamos el modelo utilizando pickle
                        with open('modelo.pkl', 'wb') as file:
                            pickle.dump(knn, file)
    print("Numero de iteraciones: ", i)
    print("k_optimo: ", str(k_optimo) + " w_optimo: ", str(w_optimo) + " p_optimo: ", str(p_optimo))
    print("F1-score micro: ", str(f1_score_micro) + " F1-score macro: ", str(f1_score_macro))
                
    
def decision_tree(data):
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Sacamos la columna a predecir
    y = data[args.prediction]
    x = data.drop(columns=[args.prediction])
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.25)
    f1_score_micro = 0
    f1_score_macro = 0
    m_optimo = 0
    c_optimo = ''
    i=0
    for m in range(int(args.decision_tree["min_depth"]), int(args.decision_tree["max_depth"])):
        for c in ['gini', 'entropy']:
            dt = DecisionTreeClassifier(max_depth=m, criterion=c)
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_dev)
            i+=1
            if f1_score(y_dev, y_pred, average='micro') > f1_score_micro:
                if f1_score(y_dev, y_pred, average='macro') > f1_score_macro:
                    f1_score_micro = f1_score(y_dev, y_pred, average='micro')
                    f1_score_macro = f1_score(y_dev, y_pred, average='macro')
                    m_optimo = m
                    c_optimo = c
                    # Guardamos el modelo utilizando pickle
                    with open('modelo.pkl', 'wb') as file:
                        pickle.dump(dt, file)
    print("Numero de iteraciones: ", i)
    print("m_optimo: ", str(m_optimo) + " c_optimo: ", str(c_optimo))
    print("F1-score micro: ", str(f1_score_micro) + " F1-score macro: ", str(f1_score_macro))
    
def random_forest(data):
    """
    Función para implementar el algoritmo de random forest.

    :param train: Conjunto de datos de entrenamiento.
    :type train: pandas.DataFrame
    :param dev: Conjunto de datos de desarrollo.
    :type dev: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """

if __name__ == "__main__":
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Descargamos los recursos necesarios de nltk
    print("\nDescargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Cargamos los datos
    print("\nCargando datos...")
    data = load_data(args.file)
    # Preprocesamos los datos
    print("\nPreprocesando datos...")
    data=preprocesar_datos(data, args)
    # Ejecutamos el algoritmo seleccionado
    print("\nEjecutando algoritmo...")
    if args.algorithm == "kNN":
        kNN(data)
    elif args.algorithm == "decision_tree":
        decision_tree(data)
    elif args.algorithm == "random_forest":
        random_forest(data)
    else:
        print("Algoritmo no soportado")
        sys.exit(1)
