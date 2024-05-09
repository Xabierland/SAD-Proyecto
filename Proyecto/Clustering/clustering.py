# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña, Ibai Sologuestoa, Unai Garcia, Luken Bilbao
Descripcion: Script para clustering de datos usando gensim LDA
"""

## Librerias
import sys
import json
import argparse
import signal
import os
import traceback
import csv
import string
import unicodedata
import numpy as np
import pandas as pd
# Colorama - Colores en la terminal
from colorama import Fore, Style
# tqdm - Barra de progreso
from tqdm import tqdm
# NLTK - Procesado de texto
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
# Gensim - Preprocesado de texto y clustering con LDA de datos
from gensim.models import Phrases, LdaModel, Nmf,CoherenceModel, TfidfModel
from gensim.corpora import Dictionary
# Imblearn - Balanceo de datos
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
# Matplotlib - Graficas
import matplotlib.pyplot as plt

## Funciones auxiliares
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
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clustering de datos.")
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('clustering.json') as json_file:
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

## Preprocesado de datos
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

def process_missing_values(numerical_feature, categorical_feature, text_feature):
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
            data = data.dropna(subset=text_feature.columns)
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
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].mode().iloc[0])
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].mode().iloc[0])
                print(Fore.GREEN+"Missing values imputados con éxito usando la moda"+Fore.RESET)
            else:
                print(Fore.GREEN+"No se ha seleccionado ninguna estrategia de imputación"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se están tratando los missing values"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar los missing values"+Fore.RESET)
        print(e)
        sys.exit(1)

def process_text(text_feature):
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
                if args.debug:
                    print(Fore.MAGENTA+f"> Columna {col} en minúsculas"+Fore.RESET)
                # Tokenizamos
                data[col] = data[col].apply(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
                if args.debug:
                    print(Fore.MAGENTA+f"> Columna {col} tokenizada"+Fore.RESET)
                # Borrar numeros
                data[col] = data[col].apply(lambda x: [word for word in x if not word.isnumeric()])
                if args.debug:
                    print(Fore.MAGENTA+f"> Columna {col} sin números"+Fore.RESET)
                # Borrar stopwords
                data[col] = data[col].apply(lambda x: [word for word in x if word not in nltk.corpus.stopwords.words('english')])
                if args.debug:
                    print(Fore.MAGENTA+f"> Columna {col} sin stopwords"+Fore.RESET)
                # Lemmatizar
                data[col] = data[col].apply(lambda x: [WordNetLemmatizer().lemmatize(word) for word in x])
                if args.debug:
                    print(Fore.MAGENTA+f"> Columna {col} lemmatizada"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No hay columnas de texto en el dataset"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al simplificar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def unir_columnas():
    global data
    try:
        # Tenemos X columnas, cada una con una lista de palabras, unimos las X columnas en una sola columna y una sola lista
        data['text'] = data[data.columns[0]].apply(lambda x: x)
        for i in range(1, len(data.columns)):
            data['text'] = data['text'] + data[data.columns[i]].apply(lambda x: x)
        
        # Borramos las columnas que hemos unido
        data.drop(data.columns[0:len(data.columns)-1], axis=1, inplace=True)
                
        print(Fore.GREEN+"Columnas unidas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al unir las columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Borramos las columnas no necesarias
        2. Separamos los datos por tipos (Categoriales, numéricos y textos)
        3. Simplificamos el texto
            3.1. Tokenizar
            3.2. Eliminar stopwords
            3.3. Lemmatizar
            3.4. Ordenar alfabéticamente
        4. Tratamos el texto (TF-IDF, BOW)
    :param data: Datos a preprocesar
    :return: Datos preprocesados
    """
    # Eliminamos columnas no necesarias
    drop_features()    
    
    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()
    
    # Missing values
    process_missing_values(numerical_feature, categorical_feature, text_feature)

    # Simplificamos el texto
    process_text(text_feature)
    
    # Unir todas las columnas en una
    unir_columnas()

## Clustering
def lda():
    """
    Función para realizar el clustering utilizando el modelo LDA (Latent Dirichlet Allocation).

    Esta función realiza el clustering de un conjunto de textos utilizando el modelo LDA. 
    El proceso de clustering se realiza en varias etapas, donde se ajustan los parámetros 
    del modelo LDA y se calcula la coherencia de los tópicos generados. El objetivo es 
    encontrar la combinación de parámetros que genere los tópicos más coherentes.

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Si ocurre algún error durante el proceso de clustering.

    """
    try:
        # Añadimos los bigramas si aparecen en al menos 20 documentos
        bigram = Phrases(data['text'], min_count=20)
        for idx in range(len(data['text'])):
            for token in bigram[data['text'][idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    data['text'][idx].append(token)
        
        # Creamos el diccionario eliminando palabras que aparecen en menos de 20 documentos y en más del 50% de los documentos
        # El diccionario es un mapeo de palabras a IDs
        dictionary = Dictionary(data['text'])
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        
        # Creamos el corpus
        # El corpus es una lista de listas, donde cada lista contiene tuplas (id_palabra, frecuencia)
        if args.preprocessing["text_process"] == "bow":
            corpus = [dictionary.doc2bow(doc) for doc in data['text']]
        elif args.preprocessing["text_process"] == "tf-idf":
            tfidf = TfidfModel(dictionary=dictionary)
            corpus = [tfidf[dictionary.doc2bow(doc)] for doc in data['text']]
           
        
        # Creamos el id2word
        # El id2word es un mapeo de IDs a palabras, se diferencia del diccionario en que el diccionario mapea palabras a IDs
        temp = dictionary[0]
        id2word = dictionary.id2token
        
        # Perform LDA
        avg_topic_coherence = 0
        best_avg_topic_coherence = 0
        
        # Valores para la grafica
        umass_values = []
        cv_values = []
        num_topics_values = []

        with tqdm(total=len(args.lda["num_topics"]) * len(args.lda["passes"]) * len(args.lda["iterations"])) as pbar:
            with open(safe_folder+'/clustering_results.csv', 'w', newline='') as csvfile:
                fieldnames = ['Num Topics', 'Passes', 'Iterations', 'Coherence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for num_topic in args.lda["num_topics"]:
                    for passes in args.lda["passes"]:
                        for iterations in args.lda["iterations"]:
                            lda = LdaModel(corpus=corpus,
                                id2word=id2word,
                                alpha='auto',
                                eta='auto',
                                iterations=int(iterations),
                                num_topics=int(num_topic),
                                passes=int(passes),
                                random_state=42)

                            # Calculamos la coherencia de los topicos
                            avg_topic_coherence = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass').get_coherence()

                            if args.debug:
                                print('\nCoherencia %.4f.' % avg_topic_coherence)

                            # Guardamos los resultados
                            writer.writerow({'Num Topics': num_topic, 'Passes': passes, 'Iterations': iterations, 'Coherence': avg_topic_coherence})
                            
                            # Si la coherencia es mejor que la anterior, guardamos el modelo
                            if avg_topic_coherence < best_avg_topic_coherence:
                                best_model = lda
                                best_num_topic = num_topic
                                best_passes = passes
                                best_iterations = iterations
                                best_avg_topic_coherence = avg_topic_coherence

                            # Añade valor a la grafica
                            umass_values.append(avg_topic_coherence)
                            cv_values.append(CoherenceModel(model=lda, texts=data['text'], dictionary=dictionary, coherence='c_v').get_coherence())
                            num_topics_values.append(num_topic)

                            # Actualizamos la barra de progreso
                            pbar.update(1)

        # Graficamos la coherencia en función del número de tópicos
        plt.figure()
        plt.plot(num_topics_values, umass_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (u_mass)')
        plt.title('Coherencia en función del número de tópicos')
        plt.savefig(safe_folder+'/coherence_umass.png')
        
        plt.figure()
        plt.plot(num_topics_values, cv_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (c_v)')
        plt.title('Coherencia en función del número de tópicos')
        plt.savefig(safe_folder+'/coherence_cv.png')

        # Imprimimos el mejor resultado
        if args.verbose:
            print('Media coherencia de topico: %.4f.' % best_avg_topic_coherence)
            print('Mejores parametros: num_topics=%d, passes=%d, iterations=%d' % (best_num_topic, best_passes, best_iterations))
            i=0
            for topic in best_model.top_topics(corpus):
                i+=1
                print('Topic', i)
                print(topic)
        
        # Guardamos los mejores topicos
        with open(safe_folder+'/topics.txt', 'w') as f:
            f.write('Media coherencia de topico: %.4f.\n' % best_avg_topic_coherence)
            f.write('Mejores parametros: num_topics=%d, passes=%d, iterations=%d\n' % (best_num_topic, best_passes, best_iterations))
            i=0
            for topic in best_model.top_topics(corpus):
                i+=1
                f.write('Topic %d\n' % i)
                f.write(str(topic)+'\n')
        
        # Guardamos el modelo
        best_model.save(safe_folder+'/lda_model')
    except Exception as e:
        print(Fore.RED+"Error al realizar el clustering"+Fore.RESET)
        print(e)
        traceback.print_exc()
        sys.exit(1)

def nmf():
    """
    Función para realizar el clustering utilizando el modelo NMF (Non-negative Matrix Factorization).

    Esta función realiza el clustering de un conjunto de textos utilizando el modelo NMF. 
    El proceso de clustering se realiza en varias etapas, donde se ajustan los parámetros 
    del modelo NMF y se calcula la coherencia de los tópicos generados. El objetivo es 
    encontrar la combinación de parámetros que genere los tópicos más coherentes.

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Si ocurre algún error durante el proceso de clustering.

    """
    try:
        # Añadimos los bigramas si aparecen en al menos 20 documentos
        bigram = Phrases(data['text'], min_count=20)
        for idx in range(len(data['text'])):
            for token in bigram[data['text'][idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    data['text'][idx].append(token)
        
        # Creamos el diccionario eliminando palabras que aparecen en menos de 20 documentos y en más del 50% de los documentos
        # El diccionario es un mapeo de palabras a IDs
        dictionary = Dictionary(data['text'])
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        
        # Creamos el corpus
        # El corpus es una lista de listas, donde cada lista contiene tuplas (id_palabra, frecuencia)
        if args.preprocessing["text_process"] == "bow":
            corpus = [dictionary.doc2bow(doc) for doc in data['text']]
        elif args.preprocessing["text_process"] == "tf-idf":
            tfidf = TfidfModel(dictionary=dictionary)
            corpus = [tfidf[dictionary.doc2bow(doc)] for doc in data['text']]
           
        
        # Creamos el id2word
        # El id2word es un mapeo de IDs a palabras, se diferencia del diccionario en que el diccionario mapea palabras a IDs
        temp = dictionary[0]
        id2word = dictionary.id2token
        
        # Perform NMF
        avg_topic_coherence = 0
        best_avg_topic_coherence = 0

        # Valores para la grafica
        umass_values = []
        cv_values = []
        num_topics_values = []

        with tqdm(total=len(args.nmf["num_topics"]) * len(args.nmf["passes"]) * len(args.nmf["iterations"])) as pbar:
            with open(safe_folder+'/clustering_results.csv', 'w', newline='') as csvfile:
                fieldnames = ['Num Topics', 'Passes', 'Iterations', 'Coherence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for num_topic in args.nmf["num_topics"]:
                    for passes in args.nmf["passes"]:
                        for iterations in args.nmf["iterations"]:
                            nmf = Nmf(corpus=corpus,
                                id2word=id2word,
                                num_topics=int(num_topic),
                                passes=int(passes),
                                random_state=42)

                            # Calculamos la coherencia de los topicos
                            avg_topic_coherence = CoherenceModel(model=nmf, corpus=corpus, dictionary=dictionary, coherence='u_mass').get_coherence()

                            if args.debug:
                                print('\nCoherencia %.4f.' % avg_topic_coherence)

                            # Guardamos los resultados
                            writer.writerow({'Num Topics': num_topic, 'Passes': passes, 'Iterations': iterations, 'Coherence': avg_topic_coherence})
                            
                            # Si la coherencia es mejor que la anterior, guardamos el modelo
                            if avg_topic_coherence < best_avg_topic_coherence:
                                best_model = nmf
                                best_num_topic = num_topic
                                best_passes = passes
                                best_iterations = iterations
                                best_avg_topic_coherence = avg_topic_coherence

                            # Añade valor a la grafica
                            umass_values.append(avg_topic_coherence)
                            cv_values.append(CoherenceModel(model=nmf, texts=data['text'], dictionary=dictionary, coherence='c_v').get_coherence())
                            num_topics_values.append(num_topic)

                            # Actualizamos la barra de progreso
                            pbar.update(1)

        # Graficamos la coherencia en función del número de tópicos
        plt.plot(num_topics_values, umass_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (u_mass)')
        plt.title('Coherencia en función del número de tópicos')
        plt.savefig(safe_folder+'/coherence_umass.png')

        plt.plot(num_topics_values, cv_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (c_v)')
        plt.title('Coherencia en función del número de tópicos')
        plt.savefig(safe_folder+'/coherence_cv.png')

        # Imprimimos el mejor resultado
        if args.verbose:
            print('Media coherencia de topico: %.4f.' % best_avg_topic_coherence)
            print('Mejores parametros: num_topics=%d, passes=%d, iterations=%d' % (best_num_topic, best_passes, best_iterations))
            i=0
            for topic in best_model.top_topics(corpus):
                i+=1
                print('Topic', i)
                print(topic)
        
        # Guardamos los mejores topicos
        with open(safe_folder+'/topics.txt', 'w') as f:
            f.write('Media coherencia de topico: %.4f.\n' % best_avg_topic_coherence)
            f.write('Mejores parametros: num_topics=%d, passes=%d, iterations=%d\n' % (best_num_topic, best_passes, best_iterations))
            i=0
            for topic in best_model.top_topics(corpus):
                i+=1
                f.write('Topic %d\n' % i)
                f.write(str(topic)+'\n')
        
        # Guardamos el modelo
        best_model.save(safe_folder+'/nmf_model')
    except Exception as e:
        print(Fore.RED+"Error al realizar el clustering"+Fore.RESET)
        print(e)
        traceback.print_exc()
        sys.exit(1)

## Main
if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clustering ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    if os.name == 'nt':
        safe_folder = args.file.split('\\')[-1].split('.')[0]
    else:
        safe_folder = args.file.split('/')[-1].split('.')[0]
    try:
        # Creamos la carpeta con el nombre del fichero csv recibido como argumento
        # Ten en cuenta que args.file es el path completo al fichero csv - file='..\\Datos\\Output\\BritishPos.csv'
        # Por lo que si lo dividimos por '\' y cogemos el último elemento, obtendremos el nombre del fichero
        # En este caso BritishPos
        os.makedirs(safe_folder)
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
            data.to_csv(safe_folder+'/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
    print("\n- Realizando clustering...")
    try:
        lda()
    except Exception as e:
        print(Fore.RED+"Error al realizar el clustering"+Fore.RESET)
        print(e)
        sys.exit(1)
    print(Fore.GREEN+"Clustering realizado con éxito"+Fore.RESET)
    print("\nSaliendo del programa...")
    sys.exit(0)
    