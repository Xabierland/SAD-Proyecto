"""
Autor: Xabier Gabiña Barañano a.k.a. Xabierland
Fecha: 2024/04/19
Descripción: Script junta todos los CSVs de los datos en uno solo.
"""

# Importamos las librerías necesarias
import pandas as pd
import os
import csv

# Funcion Main
if __name__ == "__main__":
    # Leemos Airlines.csv
    Airlines = pd.read_csv("Airlines.csv")
    British = pd.read_csv("BritishAirlines.csv")
    
    # Dropeamos las columnas que no existan
    Airlines = Airlines.drop(columns=["Review Date"])
    British = British.drop(columns=["aircraft"])
    
    # Dado que las columnas no se llaman igual ni estan ordenadas igual, las ordenamos y renombramos de una en una
    #   Columnas de Airlines.csv
    #       - Title,Name,Review Date,Airline,Verified,Reviews,Type of Traveller,Month Flown,Route,Class,Seat Comfort,Staff Service,Food & Beverages,Inflight Entertainment,Value For Money,Overall Rating,Recommended
    #   Columnas de BritishAirways.csv
    #       - header,author,date,place,content,aircraft,traveller_type,seat_type,route,date_flown,recommended,trip_verified,rating,seat_comfort,cabin_staff_service,food_beverages,ground_service,value_for_money,entertainment
    #   Columnas de Datos.csv
    #       - Title - header
    #       - Name - author
    #       - Airline - "British Airlines"
    #       - Verified - trip_verified
    #       - Reviews - content
    #       - Type of Traveller - traveller_type
    #       - Month Flown - date_flown
    #       - Route - route
    #       - Class - seat_type
    #       - Seat Comfort - seat_comfort
    #       - Staff Service - cabin_staff_service
    #       - Food & Beverages - food_beverages
    #       - Inflight Entertainment - entertainment
    #       - Value For Money - value_for_money
    #       - Overall Rating - rating
    #       - Recommended - recommended
    
    # Modificaciones pre-merge
    #   Cambiamos el formato de la fecha de British
    British["date_flown"] = pd.to_datetime(British["date_flown"], format="%d-%m-%Y").dt.strftime("%B %Y")
    # Concatenamos los dos DataFrames en uno solo teniendo en cuenta el titulo de las columnas y no el orden
    Datos = pd.DataFrame()
    Datos["Title"] = pd.concat([Airlines["Title"], British["header"]], axis=0)
    Datos["Name"] = pd.concat([Airlines["Name"], British["author"]], axis=0)
    Datos["Airline"] = pd.concat([Airlines["Airline"], pd.Series(["British Airlines"] * len(British))], axis=0)
    Datos["Date"] = pd.concat([Airlines["Month Flown"], British["date_flown"]], axis=0)
    Datos["Verified"] = pd.concat([Airlines["Verified"], British["trip_verified"]], axis=0)
    Datos["Reviews"] = pd.concat([Airlines["Reviews"], British["content"]], axis=0)
    Datos["Type of Traveller"] = pd.concat([Airlines["Type of Traveller"], British["traveller_type"]], axis=0)
    Datos["Route"] = pd.concat([Airlines["Route"], British["route"]], axis=0)
    Datos["Class"] = pd.concat([Airlines["Class"], British["seat_type"]], axis=0)
    Datos["Seat Comfort"] = pd.concat([Airlines["Seat Comfort"], British["seat_comfort"]], axis=0)
    Datos["Staff Service"] = pd.concat([Airlines["Staff Service"], British["cabin_staff_service"]], axis=0)
    Datos["Food & Beverages"] = pd.concat([Airlines["Food & Beverages"], British["food_beverages"]], axis=0)
    Datos["Inflight Entertainment"] = pd.concat([Airlines["Inflight Entertainment"], British["entertainment"]], axis=0)
    Datos["Value For Money"] = pd.concat([Airlines["Value For Money"], British["value_for_money"]], axis=0)
    Datos["Overall Rating"] = pd.concat([Airlines["Overall Rating"], British["rating"]], axis=0)
    # Modificaciones post-merge
    #   Existen saltos de linea en los textos de las columnas Title y Reviews, los eliminamos
    Datos["Title"] = Datos["Title"].str.replace("\n","")
    Datos["Reviews"] = Datos["Reviews"].str.replace(r"[\n\r]+", " ")
    #   Eliminamos el caracter U+00a0 (Non-breaking space) de las columnas
    Datos["Title"] = Datos["Title"].str.replace("\u00a0","")
    Datos["Reviews"] = Datos["Reviews"].str.replace("\u00a0","")
    # Eliminamos el caracter U+2013 y lo reemplazamos por un guion
    Datos["Title"] = Datos["Title"].str.replace("\u2013","-")
    Datos["Reviews"] = Datos["Reviews"].str.replace("\u2013","-")
    # Buscamos dos * juntos con cualquier caracter por detras y espacio por delante
    Datos["Reviews"] = Datos["Reviews"].str.replace("**", "")
    #   Elimina los caracteres en blanco al principio y al final de las columnas
    Datos["Title"] = Datos["Title"].str.strip()
    Datos["Reviews"] = Datos["Reviews"].str.strip()
    #   Cambiamos los valores de las columnas Verified a True y False
    Datos["Verified"] = Datos["Verified"].replace("Verified", True)
    Datos["Verified"] = Datos["Verified"].replace("Not Verified", False)
    
    # Dividimos el DataFrame
    #   Datos de British Airlines
    British = Datos[Datos["Airline"] == "British Airlines"]
    #   Datos de Air France
    AirFrance = Datos[Datos["Airline"] == "Air France"]
    #   Dividimos los datos de British en tres, Overall Rating de 0-5, 5-7 y 7-10
    BritishNeg = British[British["Overall Rating"] <= 5]
    BritishNeu = British[(British["Overall Rating"] > 5) & (British["Overall Rating"] <= 7)]
    BritishPos = British[British["Overall Rating"] > 7]
    #   Dividimos los datos de Air France en tres, Overall Rating de 0-5, 5-7 y 7-10
    AirFranceNeg = AirFrance[AirFrance["Overall Rating"] <= 5]
    AirFranceNeu = AirFrance[(AirFrance["Overall Rating"] > 5) & (AirFrance["Overall Rating"] <= 7)]
    AirFrancePos = AirFrance[AirFrance["Overall Rating"] > 7]
    
    # Creamos la carpeta Output si no existe y dentro guardamos los CSVs
    if not os.path.exists("Output"):
        os.makedirs("Output")
    # Guardamos el DataFrame en un CSV
    Datos.to_csv("Output/Datos.csv", index=False)
    British.to_csv("Output/British.csv", index=False)
    BritishNeg.to_csv("Output/BritishNeg.csv", index=False)
    BritishNeu.to_csv("Output/BritishNeu.csv", index=False)
    BritishPos.to_csv("Output/BritishPos.csv", index=False)
    AirFrance.to_csv("Output/AirFrance.csv", index=False)
    AirFranceNeg.to_csv("Output/AirFranceNeg.csv", index=False)
    AirFranceNeu.to_csv("Output/AirFranceNeu.csv", index=False)
    AirFrancePos.to_csv("Output/AirFrancePos.csv", index=False)