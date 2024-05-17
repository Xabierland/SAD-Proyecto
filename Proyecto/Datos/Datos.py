"""
Autor: Xabier Gabiña Barañano a.k.a. Xabierland
Fecha: 2024/04/19
Descripción: Script junta todos los CSVs de los datos en uno solo.
"""

# Importamos las librerías necesarias
import pandas as pd
import os
import csv
import pycountry

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
    
    # Borramos filas con NA
    Datos = Datos.dropna()
    
    #   Existen saltos de linea en los textos de las columnas Title y Reviews, los eliminamos
    Datos["Title"] = Datos["Title"].str.replace(r"[\n\r]+", " ")
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
    
    #   Copia Overall Rating a Numerical Overall Rating
    Datos["Numerical Overall Rating"] = Datos["Overall Rating"]
    
    #   Cambiamos los valores de las columnas Rating por Pos Neu y Neg
    Datos["Overall Rating"] = Datos["Overall Rating"].apply(lambda x: "POS" if x >= 7 else "NEG" if x <= 4 else "NEU")
    
    #   Cambiamos la columna Routa para que en vez de poner "CIUDAD1 to CIUDAD3 via CIUDAD2" ponga "CIUDAD1 - CIUDAD2 - CIUDAD3"
    #       Reordenamos las ciudades para que sea CIUDAD1 via CIUDAD2 to CIUDAD3
    Datos["Route"] = Datos["Route"].str.replace(" to ", " - ")
    Datos["Route"] = Datos["Route"].str.replace(" via ", " - ")
    #      Cambiamos el orden de las ciudades
    Datos["Route"] = Datos["Route"].str.split(" - ").apply(lambda x: x[0] + " - " + x[-1] + " - " + x[1] if len(x) == 3 else x[0] + " - " + x[1] if len(x) == 2 else x[0])
    #       En caso de encontrar una / borrarla y lo que haya detrás
    Datos["Route"] = Datos["Route"].str.split("/").str[0]
    
    # Pasa los nombres de las ciudades y los codigos de ciudades a los paises usando la libreria pycountry
    for linea in Datos["Route"]:
        ciudades = linea.split(" - ")
        paises = []
        for ciudad in ciudades:
            try:
                pais = pycountry.countries.get(name=ciudad)
                paises.append(pais.name)
            except:
                paises.append(ciudad)
                print(ciudad)
        Datos["Route"] = Datos["Route"].str.replace(linea, " - ".join(paises))
    
    # Dividimos el DataFrame
    #   Datos de British Airlines
    British = Datos[Datos["Airline"] == "British Airlines"]
    #   Datos de Air France
    AirFrance = Datos[Datos["Airline"] == "Air France"]
    #   Dividimos los datos de British en tres, Overall Rating de NEG, NEU y POS
    BritishNeg = British[British["Overall Rating"] == "NEG"]
    BritishNeu = British[British["Overall Rating"] == "NEU"]
    BritishPos = British[British["Overall Rating"] == "POS"]
    #   Dividimos los datos de Air France en tres, Overall Rating de 0-5, 5-7 y 7-10
    AirFranceNeg = AirFrance[AirFrance["Overall Rating"] == "NEG"]
    AirFranceNeu = AirFrance[AirFrance["Overall Rating"] == "NEU"]
    AirFrancePos = AirFrance[AirFrance["Overall Rating"] == "POS"]
    #   Dividimos los Datos en tres, Overall Rating de 0-5, 5-7 y 7-10
    DatosNeg = Datos[Datos["Overall Rating"] == "NEG"]
    DatosNeu = Datos[Datos["Overall Rating"] == "NEU"]
    DatosPos = Datos[Datos["Overall Rating"] == "POS"]
    # Creamos la carpeta Output si no existe y dentro guardamos los CSVs
    if not os.path.exists("output"):
        os.makedirs("output")
    # Guardamos el DataFrame en un CSV
    Datos.to_csv("output/Datos.csv", index=False)
    DatosNeg.to_csv("output/DatosNeg.csv", index=False)
    DatosNeu.to_csv("output/DatosNeu.csv", index=False)
    DatosPos.to_csv("output/DatosPos.csv", index=False)
    British.to_csv("output/British.csv", index=False)
    BritishNeg.to_csv("output/BritishNeg.csv", index=False)
    BritishNeu.to_csv("output/BritishNeu.csv", index=False)
    BritishPos.to_csv("output/BritishPos.csv", index=False)
    AirFrance.to_csv("output/AirFrance.csv", index=False)
    AirFranceNeg.to_csv("output/AirFranceNeg.csv", index=False)
    AirFranceNeu.to_csv("output/AirFranceNeu.csv", index=False)
    AirFrancePos.to_csv("output/AirFrancePos.csv", index=False)