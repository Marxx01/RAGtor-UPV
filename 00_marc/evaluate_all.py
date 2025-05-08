import os

modelos = ["sentence-transformers/LaBSE", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/multi-qa-mpnet-base-dot-v1"]
chunks = [(1000, 50), (300, 100), (200, 50)]
ks = [2, 4, 6]

for modelo in modelos:
    for chunk in chunks:
        print(f"Modelo: {modelo}, Chunk: {chunk}")

        # Detectar si esta la base de datos sql y faiss 
        if os.path.exists("01_data/project_database.db"):
            os.remove("01_data/project_database.db")
            print("Base de datos sql eliminada")
        if os.path.exists("01_data/faiss_index"):
            os.remove("01_data/faiss_index")
            print("Base de datos faiss eliminada")

        # Ejecutar el script de creacion de la base de datos sql
        os.system(f"python 00_marc/database_sql.py")
        print("Base de datos sql creada")
        
        os.system(f"python 00_marc/database_faiss_murta.py {modelo} {chunk[0]} {chunk[1]}")
        print("Base de datos faiss creada")

        # Ejecutar el script de evaluacion
        for k in ks:
            print(f"Evaluando con k = {k}")
            # Ejecutar el script de obtencion de metricas
            os.system(f"python 00_marc/obtain_metrics.py {modelo} {chunk[0]} {chunk[1]} {k}")

            