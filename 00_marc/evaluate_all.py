import os
import shutil

modelos = ["sentence-transformers/LaBSE"] #["sentence-transformers/LaBSE", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/multi-qa-mpnet-base-dot-v1"]
chunks = [(500, 150)] #[(500, 150), (300, 100), (200, 50)]
ks =  [6] #[2, 4, 6]

for modelo in modelos:
    for chunk in chunks:
        print(f"Modelo: {modelo}, Chunk: {chunk}")

        # Detectar si esta la base de datos sql y faiss 
        if os.path.exists("01_data/project_database.db"):
            os.remove("01_data/project_database.db")
            print("Base de datos sql eliminada")

            os.system(f"python 00_marc/database_sql.py")
            print("Base de datos sql creada")
        
        else:
            os.system(f"python 00_marc/database_sql.py")
            print("Base de datos sql creada")

        if os.path.exists("01_data/project_faiss"):
            shutil.rmtree("01_data/project_faiss")
            print("Base de datos faiss eliminada")

            os.system(f"python 00_marc/database_faiss_murta.py {modelo} {chunk[0]} {chunk[1]}")
            print("Base de datos faiss creada")
        else:
            os.system(f"python 00_marc/database_faiss_murta.py {modelo} {chunk[0]} {chunk[1]}")
            print("Base de datos faiss creada")
        # Ejecutar el script de evaluacion
        for k in ks:
            print(f"Evaluando con k = {k}")
            # Ejecutar el script de obtencion de metricas
            os.system(f"python 00_marc/obtain_metrics.py {modelo} {chunk[0]} {chunk[1]} {k}")

            