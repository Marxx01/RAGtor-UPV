from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Crear el objeto de embeddings usando LaBSE
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

# Obtener la dimensión de los embeddings
embedding_dim = len(embedding_model.embed_query("hello world"))

import faiss

index = faiss.IndexFlatL2(embedding_dim)

# Inicializar FAISS con el objeto de embeddings correcto
vector_store = FAISS(
    embedding_function=embedding_model,  
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

texts = ["inteligencia artificial y aprendizaje", "aprendizaje profundo y redes neuronales", "procesamiento de lenguaje natural y transformers"]

# Agregar documentos al vector store
vector_store.add_texts(texts)

print("Documentos agregados a FAISS.")

# Guardar el índice en local
vector_store.save_local("faiss_index")

print("Índice FAISS guardado en local.")

vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization = True)

query = "¿Qué es el aprendizaje profundo?"

resultados = vector_store.similarity_search_with_score(query, k=3)

# Mostrar resultados con similitud
for i, (doc, score) in enumerate(resultados):
    print(f"{i+1}. {doc} (Score: {score})")

