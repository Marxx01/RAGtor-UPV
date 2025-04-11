import database_faiss as faissdb
import database_sql as db
import openai
from langchain.embeddings import HuggingFaceEmbeddings
import faiss

def get_response(client, query, context):
    response = client.chat.completions.create(
        model="poligpt",
        messages = [
            {
                "role": "user",
                "content": f"Responde a la siguiente pregunta usando el contexto suministrado.\n\nPregunta: {query}\nContexto: {context}"
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = "¿Cuántas horas de prácticas son equivalentes a 1 crédito?"

    client = openai.OpenAI(
        api_key="sk-1C-hWjmHEW05iQjwmr9EnA",
        base_url="https://api.poligpt.upv.es"
    )

    # Check for updates in the SQLite database
    db.update_faiss()

    # Load the FAISS index
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    embedding_dim = len(model.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = faissdb.connect_faiss(embedding_model=model, index=index)
    faissdb.commit_faiss(vector_store)

    # Retrieve context
    context_list = faissdb.obtain_context(query, vector_store)
    print("Context obtained from FAISS index.")

    if not context_list:
        print("No se encontró contexto relevante.")
    else:
        context_text = "\n".join(context_list)
        answer = get_response(client, query, context_text)
        print("Response:", answer)