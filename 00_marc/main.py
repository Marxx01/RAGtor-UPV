import database_faiss as faissdb
import database_sql as db
import openai

def response(client, query, context):
    response = client.chat.completions.create(
        model="poligpt",
        messages = [
            {
                "role": "user",
                "content": f"Responde a la siguiente pregunta usando el contexto suministrado. \n Pregunta: {query}, Contexto: {context}"
            }
        ]
    )

    return response.choices[0].message.content

query = "¿Cuantas horas de prácticas son equivalentes a 1 crédito?"

client = openai.OpenAI(
    api_key="sk-1C-hWjmHEW05iQjwmr9EnA",
    base_url="https://api.poligpt.upv.es"
)

# Check for updates in the sqlite database
db.update_faiss()
faissdb.update_faiss()
# Load the FAISS index
vector_store = faissdb.connect_faiss(embedding_model=faissdb.embedding_model, index=faissdb.index)

context = faissdb.obtain_context(query, vector_store)
print("Context obtained from FAISS index.")
print("Context: ", context)
response = response(client, query, context)
print("Response: ", response)

