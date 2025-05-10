# poli_gpt.py
import database_faiss_murta as faissdb # o database_faiss as faissdb
#import database_sql as db
import openai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np # Added for type hinting based on your input example

class PoliGPT:
    def __init__(self, faiss_index_dir='./01_data/project_faiss'):
        # Configuración inicial
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
        self.client = openai.OpenAI(
            api_key="sk-1C-hWjmHEW05iQjwmr9EnA",
            base_url="https://api.poligpt.upv.es"
        )

        # Inicializar FAISS
        self.vector_store = self._connect_faiss(faiss_index_dir)

        # Verificar conexión
        print(f"FAISS inicializado - Vectores: {self.vector_store.index.ntotal} | Dimensión: {self.vector_store.index.d}")

    def _connect_faiss(self, faiss_index_dir):
        """Configura la conexión con FAISS"""
        import faiss
        # Ensure the dimension matches the embedding model
        try:
             embedding_dim = len(self.model.embed_query("dummy"))
        except Exception as e:
             print(f"Error getting embedding dimension: {e}")
             embedding_dim = 768 # Common dimension for LaBSE, fallback
             print(f"Using fallback embedding dimension: {embedding_dim}")

        index = faiss.IndexFlatL2(embedding_dim)
        return faissdb.connect_faiss(
            embedding_model=self.model,
            index=index,
            faiss_index_dir=faiss_index_dir
        )


    def query_poligpt(self, query, k_context=6):
        """
        Consulta al sistema PoliGPT
        Args:
            query (str): Pregunta del usuario
            k_context (int): Número de resultados contextuales a considerar
        Returns:
            dict: Respuesta estructurada con respuesta formateada y contextos (tanto list raw como string formateado)
        """
        # Búsqueda de contexto
        results = self.vector_store.similarity_search_with_score(query, k=k_context)

        if not results:
            # Format the response for no context found
            formatted_response = "**No se encontró contexto relevante para tu pregunta.**"
            formatted_contexts_string = "" # No contexts to list in string format
            raw_contexts_list = [] # No raw contexts to list
            return {
                "response": formatted_response,
                "contexts": formatted_contexts_string, # Keep for compatibility if needed elsewhere
                "raw_contexts": raw_contexts_list # <-- Added the raw list
            }

        # Procesar resultados y obtener solo el contenido y score
        raw_contexts_list = [{ # <-- Renamed variable for clarity
            "id": doc.id,
            "content": doc.page_content,
            "score": float(score) # Ensure score is a standard float
        } for doc, score in results]

        # Construir prompt usando el contexto principal (el de mayor score)
        main_context_content = raw_contexts_list[0]['content']
        prompt = self._build_prompt(query, main_context_content)

        # Obtener respuesta del modelo de lenguaje
        try:
            response = self.client.chat.completions.create(
                model="poligpt",
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response_text = response.choices[0].message.content
        except Exception as e:
            raw_response_text = f"Error al obtener respuesta del modelo: {e}"
            # In case of an API error, still return available context
            formatted_contexts_string = "\n".join([f"- {ctx['content']}" for ctx in raw_contexts_list])
            formatted_response = f"**{raw_response_text}**"
            return {
                "response": formatted_response,
                "contexts": formatted_contexts_string, # Keep for compatibility
                "raw_contexts": raw_contexts_list # <-- Added the raw list
            }

        # --- Formatting the output ---

        # 1. Format the main response (make it bold)
        formatted_response = f"**{raw_response_text}**"

        # 2. Format the contexts as a single string (still keeping this key)
        formatted_contexts_string = "\n".join([f"- {ctx['content']}" for ctx in raw_contexts_list])

        # Return the structured dictionary with formatted response and contexts
        return {
            "response": formatted_response,
            "contexts": formatted_contexts_string, # Keep for compatibility
            "raw_contexts": raw_contexts_list # <-- Added the raw list
        }


    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """Construye un prompt instruccional para normativa universitaria (ES)."""
        return (
            "Eres un **asistente académico** especializado en normativa y procedimientos "
            "de la universidad.\n\n"
            "════════ INSTRUCCIONES ════════\n"
            "1. **Usa exclusivamente el CONTEXTO**; no inventes nada que no esté presente.\n"
            "2. Intenta responder con la mayor cantidad de palabras exactas a al contexto.\n"
            "3. Incluye cifras, fechas o porcentajes *tal y como aparecen* en el contexto.\n"
            "4. Cita textualmente la norma relevante entre comillas y añade la referencia "
            "entre corchetes al final de la frase, p.ej.: \"…\" [Reglamento 2024, art.5].\n"
            "5. Si en el contexto no existe la información suficiente para contestar la respuesta, responde EXACTAMENTE:\n"
            "   «No dispongo de información suficiente en el contexto proporcionado.»\n"
            "6. Responde en el idioma de la respuesta.\n\n"
            "════════ FORMATO DE SALIDA ════════\n"
            "- Respuesta breve (máx. 3 frases) o lista con viñetas si hay varios puntos.\n"
            "- Cada afirmación respaldada por una cita.\n"
            "- No añadas secciones extra como \"Fuentes\" ni firmas.\n\n"
            "════════ PREGUNTA ════════\n"
            f"{query}\n\n"
            "════════ CONTEXTO ════════\n"
            "Ahora se va a suministrar los contextos relevantes para la respuesta, ten en cuenta que los contextos proporcionados con un menor score son los más relevantes para la pregunta formulada.\n"
            f"Contexto: {context}\n\n"
            "════════ FIN ════════"
        )

# # Ejemplo de uso desde otro script:
# if __name__ == "__main__":
#     # Inicialización
#     poligpt = PoliGPT()

#     # Consulta de ejemplo
#     # query = "¿Cuántas horas de prácticas curriculares equivalen a un crédito ECTS?"
#     query = "¿Cuántos créditos son una asignatura?" # Example query that might match your context

#     resultado = poligpt.query_poligpt(query)

#     print("\nRespuesta Formateada:\n", resultado['response'])
#     print("\nContextos Formateados:\n", resultado['contexts'])