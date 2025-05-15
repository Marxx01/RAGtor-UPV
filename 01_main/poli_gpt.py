# poli_gpt.py
import os
import asyncio
import nest_asyncio
import torch
import openai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import database_faiss_murta as faissdb
import time
import httpx
from typing import Optional, Dict, Any, List
# Set environment variables
os.environ["STATIC_DIRECTORY"] = os.path.dirname(os.path.abspath(__file__))

# Fix PyTorch class path issue
if hasattr(torch.classes, '__dict__'):
    torch.classes.__dict__['_path'] = []

class PoliGPT:
    def __init__(self, faiss_index_dir='./01_data/project_faiss'):
        # Set up event loop
        self._setup_event_loop()
        
        # Initialize components
        self._init_embeddings()
        self._init_openai_client()
        self._init_faiss(faiss_index_dir)

    def _setup_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        nest_asyncio.apply()

    def _init_embeddings(self):
        self.model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )


    def _init_openai_client(self):
        self.client = openai.OpenAI(
            api_key="sk-1mayqHi3ba5ey5jfTZaYeQ",
            base_url="https://api.poligpt.upv.es",
            timeout=300.0,  # Timeout más corto por intento
            max_retries=5  # Más reintentos
        )

    def _init_faiss(self, faiss_index_dir):
        try:
            embedding_dim = len(self.model.embed_query("dummy"))
        except Exception as e:
            print(f"Error getting embedding dimension: {e}")
            embedding_dim = 768
        
        import faiss
        index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = faissdb.connect_faiss(
            embedding_model=self.model,
            index=index,
            faiss_index_dir=faiss_index_dir
        )

        # Verify FAISS connection
        if hasattr(self.vector_store, 'index'):
            print(f"FAISS initialized - Vectors: {self.vector_store.index.ntotal} | Dimension: {self.vector_store.index.d}")

    def query_poligpt(self, query, k_context=3, max_retries=5):
        print("[PoliGPT] Recibida query:", query)
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k_context)
            print(f"[PoliGPT] Resultados de FAISS: {len(results)} encontrados")
            if not results:
                print("[PoliGPT] No se encontró contexto relevante.")
                return {"error": "No se encontró contexto relevante"}
            
            contextos = [{
                "id": doc.id,
                "content": doc.page_content,
                "score": score
            } for doc, score in results]
            print(f"[PoliGPT] IDs de contexto: {[c['id'] for c in contextos]}")

            main_context = contextos[0]['content']
            prompt = self._build_prompt(query, main_context)

            # Implementar reintentos con backoff exponencial
            for attempt in range(max_retries):
                try:
                    print(f"[PoliGPT] Intento {attempt + 1} de {max_retries}...")
                    start_time = time.time()
                    
                    # Timeout progresivo
                    timeout = min(15 * (1.5 ** attempt), 60)  # Incrementa el timeout pero max 60s
                    
                    response = self.client.chat.completions.create(
                        model="poligpt",
                        messages=[{"role": "user", "content": prompt}],
                        timeout=timeout
                    )
                    


                    elapsed = time.time() - start_time
                    print(f"[PoliGPT] Respuesta exitosa en {elapsed:.2f}s (intento {attempt + 1})")
                    
                    raw_response_text = response.choices[0].message.content
                    break  # Si llegamos aquí, salimos del bucle
                    
                except (openai.APITimeoutError, httpx.ConnectTimeout) as e:
                    print(f"[PoliGPT] Timeout en intento {attempt + 1}: {str(e)}")
                    wait_time = 2 * (1.5 ** attempt)  # Backoff exponencial
                    
                    if attempt < max_retries - 1:
                        print(f"[PoliGPT] Esperando {wait_time:.1f}s antes del siguiente intento...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # En el último intento, devolver respuesta parcial si la hay
                        raw_response_text = "La respuesta está tardando más de lo habitual, pero aquí tienes la información del contexto que puede ser relevante."
                        break
                        
                except Exception as e:
                    print(f"[PoliGPT] Error no esperado en intento {attempt + 1}: {str(e)}")
                    raw_response_text = f"**Error al obtener respuesta del modelo: {str(e)}**"
                    break

            return {
                "response": raw_response_text,
                "contexts": "\n".join([f"- {ctx['content']}" for ctx in contextos]),
                "raw_contexts": contextos
            }
        except Exception as e:
            print("[PoliGPT] Error general en query_poligpt:", repr(e))
            return {
                "response": f"**Error en el procesamiento: {str(e)}**",
                "contexts": "",
                "raw_contexts": []
            }

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """Construye un prompt instruccional para normativa universitaria (ES)."""
        return (
            "Responde con la información del contexto proporcionado.\n\n"
            f"Pregunta: {query}\n"
            f"Contexto: {context}\n\n"
            "Formato requerido:\n"
            "- Respuesta directa y concisa\n"
            "- Incluir valores numéricos cuando aplique\n"
            "- Citar la normativa relevante\n"
            "- Si no hay información suficiente, indicarlo explícitamente"
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

# #poli_gpt.py Gemini version

# import os
# import asyncio
# import nest_asyncio
# import torch
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# import database_faiss_murta as faissdb
# import time
# import requests

# class PoliGPT:
#     def __init__(self, faiss_index_dir='./01_data/project_faiss'):
#         self._setup_event_loop()
#         self._init_embeddings()
#         self.hf_token = "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"  # <-- Pega aquí tu token de Hugging Face
#         self.hf_model = "HuggingFaceH4/zephyr-7b-beta"  # Puedes cambiar por otro modelo open source
#         self._init_faiss(faiss_index_dir)

#     def _setup_event_loop(self):
#         try:
#             loop = asyncio.get_event_loop()
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#         nest_asyncio.apply()

#     def _init_embeddings(self):
#         self.model = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/LaBSE",
#             model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
#         )

#     def _init_faiss(self, faiss_index_dir):
#         try:
#             embedding_dim = len(self.model.embed_query("dummy"))
#         except Exception as e:
#             print(f"Error getting embedding dimension: {e}")
#             embedding_dim = 768

#         import faiss
#         index = faiss.IndexFlatL2(embedding_dim)
#         self.vector_store = faissdb.connect_faiss(
#             embedding_model=self.model,
#             index=index,
#             faiss_index_dir=faiss_index_dir
#         )

#         if hasattr(self.vector_store, 'index'):
#             print(f"FAISS initialized - Vectors: {self.vector_store.index.ntotal} | Dimension: {self.vector_store.index.d}")

#     def query_poligpt(self, query, k_context=3):
#         print("[PoliGPT] Recibida query:", query)
#         try:
#             results = self.vector_store.similarity_search_with_score(query, k=k_context)
#             print(f"[PoliGPT] Resultados de FAISS: {len(results)} encontrados")
#             if not results:
#                 print("[PoliGPT] No se encontró contexto relevante.")
#                 return {"error": "No se encontró contexto relevante"}

#             contextos = [{
#                 "id": doc.id,
#                 "content": doc.page_content,
#                 "score": score
#             } for doc, score in results]
#             print(f"[PoliGPT] IDs de contexto: {[c['id'] for c in contextos]}")

#             main_context = contextos[0]['content']
#             prompt = self._build_prompt(query, main_context)

#             # Llama a Hugging Face Inference API
#             api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
#             headers = {"Authorization": f"Bearer {self.hf_token}"}
#             payload = {
#                 "inputs": prompt,
#                 "parameters": {"max_new_tokens": 512, "temperature": 0.7}
#             }
#             print(f"[PoliGPT] Llamando a Hugging Face API: {self.hf_model}")
#             response = requests.post(api_url, headers=headers, json=payload)
#             if response.status_code == 200:
#                 result = response.json()
#                 # El formato depende del modelo, pero normalmente es así:
#                 if isinstance(result, list) and "generated_text" in result[0]:
#                     raw_response_text = result[0]["generated_text"]
#                 elif isinstance(result, dict) and "generated_text" in result:
#                     raw_response_text = result["generated_text"]
#                 else:
#                     raw_response_text = str(result)
#             else:
#                 raw_response_text = f"Error llamando a Hugging Face API: {response.status_code} - {response.text}"

#             return {
#                 "response": raw_response_text,
#                 "contexts": "\n".join([f"- {ctx['content']}" for ctx in contextos]),
#                 "raw_contexts": contextos
#             }
#         except Exception as e:
#             print("[PoliGPT] Error general en query_poligpt:", repr(e))
#             return {
#                 "response": f"**Error en el procesamiento: {str(e)}**",
#                 "contexts": "",
#                 "raw_contexts": []
#             }

#     @staticmethod
#     def _build_prompt(query: str, context: str) -> str:
#         return (
#             "Responde con la información del contexto proporcionado.\n\n"
#             f"Pregunta: {query}\n"
#             f"Contexto: {context}\n\n"
#             "Formato requerido:\n"
#             "- Respuesta directa y concisa\n"
#             "- Incluir valores numéricos cuando aplique\n"
#             "- Citar la normativa relevante\n"
#             "- Si no hay información suficiente, indicarlo explícitamente"
#         )