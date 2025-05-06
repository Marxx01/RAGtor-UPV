# poli_gpt.py
import database_faiss_murta as faissdb # o database_faiss as faissdb
#import database_sql as db
import openai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


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
        index = faiss.IndexFlatL2(len(self.model.embed_query("dummy")))
        return faissdb.connect_faiss(
            embedding_model=self.model,
            index=index,
            faiss_index_dir=faiss_index_dir
        )

    def query_poligpt(self, query, k_context=3):
        """
        Consulta al sistema PoliGPT
        Args:
            query (str): Pregunta del usuario
            k_context (int): Número de resultados contextuales a considerar
        Returns:
            dict: Respuesta estructurada
        """
        # Búsqueda de contexto
        results = self.vector_store.similarity_search_with_score(query, k=k_context)
        
        if not results:
            return {"error": "No se encontró contexto relevante"}
        
        # Procesar resultados
        contextos = [{
            "id": doc.id,
            "content": doc.page_content,
            "score": score
        } for doc, score in results]
        
        # Construir prompt
        main_context = contextos[0]['content']
        prompt = self._build_prompt(query, main_context)
        
        # Obtener respuesta
        response = self.client.chat.completions.create(
            model="poligpt",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "query": query,
            "context_used": contextos[0]['id'],
            "response": response.choices[0].message.content,
            "contexts": contextos
        }

    @staticmethod
    def _build_prompt(query, context):
        """Construye el prompt estructurado"""
        return (
            "Responde EXCLUSIVAMENTE con la información del contexto proporcionado.\n\n"
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
#     query = "¿Cuántas horas de prácticas curriculares equivalen a un crédito ECTS?"
#     resultado = poligpt.query_poligpt(query)
    
#     print("\nRespuesta:", resultado['response'])
#     print("\nContexto usado (ID):", resultado['context_used'])