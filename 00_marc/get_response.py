import database_faiss_murta as faissdb # o database_faiss as faissdb
import database_sql as db
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
        # 1 ) Buscar contexto
        results = self.vector_store.similarity_search_with_score(query, k=k_context)
        if not results:
            return {"error": "No se encontró contexto relevante"}

        # 2 ) Construir lista de líneas con el formato deseado
        lineas = []
        for i, (doc, score) in enumerate(results, start=1):
            lineas.append(f"- Contexto {i}: texto: {doc.page_content} score: {score:.4f}")

        contextos_fmt = "Contextos usados:\n" + "\n".join(lineas)

        # 3 ) Construir prompt usando el string recién creado
        prompt = self._build_prompt(query, contextos_fmt)

        # 4 ) Llamar al modelo
        response = self.client.chat.completions.create(
            model="poligpt",
            messages=[{"role": "user", "content": prompt}]
        )

        # 5 ) Devolver resultado con la clave correcta
        return {
            "query": query,
            "response": response.choices[0].message.content,
            "context_used": results
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
            "entre corchetes al final de la frase, p.ej.: \"…\" [Reglamento 2024, art. 5].\n"
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

# Ejemplo de uso desde otro script:
if __name__ == "__main__":
    # Inicialización
    poligpt = PoliGPT()
    
    # Consulta de ejemplo
    query = "¿Cuántas horas de prácticas curriculares equivalen a un crédito ECTS?"
    resultado = poligpt.query_poligpt(query)

    print("\nRespuesta:", resultado['response'])
    print("\nContexto usado (ID):", resultado['context_used'])