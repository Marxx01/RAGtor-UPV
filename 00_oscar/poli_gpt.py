import sys

# Lista para guardar los mensajes de error de importación
missing_modules = []

# --- Intentar importar cada módulo necesario ---

try:
    import faiss
except ImportError:
    missing_modules.append("faiss (Intenta instalar con: pip install faiss-cpu o pip install faiss-gpu)")

try:
    # Intentamos importar primero database_faiss_murta
    import database_faiss_murta as faissdb
    # Si falla, intentamos database_faiss
except ImportError:
    try:
        import database_faiss as faissdb
    except ImportError:
        missing_modules.append("database_faiss_murta o database_faiss (Asegúrate de que alguno de estos archivos personalizados esté en tu directorio de proyecto)")

try:
    import database_sql as db
except ImportError:
    missing_modules.append("database_sql (Asegúrate de que este archivo personalizado esté en tu directorio de proyecto)")

try:
    import openai
except ImportError:
    missing_modules.append("openai (Intenta instalar con: pip install openai)")

try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
except ImportError:
    missing_modules.append("langchain_huggingface (Intenta instalar con: pip install langchain-huggingface)")


# --- Verificar si faltan módulos y salir si es necesario ---

if missing_modules:
    print("\n" + "="*60)
    print("ERROR: Faltan uno o más módulos requeridos para ejecutar poli_gpt.py")
    print("Por favor, instala los módulos listados a continuación:")
    print("-" * 60)
    for module_info in missing_modules:
        print(f"- {module_info}")
    print("="*60 + "\n")
    sys.exit(1) # Sale del script con un código de error

# Si llegamos aquí, todos los módulos se importaron correctamente
print("Todos los módulos requeridos para poli_gpt.py importados correctamente.")


class PoliGPT:
    def __init__(self, faiss_index_dir='../01_data/project_faiss'):
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

# Ejemplo de uso desde otro script:
if __name__ == "__main__":
    # Inicialización
    poligpt = PoliGPT()
    
    # Consulta de ejemplo
    query = "¿Cuántas horas de prácticas curriculares equivalen a un crédito ECTS?"
    resultado = poligpt.query_poligpt(query)
    
    print("\nRespuesta:", resultado['response'])
    print("\nContexto usado (ID):", resultado['context_used'])