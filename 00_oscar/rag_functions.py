from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Importación original
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from transformers import AutoTokenizer
import os

# Cambia esta importación

class _RAGSystem:
    def __init__(self):
        self._initialize()
    
    def _initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            trust_remote_code=True
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = None
        self.vector_store = None

    def initialize_llm(self):
        try:
            self.llm = Ollama(model="deepseek-llm:7b")  # Reemplaza deepseek-rl:14b
            return True
        except Exception as e:
            print(f"Error inicializando Ollama: {str(e)}")
            return False

    def load_and_process_pdfs(self, pdf_paths):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False))
        )
        
        chunks = []
        for path in pdf_paths:
            if not os.path.exists(path):
                print(f"Archivo no encontrado: {path}")
                continue
                
            try:
                loader = PyPDFLoader(path)
                pages = loader.load()
                for page in pages:
                    splitted_texts = text_splitter.split_text(page.page_content)
                    for text in splitted_texts:
                        chunks.append(Document(
                            page_content=text,
                            metadata={
                                "page": page.metadata.get("page", 0),
                                "file": os.path.basename(path)
                            }
                        ))
            except Exception as e:
                print(f"Error procesando {path}: {str(e)}")
        return chunks

    def setup_vector_store(self, chunks, persist_dir="chroma_db_nuevo"):
        if os.path.exists(persist_dir):
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

    def generate_response(self, query):
        if not self.vector_store or not self.llm:
            return "Sistema no inicializado correctamente", []
            
        retrieved_docs = self.vector_store.similarity_search_with_score(query, k=3)
        context = "\n\n".join([
            f"📄 {doc.metadata['file']} (Página {doc.metadata['page']+1})\n{doc.page_content}"
            for doc, score in retrieved_docs
        ])
        
        prompt = f"""Responde en español usando esta información:
[CONTEXTO]
{context}

[PREGUNTA]
{query}"""
        
        try:
            response = self.llm.invoke(prompt)
            return response, retrieved_docs
        except Exception as e:
            return f"Error generando respuesta: {str(e)}", []

_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
        _rag_system.initialize_llm()
    return _rag_system

# Funciones de interfaz compatibles con tu código actual
def load_and_process_pdfs(pdf_paths):
    return get_rag_system().load_and_process_pdfs(pdf_paths)

def setup_vector_store(chunks, persist_dir="chroma_db_nuevo"):
    return get_rag_system().setup_vector_store(chunks, persist_dir)

def generate_rag_response(query, vector_store):
    rag = get_rag_system()
    rag.vector_store = vector_store
    return rag.generate_response(query)

RAGSystem = _RAGSystem
