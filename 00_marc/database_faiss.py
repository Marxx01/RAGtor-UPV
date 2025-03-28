import os
import time
import sqlite3
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import database_sql as db

# =============================================================================
# FAISS Functions
# =============================================================================

def connect_faiss(embedding_model, index, faiss_index_dir='./01_data/project_faiss'):
    """
    Connects to or creates a FAISS vector store in a subprocess.
    """
    try:
        vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization = True)
    except:
        vector_store = FAISS(
            embedding_function=embedding_model,  
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
            )
    return vector_store


def commit_faiss(vector_store, faiss_index_dir='./01_data/project_faiss'):
    """
    Commits the FAISS vector store to disk.
    """
    vector_store.save_local(faiss_index_dir)
    print("FAISS index saved.")
    return True

# =============================================================================
# PDF Processing Functions
# =============================================================================

def load_and_split_pdf(pdf_path, chunk_size=768, overlap_size=50):
    """
    Loads a PDF and splits it into chunks.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, overlap_size=overlap_size)
        document_chunks = text_splitter.split_documents(documents)
        return document_chunks
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []


def update_faiss():
    """
    Orchestrates the update of the FAISS vector store.
    """
    faiss_index_path = connect_faiss()
    print(f"FAISS index stored at: {faiss_index_path}")
    
    pdfs = determine_pdfs()
    if pdfs:
        print("Adding new PDFs to FAISS...")
        # Your logic to add PDFs to FAISS would go here
    else:
        print("No new PDFs to add.")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Load the embedding model
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    embedding_dim = len(model.embed_query("hello world"))
    import faiss
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = connect_faiss(embedding_model=model, index=index)
    commit_faiss(vector_store)