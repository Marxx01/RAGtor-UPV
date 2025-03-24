import os
import time
import sqlite3
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import database_sql as db

# =============================================================================
# FAISS Functions
# =============================================================================

def create_empty_faiss(embedding_model):
    """
    Creates an empty FAISS vector store by initializing an empty FAISS index,
    and creating empty docstore and index_to_docstore_id structures.
    """
    # Use a dummy query to get the embedding dimension
    dummy_embedding = embedding_model.embed_query("dummy")
    print(dummy_embedding)
    print('error')
    # Create a FAISS index (using L2 distance)
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Instantiate the FAISS vector store using the low-level constructor
    faiss.write_index(index, "faiss_db.index")
    print("Empty FAISS index created.")
    return None

def connect_faiss(faiss_index_dir='./01_data/project_faiss', embedding_model_name='sentence-transformers/LaBSE'):
    """
    Connects to or creates a FAISS vector store using the specified embeddings model.
    If the directory exists and contains data, the vector store is loaded;
    otherwise, a new empty vector store is created.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    if os.path.exists(faiss_index_dir) and os.listdir(faiss_index_dir):
        try:
            faiss_vector_store = FAISS.load_local(faiss_index_dir, embedding_model)
            print("Vector store loaded.")
        except Exception as e:
            print("Error loading vector store:", e)
            # Create a new empty vector store if loading fails
            faiss_vector_store = create_empty_faiss(embedding_model)
            print("New vector store created.")
    else:
        # Create an empty vector store from an empty list of texts
        faiss_vector_store = create_empty_faiss(embedding_model)
        print("Vector store created.")
        # Save locally for future loads
        faiss_vector_store.save_local(faiss_index_dir)
    
    return faiss_vector_store

# =============================================================================
# PDF Processing Functions
# =============================================================================

def load_and_split_pdf(pdf_path, chunk_size=768, overlap_size=50):
    """
    Loads a PDF from the specified path, processes it, and returns text chunks.
    
    Parameters:
      pdf_path (str): Path to the PDF file.
      chunk_size (int): Maximum size of each text chunk.
      overlap_size (int): Number of overlapping characters between chunks.
    
    Returns:
      A list of document chunks (with metadata).
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

def add_pdf_to_faiss(vector_store, pdfs):
    """
    Processes and adds the provided PDFs to the FAISS vector store and records the information in the SQLite database.
    
    Parameters:
      vector_store: FAISS vector store instance.
      pdfs (list): List of tuples (pdf_path, pdf_id).
      
    Returns:
      The updated vector store.
    """
    # Use a single database connection for all PDFs
    connection, cursor = db.connect_db()
    
    for pdf_info in pdfs:
        pdf_path, pdf_id = pdf_info
        
        # Load and split the PDF into chunks
        doc_chunks = load_and_split_pdf(pdf_path)
        if not doc_chunks:
            continue
        
        for i, doc in enumerate(doc_chunks):
            chunk_id = f"{pdf_id}_{i}"
            page = doc.metadata.get('page_label', None)
            try:
                vector_store.add_document(doc, chunk_id)
                print(f"Added document {chunk_id} to the FAISS vector store.")
                
                # Insert record into the 'chunks' table (assuming table has 3 columns: id, pdf_id, page)
                cursor.execute(
                    "INSERT INTO chunks (id, pdf_id, page) VALUES (?, ?, ?)",
                    (chunk_id, pdf_id, page)
                )
            except Exception as e:
                print(f"Error adding chunk {chunk_id}: {e}")
    
    db.disconnect_db(connection)
    return vector_store

def determine_pdfs(db_path="./01_data/project_database.db"):
    """
    Determines which PDFs need to be added to the FAISS vector store.
    Returns a list of tuples (pdf_path, pdf_id) for those PDFs that are in use,
    have been checked, and are not yet in the 'chunks' table.
    """
    connection, cursor = db.connect_db(db_path)
    try:
        query = """
        SELECT pdfs.path, pdfs.id 
        FROM pdfs 
        WHERE pdfs.in_use = TRUE 
          AND pdfs.checked = TRUE 
          AND pdfs.id NOT IN (SELECT pdf_id FROM chunks)
        """
        cursor.execute(query)
        pdfs = cursor.fetchall()
        print(f"Found {len(pdfs)} PDFs to add to the FAISS vector store.")
        return pdfs
    except Exception as e:
        print("Error determining PDFs:", e)
        return []
    finally:
        db.disconnect_db(connection)

def update_faiss():
    """
    Orchestrates the update of the FAISS vector store:
      - Connects to or creates the FAISS vector store.
      - Determines which PDFs need to be added.
      - Processes and adds the PDFs to the vector store and updates the database.
    """
    # Connect or create the FAISS vector store
    faiss_vector_store = connect_faiss()
    # Determine which PDFs should be added
    pdfs = determine_pdfs()
    
    if pdfs:
        faiss_vector_store = add_pdf_to_faiss(faiss_vector_store, pdfs)
    else:
        print("No new or modified PDFs to add.")
    
    return faiss_vector_store

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print('aaa')
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    create_empty_faiss(hf)


