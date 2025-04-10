import os
import sqlite3
import PyPDF2
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def determine_new_pdfs():
    """
    Determines the PDFs to be processed by consulting the database.
    """
    connection, cursor = db.connect_db()
    cursor.execute("SELECT pdfs.path, pdfs.id FROM pdfs, chunks WHERE in_use = TRUE AND pdfs.id not in chunks.pdf_id;")
    pdfs = cursor.fetchall()
    db.disconnect_db(connection)
    return pdfs

def determine_chunks_to_delete():
    """
    Determines the chunks to be deleted by consulting the database.
    """
    connection, cursor = db.connect_db()
    cursor.execute("SELECT chunks.id FROM pdfs, chunks WHERE in_use = FALSE AND checked = TRUE AND pdfs.id in chunks.pdf_id;")
    chunks = cursor.fetchall()
    db.disconnect_db(connection)
    return [chunk[0] for chunk in chunks]

def read_pdfs(pdf_info_list, chunk_size=1000, chunk_overlap=200):
    """
    Processes multiple PDFs and returns a list of tuples (chunk_text, page_number, pdf_id).

    Args:
        pdf_info_list (List[Tuple[str, int]]): List of (path_to_pdf, pdf_id)
        chunk_size (int): Maximum chunk size
        chunk_overlap (int): Overlap between chunks

    Returns:
        List[Tuple[str, int, int]]: List of (chunk, page_number, pdf_id)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    all_chunks = []

    for file_path, pdf_id in pdf_info_list:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()

                    if page_text:
                        # Clean and normalize text
                        page_text = page_text.replace('\xa0', ' ').strip()

                        # Split the page text into chunks
                        page_chunks = splitter.split_text(page_text)

                        # Append each chunk with page number and PDF ID
                        for chunk in page_chunks:
                            all_chunks.append((chunk, page_num, pdf_id))
        except Exception as e:
            print(f"Error processing {file_path} (PDF ID: {pdf_id}): {e}")

    return all_chunks

def update_chunks_in_db(chunks):
    """
    Updates the chunks in the database.
    """
    connection, cursor = db.connect_db()
    for chunk, page_num, pdf_id in chunks:
        cursor.execute("INSERT INTO chunks (chunk, pdf_id, page) VALUES (?, ?, ?)", (chunk, pdf_id, page_num))
    db.commit_db(connection)
    db.disconnect_db(connection)

def delete_chunks_from_db(chunk_ids):
    """
    Deletes chunks from the database.
    """
    connection, cursor = db.connect_db()
    for chunk_id in chunk_ids:
        cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    db.commit_db(connection)
    db.disconnect_db(connection)

def insert_faiss(chunks):
    """
    Inserts chunks into the FAISS index.
    
    Args:
        chunks (List[Tuple[str, int, int]]): List of (chunk_text, page_number, chunk_id)
    """
    for chunk, page_num, chunk_id in chunks:
        vector_store.add(chunk, chunk_id)
    print("Chunks inserted into FAISS index.")
    return True

def delete_faiss(chunk_ids):
    """
    Deletes chunks from the FAISS index.
    
    Args:
        chunk_ids (List[int]): List of chunk IDs to delete
    """
    for chunk_id in chunk_ids:
        vector_store.delete(chunk_id)
    print("Chunks deleted from FAISS index.")
    return True

def update_faiss(vector_store):
    
    """
    Updates the FAISS index with the latest chunks.
    """
    pdfs_to_process = determine_new_pdfs()
    chunks_to_delete = determine_chunks_to_delete()
    new_chunks = read_pdfs(pdfs_to_process)
    insert_faiss(new_chunks)
    delete_chunks_from_db(chunks_to_delete)
    delete_faiss(chunks_to_delete)
    commit_faiss(vector_store)
    print("FAISS index updated.")
    return True

def obtain_context(query, vector_store):
    """
    Obtains context from the FAISS index based on the query.
    
    Args:
        query (str): The query string to search for.
        vector_store: The FAISS vector store to search in.
    """

    results = vector_store.similarity_search_with_score(query, k=3)
    context = []
    for i, (doc, score) in enumerate(results):
        context.append(f"{i+1}. {doc} (Score: {score})")
    return context




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