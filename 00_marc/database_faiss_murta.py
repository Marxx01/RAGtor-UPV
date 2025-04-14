import os
import multiprocessing
import sqlite3
from typing import List, Tuple
from itertools import chain
import PyPDF2
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import database_sql as db
import torch
import faiss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
def connect_faiss(embedding_model, index, faiss_index_dir='./01_data/project_faiss'):
    """
    Connects to or creates a FAISS vector store in a subprocess.
    """
    try:
        vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Creating new FAISS vector store. Reason: {e}")
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

def determine_new_pdfs():
    """
    Determines the PDFs to be processed by consulting the database.
    """
    connection, cursor = db.connect_db('./01_data/project_database.db')
    cursor.execute("SELECT pdfs.path, pdfs.id FROM pdfs WHERE in_use = TRUE AND pdfs.id NOT IN (SELECT pdf_id FROM chunks);")
    pdfs = cursor.fetchall()
    db.disconnect_db(connection)
    return pdfs

def determine_chunks_to_delete():
    """
    Determines the chunks to be deleted by consulting the database.
    """
    connection, cursor = db.connect_db('./01_data/project_database.db')
    cursor.execute("SELECT chunks.id FROM pdfs, chunks WHERE in_use = FALSE AND checked = TRUE AND pdfs.id = chunks.pdf_id;")
    chunks = cursor.fetchall()
    db.disconnect_db(connection)
    return [chunk[0] for chunk in chunks]

def process_single_pdf(file_path: str, pdf_id: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
    """
    Processes a single PDF file: extracts text, cleans, splits into chunks.
    """
    pdf_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception as decrypt_error:
                    print(f"Warning: Could not decrypt {file_path} (ID: {pdf_id}). Skipping. Error: {decrypt_error}")
                    return []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = page_text.replace('\xa0', ' ').strip()  # Clean text
                        page_chunks_split = splitter.split_text(page_text)
                        for chunk in page_chunks_split:
                            pdf_chunks.append((chunk, page_num, pdf_id))
                except Exception as page_error:
                    print(f"Error processing page {page_num} of {file_path} (ID: {pdf_id}): {page_error}")
                    continue
    except FileNotFoundError:
        print(f"Error: File not found {file_path} (ID: {pdf_id}). Skipping.")
    except PyPDF2.errors.PdfReadError as pdf_error:
        print(f"Error reading PDF structure {file_path} (ID: {pdf_id}): {pdf_error}. Skipping.")
    except Exception as e:
        print(f"Error processing {file_path} (ID: {pdf_id}): {e}. Skipping.")
    return pdf_chunks

def read_pdfs_parallel(pdf_info_list: List[Tuple[str, int]], chunk_size: int = 500, chunk_overlap: int = 150, num_processes: int = None) -> List[Tuple[str, int, int]]:
    """
    Processes multiple PDFs in parallel using multiprocessing.Pool.
    """
    if not num_processes:
        num_processes = os.cpu_count()
    print(f"Using {num_processes} processes for PDF reading...")
    process_args = [(file_path, pdf_id, chunk_size, chunk_overlap) for file_path, pdf_id in pdf_info_list]
    all_results = []
    results_list_of_lists = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            results_list_of_lists = pool.starmap(process_single_pdf, process_args)
            all_results = list(chain.from_iterable(results_list_of_lists))
        except Exception as pool_error:
            print(f"An error occurred during parallel processing: {pool_error}")
            if 'all_results' not in locals():
                all_results = []
    total_pdfs = len(pdf_info_list)
    successful_pdfs = sum(1 for res_list in results_list_of_lists if isinstance(res_list, list))
    print(f"Parallel processing completed. Attempted {total_pdfs} PDFs, {successful_pdfs} processed (partially or fully).")
    print(f"Generated {len(all_results)} chunks in total.")
    return all_results

def update_chunks_in_db(chunks):
    """
    Inserts chunks into the database and retrieves their auto-generated IDs.
    """
    connection, cursor = db.connect_db('./01_data/project_database.db')
    stored_chunks = []
    for chunk_text, page_num, pdf_id in chunks:
        cursor.execute("INSERT INTO chunks (pdf_id, page) VALUES (?, ?)", (pdf_id, page_num))
        chunk_id = cursor.lastrowid
        stored_chunks.append((chunk_text, page_num, chunk_id))
    db.disconnect_db(connection)
    return stored_chunks

def delete_chunks_from_db(chunk_ids):
    """
    Deletes chunks from the database.
    """
    connection, cursor = db.connect_db('./01_data/project_database.db')
    for chunk_id in chunk_ids:
        cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    db.disconnect_db(connection)

def insert_faiss(chunks, vector_store):
    """
    Inserts chunks into the FAISS index using LangChain, with custom chunk_ids as document IDs.
    """
    embedding_model = vector_store.embedding_function
    documents = [
        Document(
            page_content=chunk,
            metadata={"page": page_num}
        )
        for chunk, page_num, chunk_id in chunks
    ]
    ids = [str(chunk_id) for _, _, chunk_id in chunks]
    vector_store.add_documents(documents, embedding=embedding_model, ids=ids)
    print("Chunks inserted into FAISS index.")
    return True

def delete_faiss(chunk_ids, faiss_index):
    """
    Deletes chunks from the FAISS index by chunk_id.
    """
    ids_to_delete = [str(chunk_id) for chunk_id in chunk_ids]
    faiss_index.delete(ids_to_delete)
    print("Chunks deleted from FAISS index.")
    return True

def update_faiss(vector_store):
    """
    Updates the FAISS index with the latest PDF chunks.
    """
    print('Determining new PDFs to process...')
    pdfs_to_process = determine_new_pdfs()
    print('Determining chunks to delete...')
    chunks_to_delete = determine_chunks_to_delete()
    if chunks_to_delete:
        print("Deleting chunks from FAISS index...")
        delete_faiss(chunks_to_delete, vector_store)
        print("Deleting chunks from the database...")
        delete_chunks_from_db(chunks_to_delete)
    print('Reading new PDFs...')
    new_chunks = read_pdfs_parallel(pdfs_to_process)
    print('Inserting new chunks into the database and retrieving their IDs...')
    stored_chunks = update_chunks_in_db(new_chunks)
    print('Inserting new chunks into the FAISS index...')
    insert_faiss(stored_chunks, vector_store)
    commit_faiss(vector_store)
    print("FAISS index updated consistently.")
    return True

def obtain_context(query, vector_store):
    """
    Obtains context from the FAISS index based on the query.
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
    # Verifica si hay GPU disponible y establece el dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Inicializa el modelo de embeddings configurando el dispositivo (si es soportado)
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE", device=device)

    # Determinar la dimensión de los embeddings
    embedding_dim = len(model.embed_query("hello world"))

    # Crea el índice FAISS en CPU
    cpu_index = faiss.IndexFlatL2(embedding_dim)

    # Intenta transferir el índice a la GPU, si es posible
    if device == "cuda":
        try:
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
            print("GPU available. Using GPU for FAISS index.")
        except Exception as e:
            print("Error al inicializar FAISS en GPU, se usará CPU en su lugar. Error:", e)
            index = cpu_index
    else:
        index = cpu_index

    # Conecta o crea el vector store FAISS
    vector_store = connect_faiss(embedding_model=model, index=index)
    commit_faiss(vector_store)

    # Actualiza el índice FAISS
    print("Updating FAISS index...")
    update_faiss(vector_store)
