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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
def connect_faiss(embedding_model, index, faiss_index_dir='./01_data/project_faiss'):
    """
    Connects to or creates a FAISS vector store in a subprocess.
    """
    try:
        vector_store = FAISS.load_local(faiss_index_dir, embedding_model, allow_dangerous_deserialization = True)
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

# --- Worker Function (Processes a single PDF) ---
# This function will run in separate processes.
# It must take all necessary arguments and return the chunks for THAT pdf.
def process_single_pdf(file_path: str, pdf_id: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
    """
    Processes a single PDF file: extracts text, cleans, splits into chunks.

    Args:
        file_path (str): Path to the PDF file.
        pdf_id (int): Unique identifier for this PDF.
        chunk_size (int): Maximum chunk size.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Tuple[str, int, int]]: List of (chunk, page_number, pdf_id) for this PDF.
                                     Returns an empty list if an error occurs with this PDF.
    """
    pdf_chunks = []
    # It's good practice to create the splitter inside the worker if not too costly,
    # to avoid serialization issues (although RecursiveCharacterTextSplitter is usually safe).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Basic handling for encrypted PDFs (try without password)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception as decrypt_error:
                    # Warn but do not stop the entire process
                    print(f"Warning: Could not decrypt {file_path} (ID: {pdf_id}). Skipping. Error: {decrypt_error}")
                    return [] # Return empty list for this PDF

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = page_text.replace('\xa0', ' ').strip() # Clean text
                        page_chunks_split = splitter.split_text(page_text)
                        for chunk in page_chunks_split:
                            pdf_chunks.append((chunk, page_num, pdf_id))
                except Exception as page_error:
                     # Error processing a specific page, log and continue if possible
                     print(f"Error processing page {page_num} of {file_path} (ID: {pdf_id}): {page_error}")
                     continue # Skip to the next page

    except FileNotFoundError:
        print(f"Error: File not found {file_path} (ID: {pdf_id}). Skipping.")
    except PyPDF2.errors.PdfReadError as pdf_error:
        print(f"Error reading PDF structure {file_path} (ID: {pdf_id}): {pdf_error}. Skipping.")
    except Exception as e:
        # Catch other unexpected errors during file processing
        print(f"Error processing {file_path} (ID: {pdf_id}): {e}. Skipping.")

    return pdf_chunks # Return the chunks from this PDF

# --- Main Function (Orchestrator) ---
def read_pdfs_parallel(pdf_info_list: List[Tuple[str, int]], chunk_size: int = 500, chunk_overlap: int = 150, num_processes: int = None) -> List[Tuple[str, int, int]]:
    """
    Processes multiple PDFs in parallel using multiprocessing.Pool and returns
    a list of tuples (chunk_text, page_number, pdf_id).

    Args:
        pdf_info_list (List[Tuple[str, int]]): List of (path_to_pdf, pdf_id).
        chunk_size (int): Maximum chunk size for the splitter.
        chunk_overlap (int): Overlap between chunks for the splitter.
        num_processes (int, optional): Number of processes to use.
                                       Defaults to os.cpu_count().

    Returns:
        List[Tuple[str, int, int]]: Combined list of (chunk, page_number, pdf_id)
                                     from all processed PDFs.
    """
    if not num_processes:
        num_processes = os.cpu_count()
    # Using {num_processes} processes to read PDFs...
    print(f"Using {num_processes} processes for PDF reading...")

    # Prepare the arguments for each call to process_single_pdf
    # Each item in the list will be a tuple of arguments for one worker execution
    process_args = [
        (file_path, pdf_id, chunk_size, chunk_overlap)
        for file_path, pdf_id in pdf_info_list
    ]

    all_results = []
    results_list_of_lists = [] # Initialize outside try block
    # Use Pool as a context manager to ensure proper cleanup (close/join)
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            # pool.starmap applies the worker function to each tuple of arguments in process_args
            # It waits for all processes to finish and returns a list of results
            # Each result will be the list of chunks returned by process_single_pdf
            results_list_of_lists = pool.starmap(process_single_pdf, process_args)

            # Flatten the list of lists [[chunks_pdf1], [chunks_pdf2], ...] into a single list
            all_results = list(chain.from_iterable(results_list_of_lists))

        except Exception as pool_error:
            print(f"An error occurred during parallel processing: {pool_error}")
            # Decide how to handle pool-level errors (rare if workers handle their own errors well)
            # We return whatever might have been collected or empty.
            if 'all_results' not in locals(): # Ensure all_results exists
                 all_results = []


    total_pdfs = len(pdf_info_list)
    # Count how many PDFs didn't cause a fatal error (returned a list, even if empty)
    successful_pdfs = sum(1 for res_list in results_list_of_lists if isinstance(res_list, list))
    # Parallel processing completed...
    print(f"Parallel processing completed. Attempted {total_pdfs} PDFs, {successful_pdfs} processed (partially or fully).")
    # Generated {len(all_results)} chunks in total.
    print(f"Generated {len(all_results)} chunks in total.")
    return all_results

def update_chunks_in_db(chunks):
    """
    Inserts chunks into the database.

    For each inserted chunk, retrieves the auto-generated ID (chunk_id) and builds a list 
    of tuples in the format (chunk_text, page_num, chunk_id) to ensure consistency between 
    the database and the FAISS index.

    Args:
        chunks (List[Tuple[str, int, int]]): List of (chunk_text, page_num, pdf_id).

    Returns:
        List[Tuple[str, int, int]]: List of (chunk_text, page_num, chunk_id) with the auto-generated ID.
    """
    connection, cursor = db.connect_db('./01_data/project_database.db')
    stored_chunks = []  # List to store (chunk_text, page_num, chunk_id)
    for chunk_text, page_num, pdf_id in chunks:
        # Insert into DB using the pdf_id and page; the chunk_text itself is stored in FAISS.
        cursor.execute("INSERT INTO chunks (pdf_id, page) VALUES (?, ?)", (pdf_id, page_num))
        # Retrieve the auto-generated ID for this chunk
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
    
    Args:
        chunks (List[Tuple[str, int, int]]): List of (chunk_text, page_number, chunk_id)
        vector_store (FAISS): FAISS vector store with embedding model
    
    Returns:
        bool: True if chunks were inserted successfully.
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
    
    Args:
        chunk_ids (List[int]): List of chunk IDs to delete
        faiss_index (FAISS): The vector store instance
    """
    ids_to_delete = [str(chunk_id) for chunk_id in chunk_ids]
    faiss_index.delete(ids_to_delete)

    print("Chunks deleted from FAISS index.")
    return True

def update_faiss(vector_store):
    """
    Updates the FAISS index with the latest PDF chunks, ensuring consistency 
    between the database and the FAISS index.

    The operation flow is as follows:
      1. Determine new PDFs to process and obsolete chunks to delete.
      2. Delete obsolete chunks first from both FAISS and the database to avoid inconsistencies.
      3. Read new PDFs and process chunks.
      4. Insert new chunks into the database (retrieving the auto-generated IDs).
      5. Insert new chunks into the FAISS index using these new IDs.
      6. Save the FAISS index to disk to confirm the changes.

    Args:
        vector_store (FAISS): Instance of the FAISS vector store.

    Returns:
        bool: True if the update was completed successfully.
    """
    print('Determining new PDFs to process...')
    pdfs_to_process = determine_new_pdfs()

    print('Determining chunks to delete...')
    chunks_to_delete = determine_chunks_to_delete()

    # Step 1: Delete obsolete chunks from FAISS and the database
    if chunks_to_delete:
        print("Deleting chunks from FAISS index...")
        delete_faiss(chunks_to_delete, vector_store)
        print("Deleting chunks from the database...")
        delete_chunks_from_db(chunks_to_delete)

    # Step 2: Process new PDFs and obtain new chunks
    print('Reading new PDFs...')
    new_chunks = read_pdfs_parallel(pdfs_to_process)
    print(f"New chunks obtained: {len(new_chunks)}")

    # Step 3: Insert new chunks into the database and retrieve their auto-generated IDs
    print('Inserting new chunks into the database and retrieving their IDs...')
    stored_chunks = update_chunks_in_db(new_chunks)
    print(f"Stored chunks obtained: {len(stored_chunks)}")

    # Step 4: Insert the new chunks into the FAISS index using the generated IDs
    print('Inserting new chunks into the FAISS index...')
    if len(stored_chunks) != 0:
        insert_faiss(stored_chunks, vector_store)

    # Step 5: Save the FAISS index to disk
    commit_faiss(vector_store)
    print("FAISS index updated consistently.")
    return True

def obtain_context(query, vector_store):
    """
    Obtains context from the FAISS index based on the query.
    
    Args:
        query (str): The query string to search for.
        vector_store: The FAISS vector store to search in.
    """

    results = vector_store.similarity_search_with_score(query, k=3)
    return results

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
    
    # Update the FAISS index
    print("Updating FAISS index...")
    update_faiss(vector_store)