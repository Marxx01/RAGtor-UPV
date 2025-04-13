import os
import multiprocessing
import sqlite3
# Añadir Optional para el tipo del índice base
from typing import List, Tuple, Dict, Any, Optional
from itertools import chain
import PyPDF2
# Mover import faiss DENTRO del bloque main según solicitado
# import faiss
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import database_sql as db # Asumimos que este módulo existe y funciona
import time # Añadido para el ejemplo de main

# =============================================================================
# Configuration / Configuración
# =============================================================================
BASE_DATA_DIR = './01_data'
DB_FILENAME = 'project_database.db'
FAISS_INDEX_SUBDIR = 'project_faiss'
DB_PATH = os.path.join(BASE_DATA_DIR, DB_FILENAME)
FAISS_INDEX_DIR = os.path.join(BASE_DATA_DIR, FAISS_INDEX_SUBDIR)

DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 150
DEFAULT_NUM_PROCESSES: int = os.cpu_count() # Usar todos los cores por defecto

EMBEDDING_MODEL_NAME: str = "sentence-transformers/LaBSE"
# Nota: FAISS_METRIC ya no es necesaria aquí si el índice se crea siempre en main
# FAISS_METRIC = faiss.METRIC_L2

# =============================================================================
# FAISS Functions / Funciones FAISS
# =============================================================================
# Modificar connect_faiss para aceptar un índice base opcional
def connect_faiss(embedding_model,
                  index: Optional[Any] = None, # Aceptar el índice pre-creado
                  faiss_index_dir=FAISS_INDEX_DIR):
    """
    Connects to or creates a FAISS vector store.
    Accepts a pre-created base index for new store initialization.
    Se conecta a un almacén vectorial FAISS existente o crea uno nuevo.
    Acepta un índice base pre-creado para inicializar un almacén nuevo.

    Args:
        embedding_model: The loaded embedding model instance.
        index (Optional[faiss.Index]): A pre-created FAISS index object (e.g., IndexFlatL2).
                                        Used only if creating a new store.
        faiss_index_dir (str): Directory path for the FAISS index.

    Returns:
        FAISS: The connected or newly created FAISS vector store instance, or None on fatal error.
    """
    vector_store = None
    try:
        print(f"Attempting to load FAISS index from: {faiss_index_dir}")
        vector_store = FAISS.load_local(
            folder_path=faiss_index_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully.")
    except RuntimeError as e:
         print(f"RuntimeError loading FAISS index (may not exist or be incompatible): {e}")
    except FileNotFoundError:
         print(f"FileNotFoundError: FAISS index directory not found at {faiss_index_dir}. Will create a new one.")
    except Exception as e:
        print(f"An unexpected error occurred loading FAISS index: {e}. Will create a new one.")

    if vector_store is None:
        print("Creating a new FAISS index.")
        # Si no se pudo cargar, crear uno nuevo USANDO el índice base pasado desde main
        if index is None:
            # Esto es un fallback por si se llama sin índice desde otro sitio,
            # pero según la restricción de main, 'index' debería estar presente.
            print("ERROR: Base FAISS index was not provided for new store creation.")
            # Podríamos intentar crearlo aquí, pero respetando la restricción de main, fallamos.
            return None # No se puede crear sin el índice base según la estructura de main
            # --- Código alternativo si se permitiera crear aquí ---
            # try:
            #     print("WARNING: Base index not provided, attempting to create one.")
            #     import faiss # Importar aquí si es necesario
            #     embedding_dim = len(embedding_model.embed_query("test query"))
            #     index = faiss.IndexFlatL2(embedding_dim) # O IndexFlatIP
            #     print(f"Fallback: Created base FAISS index with dimension {embedding_dim}.")
            # except Exception as fallback_e:
            #      print(f"FATAL: Could not create fallback base index: {fallback_e}")
            #      return None
            # --- Fin código alternativo ---

        try:
             # Crear la instancia de LangChain FAISS usando el índice base proporcionado
             vector_store = FAISS(
                 embedding_function=embedding_model,
                 index=index, # Usa el índice pasado como argumento
                 docstore=InMemoryDocstore(),
                 index_to_docstore_id={}
             )
             print("New FAISS vector store created in memory using provided base index.")
        except Exception as create_error:
             print(f"FATAL: Could not create new FAISS vector store instance: {create_error}")
             return None # Devuelve None si la creación falla

    return vector_store

# (commit_faiss, determine_new_pdfs, determine_chunks_to_delete sin cambios)
def commit_faiss(vector_store, faiss_index_dir=FAISS_INDEX_DIR):
    try:
        vector_store.save_local(folder_path=faiss_index_dir)
        print(f"FAISS index successfully saved to: {faiss_index_dir}")
        return True
    except Exception as e:
        print(f"Error saving FAISS index to {faiss_index_dir}: {e}")
        return False

def determine_new_pdfs(db_path=DB_PATH):
    pdfs = []
    connection, cursor = None, None
    try:
        connection, cursor = db.connect_db(db_path)
        cursor.execute("""
            SELECT pdfs.path, pdfs.id
            FROM pdfs
            WHERE pdfs.in_use = TRUE
            AND NOT EXISTS (SELECT 1 FROM chunks WHERE chunks.pdf_id = pdfs.id);
        """)
        pdfs = cursor.fetchall()
    except Exception as e:
        print(f"Error determining new PDFs from database: {e}")
    finally:
        if connection:
            db.disconnect_db(connection)
    return pdfs

def determine_chunks_to_delete(db_path=DB_PATH):
    chunk_ids = []
    connection, cursor = None, None
    try:
        connection, cursor = db.connect_db(db_path)
        cursor.execute("""
            SELECT chunks.id
            FROM chunks
            JOIN pdfs ON chunks.pdf_id = pdfs.id
            WHERE pdfs.in_use = FALSE AND pdfs.checked = TRUE;
        """)
        chunk_ids = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error determining chunks to delete from database: {e}")
    finally:
        if connection:
            db.disconnect_db(connection)
    return chunk_ids

# (process_single_pdf sin cambios)
def process_single_pdf(file_path: str, pdf_id: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
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
                        page_text = page_text.replace('\xa0', ' ').strip()
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

# (read_pdfs_parallel usando constantes por defecto)
def read_pdfs_parallel(pdf_info_list: List[Tuple[str, int]],
                       chunk_size: int = DEFAULT_CHUNK_SIZE,
                       chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                       num_processes: int = DEFAULT_NUM_PROCESSES) -> List[Tuple[str, int, int]]:
    print(f"Using {num_processes} processes for PDF reading with chunk size {chunk_size} and overlap {chunk_overlap}...")
    process_args = [
        (file_path, pdf_id, chunk_size, chunk_overlap)
        for file_path, pdf_id in pdf_info_list
    ]
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

# (update_chunks_in_db_efficient, delete_chunks_from_db_efficient, insert_faiss, delete_faiss sin cambios)
def update_chunks_in_db_efficient(new_chunks_data: List[Tuple[str, int, int]], db_path=DB_PATH) -> List[Tuple[str, int, int]]:
    if not new_chunks_data: return []
    chunks_with_db_ids = []
    connection, cursor = None, None
    inserted_row_count = 0
    try:
        connection, cursor = db.connect_db(db_path)
        data_to_insert = new_chunks_data
        cursor.executemany("INSERT INTO chunks (chunk, pdf_id, page) VALUES (?, ?, ?)", data_to_insert)
        inserted_row_count = cursor.rowcount
        if inserted_row_count == len(data_to_insert) and inserted_row_count > 0:
            last_id = cursor.lastrowid
            first_id = last_id - inserted_row_count + 1
            generated_ids = list(range(first_id, last_id + 1))
            for i, original_chunk_data in enumerate(new_chunks_data):
                chunk_text, page_num, _ = original_chunk_data
                chunks_with_db_ids.append((chunk_text, page_num, generated_ids[i]))
            print(f"Successfully inserted {inserted_row_count} chunks using executemany (IDs retrieved via lastrowid).")
        elif inserted_row_count > 0:
             print(f"Warning: executemany inserted {inserted_row_count} rows, but expected {len(data_to_insert)}. Cannot reliably determine IDs via lastrowid.")
             return []
        else:
            print("No rows were inserted.")
            return []
        db.commit_db(connection)
    except sqlite3.Error as e:
        print(f"SQLite error during efficient chunk update: {e}")
        if connection: connection.rollback()
        return []
    except Exception as e:
        print(f"Unexpected error during efficient chunk update: {e}")
        if connection: connection.rollback()
        return []
    finally:
        if connection: db.disconnect_db(connection)
    return chunks_with_db_ids

def delete_chunks_from_db_efficient(chunk_ids: List[int], db_path=DB_PATH) -> bool:
    if not chunk_ids: return True
    connection, cursor = None, None
    success = False
    try:
        connection, cursor = db.connect_db(db_path)
        ids_to_delete = [(chunk_id,) for chunk_id in chunk_ids]
        cursor.executemany("DELETE FROM chunks WHERE id = ?", ids_to_delete)
        deleted_count = cursor.rowcount
        print(f"Attempted deletion of {len(chunk_ids)} chunk IDs using executemany. Affected rows: {deleted_count}.")
        db.commit_db(connection)
        success = True
    except sqlite3.Error as e:
        print(f"SQLite error during efficient chunk deletion: {e}")
        if connection: connection.rollback()
    except Exception as e:
        print(f"Unexpected error during efficient chunk deletion: {e}")
        if connection: connection.rollback()
    finally:
        if connection: db.disconnect_db(connection)
    return success

def insert_faiss(chunks_with_db_ids: List[Tuple[str, int, int]], vector_store: FAISS):
    if not chunks_with_db_ids: return True
    try:
        documents = [
            Document(page_content=ct, metadata={"page": pn, "db_chunk_id": db_id})
            for ct, pn, db_id in chunks_with_db_ids
        ]
        ids_for_faiss = [str(db_id) for _, _, db_id in chunks_with_db_ids]
        vector_store.add_documents(documents=documents, ids=ids_for_faiss)
        print(f"{len(documents)} chunks inserted into FAISS index using DB IDs.")
        return True
    except Exception as e:
        print(f"Error inserting chunks into FAISS index: {e}")
        return False

def delete_faiss(chunk_ids_to_delete: List[int], vector_store: FAISS) -> bool:
    if not chunk_ids_to_delete: return True
    try:
        ids_to_delete_str = [str(chunk_id) for chunk_id in chunk_ids_to_delete]
        result = vector_store.delete(ids=ids_to_delete_str)
        if result: print(f"Successfully deleted {len(ids_to_delete_str)} chunk IDs from FAISS index.")
        else: print(f"Attempted to delete {len(ids_to_delete_str)} chunk IDs from FAISS, but none were found in the index.")
        return True
    except NotImplementedError:
         print("Error: The underlying FAISS index does not support deletion.")
         return False
    except Exception as e:
        print(f"Error deleting chunks from FAISS index: {e}")
        return False

# (update_faiss_pipeline sin cambios en lógica interna, solo usa las funciones eficientes)
def update_faiss_pipeline(vector_store, db_path=DB_PATH, faiss_index_dir=FAISS_INDEX_DIR):
    print("\n--- Starting FAISS Update Pipeline ---")
    something_changed = False

    print("Determining chunks to delete...")
    chunks_to_delete_ids = determine_chunks_to_delete(db_path)
    if chunks_to_delete_ids:
        print(f"Found {len(chunks_to_delete_ids)} chunk IDs marked for deletion.")
        something_changed = True
    else:
        print("No chunks marked for deletion.")

    print("Determining new PDFs to process...")
    pdfs_to_process = determine_new_pdfs(db_path)
    if pdfs_to_process:
        print(f"Found {len(pdfs_to_process)} new PDFs to process.")
        something_changed = True
    else:
        print("No new PDFs found to process.")

    new_chunks_data = []
    if pdfs_to_process:
        print("Reading and chunking new PDFs...")
        new_chunks_data = read_pdfs_parallel(pdfs_to_process)

    new_chunks_with_db_ids = []
    if new_chunks_data:
        print("Inserting new chunks into the database...")
        # Usa la función eficiente para insertar y obtener IDs
        new_chunks_with_db_ids = update_chunks_in_db_efficient(new_chunks_data, db_path)
        if not new_chunks_with_db_ids and new_chunks_data: # Comprueba si la inserción falló
             print("ERROR: Failed to insert new chunks into DB or retrieve their IDs. Aborting FAISS insertion.")
             return
        elif new_chunks_with_db_ids:
            print(f"Successfully obtained {len(new_chunks_with_db_ids)} DB IDs for new chunks.")

    if something_changed:
        if new_chunks_with_db_ids:
            print("Inserting new vectors into FAISS index...")
            insert_success = insert_faiss(new_chunks_with_db_ids, vector_store)
            if not insert_success: print("Warning: Insertion into FAISS failed.")

        if chunks_to_delete_ids:
            print("Deleting obsolete vectors from FAISS index...")
            delete_faiss_success = delete_faiss(chunks_to_delete_ids, vector_store)
            if not delete_faiss_success: print("Warning: Deletion from FAISS failed.")

            # Solo borrar de la BD si el borrado de FAISS no dio error (o si decides ignorar el error de FAISS)
            if delete_faiss_success: # O podrías borrar de la BD independientemente
                 print("Deleting obsolete chunks from the database...")
                 delete_db_success = delete_chunks_from_db_efficient(chunks_to_delete_ids, db_path)
                 if not delete_db_success: print("Warning: Deletion from database failed.")
            else:
                 print("Skipping database deletion because FAISS deletion failed.")

        print("Committing changes to FAISS index on disk...")
        commit_faiss(vector_store, faiss_index_dir)
    else:
        print("No changes detected in PDFs or chunks. FAISS index remains unchanged.")
    print("--- FAISS Update Pipeline Finished ---")

# (obtain_context sin cambios)
def obtain_context(query, vector_store, k=3):
    print(f"\nSearching for context related to: '{query}'")
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        context = []
        if results:
            print("Found relevant context:")
            for i, (doc, score) in enumerate(results):
                page_info = f" (Page: {doc.metadata.get('page', 'N/A')})" if doc.metadata else ""
                db_id_info = f" (DB_ID: {doc.metadata.get('db_chunk_id', 'N/A')})" if doc.metadata else ""
                context_line = f"{i+1}. [Score: {score:.4f}]{page_info}{db_id_info}\n   '{doc.page_content[:200]}...'"
                print(context_line)
                context.append(context_line)
        else:
            print("No relevant context found in the index.")
        return context
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []

# =============================================================================
# Main Execution / Ejecución Principal (Estructura Original Restaurada)
# =============================================================================

if __name__ == "__main__":
    print("--- Initializing RAG Pipeline ---")
    # Cargar el modelo de embedding
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # Calcular dimensión después de cargar
        embedding_dim = len(embedding_model.embed_query("hello world")) # Test query
        print(f"Embedding model loaded. Dimension: {embedding_dim}")
    except Exception as e:
        print(f"FATAL: Failed to load embedding model: {e}")
        exit() # Salir si el modelo no carga

    # --- Bloque específico solicitado por el usuario ---
    try:
        # Importar faiss aquí
        import faiss
        print(f"FAISS library imported. Creating base index...")
        # Crear índice base aquí (usando L2 por defecto o ajusta según necesites)
        index = faiss.IndexFlatL2(embedding_dim)
        print(f"Base FAISS index created (type: IndexFlatL2, dim: {embedding_dim}).")
    except ImportError:
        print("FATAL: faiss library not found. Please install it (`pip install faiss-cpu` or `pip install faiss-gpu`).")
        exit()
    except Exception as e:
        print(f"FATAL: Failed to create base FAISS index: {e}")
        exit()
    # --- Fin del bloque específico ---

    # Conectar/Crear el almacén FAISS, pasando el índice base pre-creado
    print("Connecting to FAISS vector store...")
    # Pasa el 'index' creado localmente a connect_faiss
    vector_store = connect_faiss(embedding_model=embedding_model, index=index)

    # Si la conexión/creación fue exitosa, proceder
    if vector_store:
        # Eliminar el commit_faiss inmediato que estaba aquí. Se hace al final de update_faiss_pipeline.
        # commit_faiss(vector_store) # <-- ELIMINADO/COMENTADO

        # Actualizar el índice FAISS y la base de datos
        print("\n--- Starting Update Process ---")
        update_faiss_pipeline(vector_store)

        # Ejemplo de cómo obtener contexto después de actualizar
        print("\n--- Example Usage: Obtaining Context ---")
        # Cambia esto por una consulta relevante para tus datos
        test_query = "información sobre el proyecto A"
        context_results = obtain_context(test_query, vector_store)
        # Descomenta si quieres ver el resultado formateado
        # print("\nFormatted Context Retrieved:")
        # for line in context_results:
        #     print(line)
    else:
        print("FATAL: Could not connect to or create FAISS vector store. Exiting.")

    print("\n--- RAG Pipeline Finished ---")