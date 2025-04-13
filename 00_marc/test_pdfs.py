import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import multiprocessing
import os
from typing import List, Tuple
from itertools import chain # Needed to flatten the list of results
import time # Added for timing example

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

# --- Example Usage ---
if __name__ == "__main__":
    # Make sure this block is inside if __name__ == "__main__":
    # This is necessary for multiprocessing on some operating systems (Windows)

    # Create some example PDFs (or use real paths)
    # Note: To test, you would need actual PDF files.
    # Example: pdf_files = [("path/to/doc1.pdf", 1), ("path/to/doc2.pdf", 2), ...]
    pdf_files_to_process = [
        # Replace with paths to real PDFs and their IDs
        ("01_data/pdf_actuales/1968000000000100000001.pdf", 1),
        ("01_data/pdf_actuales/1985000000000900000017.pdf", 2),
        # ... add more files
    ]

    # Create dummy files for the example if they don't exist (for demonstration only)
    # You might need to install PyPDF2 (`pip install pypdf2`) if you don't have it
    try:
        from PyPDF2 import PdfWriter
        pdf_writer_available = True
    except ImportError:
        pdf_writer_available = False
        print("PyPDF2 not found, cannot create dummy PDF files. Skipping dummy file creation.")

    if pdf_writer_available:
        for fname, _ in pdf_files_to_process:
             if "non_existent" not in fname and "encrypted" not in fname and not os.path.exists(fname):
                  try:
                      # Create a very basic empty PDF with PyPDF2 (if PyPDF2 is installed)
                      writer = PdfWriter()
                      writer.add_blank_page(width=612, height=792) # Standard US Letter size
                      with open(fname, "wb") as f:
                          writer.write(f)
                      # Dummy file created: {fname}
                      print(f"Dummy file created: {fname}")
                  except Exception as e:
                      # Could not create dummy file {fname}: {e}
                      print(f"Could not create dummy file {fname}: {e}")


    # Starting parallel PDF processing...
    print("Starting parallel PDF processing...")
    start_time = time.time() # You need to import time

    # Call the new parallel function
    final_chunks = read_pdfs_parallel(pdf_files_to_process, chunk_size=600, chunk_overlap=100)

    end_time = time.time()
    # Total processing time: {end_time - start_time:.2f} seconds
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    # Optional: Print some chunks to verify
    # print("\nSome generated chunks:")
    # for i, chunk_info in enumerate(final_chunks[:5]):
    #     # Chunk {i+1} (PDF ID: {chunk_info[2]}, Page: {chunk_info[1]}): '{chunk_info[0][:50]}...'
    #     print(f"Chunk {i+1} (PDF ID: {chunk_info[2]}, Page: {chunk_info[1]}): '{chunk_info[0][:50]}...'")