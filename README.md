# RAGtor-UPV: AI Chatbot for UPV Regulations

## Project Overview

**RAGtor-UPV** is an AI-powered chatbot designed to assist students, staff, and faculty at the Universitat Politècnica de València (UPV) by answering questions about university regulations, procedures, and academic policies. The system leverages Retrieval-Augmented Generation (RAG) with a custom vector database (FAISS) and a large language model (PoliGPT) to provide accurate, context-aware, and up-to-date responses based on official university documents.

---

## Features

- **Conversational Chatbot**: Friendly, interactive chat interface built with Streamlit.
- **Contextual Retrieval**: Uses semantic search (FAISS + sentence-transformers) to find the most relevant regulatory content.
- **RAG Pipeline**: Combines retrieved context with a generative LLM (PoliGPT) for precise, grounded answers.
- **PDF & Database Management**: Automatically scans, indexes, and updates regulatory PDFs in a SQLite database.
- **Parallel Processing**: Efficient PDF chunking and embedding using multiprocessing.
- **Session Memory**: Maintains chat history for context continuity.
- **Customizable & Extensible**: Modular codebase for easy adaptation to other institutions or document sets.

---

## How It Works

1. **PDF Ingestion & Database Update**
   - Place all regulatory PDFs in the `01_data/pdf_actuales/` directory.
   - Run the database update script to scan, register, and track new or modified PDFs in `project_database.db`.

2. **Chunking & Embedding**
   - PDFs are split into overlapping text chunks.
   - Each chunk is embedded using a multilingual transformer model (`sentence-transformers/LaBSE`).

3. **FAISS Vector Store**
   - All embeddings are stored in a FAISS index (`01_data/project_faiss/`) for fast semantic search.

4. **Chatbot Interaction**
   - Users interact via a Streamlit web interface.
   - When a question is asked, the system retrieves the top-k most relevant chunks from FAISS.
   - The retrieved context and the user’s question are sent to PoliGPT (hosted LLM API).
   - PoliGPT generates a concise, context-grounded answer, citing the relevant regulation.

5. **Session & Logging**
   - All interactions are logged for future improvements and analytics.

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/your-org/ragtor-upv.git
cd ragtor-upv
```

### 2. Prepare the Environment

- Install Python 3.9+ (recommended: use a virtual environment).
- Install dependencies:

```sh
python -m venv rag_fix
source rag_fix/Scripts/activate  # On Windows: rag_fix\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Data

- Place all regulatory PDFs in `01_data/pdf_actuales/`.
- (Optional) Place additional CSVs or data in `01_data/DB_CSV/`.

### 4. Initialize the Database

```sh
python 01_main/database_sql.py
```

This will create and populate `01_data/project_database.db` with metadata about your PDFs.

### 5. Build or Update the FAISS Index

```sh
python 01_main/database_faiss_murta.py
```

This script will:
- Chunk and embed all active PDFs.
- Store embeddings in the FAISS index (`01_data/project_faiss/`).

### 6. Launch the Chatbot Interface

```sh
streamlit run 01_main/interfaz_00_conexion_prueba.py
```

Open the provided local URL in your browser to start chatting with RAGtor-UPV.

---

## Project Structure

```
01_data/
    pdf_actuales/         # Regulatory PDFs
    project_database.db   # SQLite DB with PDF and chunk metadata
    project_faiss/        # FAISS vector index
    DB_CSV/               # Additional data (optional)
01_main/
    database_sql.py       # DB creation and update scripts
    database_faiss_murta.py # FAISS index management
    poli_gpt.py           # RAG pipeline and PoliGPT API client
    interfaz_00_conexion_prueba.py # Streamlit chatbot UI
static/                   # Static files for web interface
rag_fix/                  # Python virtual environment
```

---

## Typical Workflow

1. **Add or update PDFs** in `01_data/pdf_actuales/`.
2. **Update the database**:  
   `python 01_main/database_sql.py`
3. **Update the FAISS index**:  
   `python 01_main/database_faiss_murta.py`
4. **Launch the chatbot**:  
   `streamlit run 01_main/interfaz_00_conexion_prueba.py`
5. **Ask questions** about UPV regulations and receive grounded, referenced answers.

---

## Troubleshooting

- **No vectors in FAISS**: Make sure you have run both the database and FAISS update scripts after adding PDFs.
- **Timeouts with PoliGPT**: Check your internet connection and the availability of the PoliGPT API.
- **Database errors**: Ensure the database is created and up-to-date before running the FAISS script.
- **PDF parsing errors**: Some PDFs may be encrypted or malformed; check logs for details.

---

## Contributors

- [Marxx01](https://github.com/Marxx01) - Marc Hurtado Beneyto
- [Hervaas8](https://github.com/Hervaas8) - Alejandro Hervás Castillo
- [Vimapo23](https://github.com/Vimapo23) - Víctor Mánez Poveda
- [QuicoCaballer](https://github.com/QuicoCaballer) - Francisco Caballer Gutierrez
- [ogarmar](https://github.com/ogarmar) - Óscar García Martínez

---

## License

RAGtor-UPV, Universitat Politècnica de València

---

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [sentence-transformers](https://www.sbert.net/)
- [OpenAI](https://openai.com/)