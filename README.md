# RAGtor-UPV: Chatbot for UPV Student Regulations

## Project Overview

RAGtor-UPV is an AI-powered chatbot designed to assist students at the Universitat Politècnica de València (UPV) by answering questions about university regulations. This project utilizes Retrieval-Augmented Generation (RAG) and the Raft algorithm (RAG + Fine-Tuning) to provide accurate and context-aware responses.

## Features

- Interactive chatbot interface for students to ask questions
- Utilizes UPV's regulatory documents as the knowledge base
- Implements RAG (Retrieval-Augmented Generation) for improved answer accuracy
- Uses the Raft algorithm for consistent and reliable information retrieval, using Fine-tuning to improve the answers
- Maintains an up-to-date database of regulatory PDFs

## How It Works

1. **PDF Management**: 
   - The system scans a directory for regulatory PDF documents.
   - New or updated PDFs are detected and registered in a SQLite database.
   - Changes to the document repository are logged for tracking purposes.

2. **RAG Implementation**:
   - When a student asks a question, the system retrieves relevant information from the PDF database.
   - This retrieved context is then used to generate an accurate response.

3. **Raft Algorithm**:
   - Ensures consistency in information retrieval across potential multiple instances of the chatbot.

4. **User Interface**:
   - Students interact with the chatbot through a Streamlit-based web interface.
   - The chat history is maintained within each session for context continuity.

## Data Used

- PDF documents containing UPV regulations and normative information
- SQLite database to manage and track PDF documents
- Chat logs and user interactions (for improving the system)

## Usage

To use RAGtor-UPV:

1. Ensure all regulatory PDFs are in the designated directory.
2. Run the database update script to index new or modified documents.
3. Launch the Streamlit interface: <streamlit run test_interfaz.py>
4. Students can then access the chatbot through a web browser and start asking questions about UPV regulations.

## Components

1. **PDF Database Management** (`database.py`):
- Handles scanning, registering, and updating PDF documents in the database.

2. **Chatbot Interface** (`test_interfaz.py`):
- Provides the Streamlit-based user interface for student interactions.

3. **RAG and Raft Implementation** (not visible in provided snippets):
- Manages the retrieval and generation of responses based on the regulatory documents.

## Future Improvements

- Enhance the accuracy of information retrieval and response generation
- Implement multi-language support for international students
- Develop a mobile application for easier access
- Integrate with UPV's student portal for seamless authentication

## Contributors

This project is developed and maintained by:

- [Marxx01](https://github.com/Marxx01) -Marc Hurtado Beneyto
- [Hervaas8](https://github.com/Hervaas8) -Alejandro Hervás Castillo
- [Vimapo23](https://github.com/Vimapo23) -Víctor Mánez Poveda
- [QuicoCaballer](https://github.com/QuicoCaballer) -Francisco Caballer Gutierrez
- [ogarmar](https://github.com/ogarmar) -Óscar García Martínez

## License

[RAGTor-UPV, Universitat Politécnica de Valencia]