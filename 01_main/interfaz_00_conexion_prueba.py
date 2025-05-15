import os
import sys
import asyncio
import nest_asyncio
import streamlit as st
from datetime import datetime
import time
import torch
import os
os.environ["PYTORCH_NO_MODULE"] = "1"
# Set environment variables before any other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STATIC_DIRECTORY"] = os.path.dirname(os.path.abspath(__file__))

# Fix PyTorch class path issue
if hasattr(torch.classes, '__dict__'):
    torch.classes.__dict__['_path'] = []

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure event loop
def setup_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Apply nest_asyncio for nested event loops
try:
    setup_event_loop()
    nest_asyncio.apply()
except Exception as e:
    st.error(f"Error setting up event loop: {e}")

# Import PoliGPT after environment setup
try:
    from poli_gpt import PoliGPT
except ImportError as e:
    st.error(f"Error importing PoliGPT: {e}")
    st.stop()

# Initialize PoliGPT client with proper error handling
@st.cache_resource
def init_poligpt():
    try:
        setup_event_loop()
        return PoliGPT()
    except Exception as e:
        st.error(f"Error initializing RAG model: {str(e)}")
        return None

# Configuración de la página
st.set_page_config(
    page_title="🦖 RAGtor-UPV",
    page_icon="🦖 ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .stChatMessage {
            border-radius: 15px;
            padding: 12px 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #f0f8ff;
        }
        .bot-message {
            background-color: #f5f5f5;
        }
        .stTextInput input {
            border-radius: 20px !important;
            padding: 10px 15px !important;
        }
        .header {
            color: #4a4a4a;
            border-bottom: 2px solid #4a4a4a;
            padding-bottom: 10px;
        }
        /* Added style for the context section */
        .context-section {
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px dashed #ccc; /* Optional: adds a separator line */
            font-size: 0.9em;
            color: #555; /* Darker grey for context text */
            white-space: pre-wrap; /* Preserve formatting like newlines and bullet points */
        }
         .context-section strong {
            color: #333; /* Slightly darker color for "Contextos utilizados" */
         }

        /* Ensure lists within messages are formatted */
        .stChatMessage ul {
            padding-left: 20px;
        }
        .stChatMessage li {
            margin-bottom: 5px;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar con información adicional
with st.sidebar:
    st.header("ℹ️ Important Information")
    st.markdown("""
    <div style='padding: 15px; background-color: #2d5a45; border-radius: 15px; margin-top: 20px;'>
        <h4 style='color: white;'>Version: 0.0:</h4>
        <h4 style='color: white;'>Features:</h4>
        <ul style='color: #e8f5e9;'>
            <li>     - Contextual conversation</li>
            <li>     - Chat historial</li>
            <li>     - Intelligent responses</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
    st.divider()
    st.markdown("""
        <div style='margin-top: 20px;'>
            <h4 style='color: white; margin-bottom: 15px;'>Thinked, designed and created by:</h4>
            <div style='display: grid; grid-template-columns: 1fr; gap: 8px;'>
                <div>
                    <a href='https://github.com/Marxx01' target='_blank' style='color: #4caf50; text-decoration: none;'>
                        🦖 <span style='text-decoration: underline'>Marxx01</span> - Marc Hurtado Beneyto
                    </a>
                </div>
                <div>
                    <a href='https://github.com/Hervaas8' target='_blank' style='color: #4caf50; text-decoration: none;'>
                        🥶 <span style='text-decoration: underline'>Hervaas8</span> - Alejandro Hervás Castillo
                    </a>
                </div>
                <div>
                    <a href='https://github.com/Vimapo23' target='_blank' style='color: #4caf50; text-decoration: none;'>
                        🔥 <span style='text-decoration: underline'>Vimapo23</span> - Víctor Mánez Poveda
                    </a>
                </div>
                <div>
                    <a href='https://github.com/QuicoCaballer' target='_blank' style='color: #4caf50; text-decoration: none;'>
                        🦕 <span style='text-decoration: underline'>QuicoCaballer</span> - Francisco Caballer Gutierrez
                    </a>
                </div>
                <div>
                    <a href='https://github.com/ogarmar' target='_blank' style='color: #4caf50; text-decoration: none;'>
                        ⁉️ <span style='text-decoration: underline'>ogarmar</span> - Óscar García Martínez
                    </a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    if st.button("🧹 Clean chat"):
        # Clear both messages and the PoliGPT client
        st.session_state.messages = []
        if "poligpt_client" in st.session_state:
            del st.session_state.poligpt_client
        st.rerun()

# Título principal
st.markdown('<h1 class="header">🦖 UPV Documentation Chatbot</h1>', unsafe_allow_html=True)

# Estilos CSS personalizados con tema verde (kept for consistency, but check for conflicts with the first style block)
# Note: You might want to merge these style blocks or ensure they don't override each other undesirably.
st.markdown("""
    <style>
        :root {
            --dark-green: #1a3e2f;
            --medium-green: #2d5a45;
            --light-green: #e8f5e9;
            --accent-green: #4caf50;
        }

        /* body {
            background-color: var(--light-green);
        } */ /* Removed potential conflict */

        /* .stApp {
            background: linear-gradient(135deg, var(--light-green) 0%, #c8e6c9 100%);
        } */ /* Removed potential conflict */

        .stChatMessage {
            border-radius: 20px !important;
            padding: 15px 20px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            border: none !important;
        }

        .user-message {
            background-color: var(--medium-green) !important;
            color: white !important;
            margin-left: 15% !important;
            border-top-right-radius: 5px !important;
        }

        .bot-message {
            background-color: white !important;
            color: var(--dark-green) !important;
            margin-right: 15% !important;
            border-top-left-radius: 5px !important;
            border: 1px solid var(--accent-green) !important;
        }

        .stTextInput input {
            border-radius: 25px !important;
            padding: 12px 20px !important;
            border: 2px solid var(--accent-green) !important;
        }

        .header {
            color: var(--dark-green) !important;
            border-bottom: 3px solid var(--accent-green) !important;
            padding-bottom: 15px !important;
            margin-bottom: 30px !important;
        }

        .sidebar .sidebar-content {
            background: var(--dark-green) !important;
            color: white !important;
        }

        .stButton button {
            border-radius: 20px !important;
            background-color: var(--accent-green) !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
        }

        .stButton button:hover {
            background-color: var(--medium-green) !important;
        }
    </style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = "Hi! I'm RAGtor-UPV, grrr. Feel free to ask me anything! 🦖📚"
    # Store initial welcome message slightly differently if it doesn't have response/context structure
    st.session_state.messages.append({
        "role": "assistant",
        "message": welcome_msg, # Use 'message' key for simple messages
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# Initialize PoliGPT client only once per session
if "poligpt_client" not in st.session_state:
    with st.spinner("Initializing RAG model..."):
        st.session_state.poligpt_client = init_poligpt()
        if st.session_state.poligpt_client:
            st.success("RAG model initialized successfully!")
        else:
            st.error("Failed to initialize RAG model")
            st.stop()

# Retrieve the PoliGPT client from session state
poligpt_client = st.session_state.poligpt_client


# --- Display Chat History ---
# Modified loop to handle different message structures
for message in st.session_state.messages:
    role = message["role"]
    # content = message["message"] # No longer directly use the 'message' key for bot responses
    timestamp = message.get("timestamp", "")

    # Apply message styles based on role
    message_class = "user-message" if role == "user" else "bot-message"
    header_color = "white" if role == "user" else "#1a3e2f"
    content_color = "white" if role == "user" else "#333"
    time_color = "#c8e6c9" if role == "user" else "#666"

    with st.chat_message(role):
        # Determine the content to display based on message structure
        if role == "user":
            # User messages only have the 'message' key
            display_content = message.get("message", "")
        else:
            # Assistant messages might have 'response_text' and 'contexts_text'
            response_text = message.get("response_text", "")
            contexts_text = message.get("contexts_text", "")

            # If it's an older message or a simple message (like welcome), use the 'message' key
            if not response_text and "message" in message:
                 response_text = message["message"]

            # Combine response and contexts for display
            display_content = response_text
            if contexts_text:
                 # Use the custom CSS class for context formatting
                 display_content += f"\n\n<div class='context-section'><strong>Contextos utilizados:</strong>\n{contexts_text}</div>"


        st.markdown(f"""
            <div class="{message_class}" style="border-radius: 20px; padding: 15px 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <strong style='color: {header_color};'>
                        {'You' if role == 'user' else '🦖 RAGtor-UPV'}
                    </strong>
                    <small style='color: {time_color};'>{timestamp}</small>
                </div>
                <div style='color: {content_color};'>{display_content}</div>
            </div>
        """, unsafe_allow_html=True)


# --- User Input and Response Generation ---

# User input with green design
prompt_container = st.container()
with prompt_container:
    # Ensure the key is unique if needed, but 'user_input' seems fine here
    prompt = st.chat_input("Ask RAGtor-UPV anything...", key="user_input")

if prompt:
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Add and display user message
    st.session_state.messages.append({
        "role": "user",
        "message": prompt, # User messages still use the 'message' key
        "timestamp": timestamp
    })

    # Rerun to show the user message immediately
    st.rerun()

# After rerunning, if the last message was from the user, generate a bot response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                print("[INTERFAZ] Último mensaje del usuario:", st.session_state.messages[-1]["message"])
                bot_result_dict = poligpt_client.query_poligpt(st.session_state.messages[-1]["message"])
                print("[INTERFAZ] Diccionario de resultado recibido:", bot_result_dict)

                bot_response_text = bot_result_dict.get('response', 'Error: No response text received.')
                bot_contexts_text = bot_result_dict.get('contexts', '')
                print("[INTERFAZ] Texto de respuesta:", bot_response_text)
                print("[INTERFAZ] Contextos usados:", bot_contexts_text)

            except Exception as e:
                print("[INTERFAZ] Error al generar la respuesta:", repr(e))
                bot_response_text = f"Sorry, an error occurred while generating the response: {e}"
                bot_contexts_text = ""
                st.error(bot_response_text)

            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant",
                "response_text": bot_response_text,
                "contexts_text": bot_contexts_text,
                "timestamp": timestamp
            })
            st.rerun()


# Título principal con diseño verde (footer section)
st.markdown("""
    <div style='background-color: #1a3e2f; padding: 20px; border-radius: 15px; margin-top: 30px;'>
        <p style='color: #c8e6c9; text-align: center; margin: 0;'>Our chatbot works wonderfully and will help you with most of your questions. However, please remember that as with any AI, some answers might not be completely accurate. We trust that its assistance will be valuable!</p>
    </div>
""", unsafe_allow_html=True)