import streamlit as st
from datetime import datetime
import time

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
            <li>    - Contextual conversation</li>
            <li>    - Chat historial</li>
            <li>    - Intelligent responses</li>
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
                        🇪🇸 <span style='text-decoration: underline'>Vimapo23</span> - Víctor Mánez Poveda
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
        st.session_state.messages = []
        st.rerun()

# Título principal
st.markdown('<h1 class="header">🦖 UPV Documentation Chatbot</h1>', unsafe_allow_html=True)

# Estilos CSS personalizados con tema verde
st.markdown("""
    <style>
        :root {
            --dark-green: #1a3e2f;
            --medium-green: #2d5a45;
            --light-green: #e8f5e9;
            --accent-green: #4caf50;
        }
        
        body {
            background-color: var(--light-green);
        }
        
        .stApp {
            background: linear-gradient(135deg, var(--light-green) 0%, #c8e6c9 100%);
        }
        
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

# Inicializa el estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = "Hi! I'm RAGtor-UPV, grrr. Feel free to ask me anything! 🦖📚"
    st.session_state.messages.append({
        "role": "assistant", 
        "message": welcome_msg,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# Mostrar historial de mensajes
for message in st.session_state.messages:
    role = message["role"]
    content = message["message"]
    timestamp = message.get("timestamp", "")
    
    with st.chat_message(role):
        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <strong style='color: {'white' if role == 'user' else '#1a3e2f'};'>
                    {'Tú' if role == 'user' else '🦖 RAGtor-UPV'}
                </strong>
                <small style='color: {'#c8e6c9' if role == 'user' else '#666'};'>{timestamp}</small>
            </div>
            <div style='color: {'white' if role == 'user' else '#333'};'>{content}</div>
        """, unsafe_allow_html=True)

# Función de prueba para el LLM
def generate_response(user_input, chat_history):
    # Esta es la función de prueba - reemplazar con la conexión real al LLM
    print("Prueba de llamada al LLM")  # Esto se verá en la consola
    return f"🦖 Respuesta a: '{user_input}'. [Este es un mensaje de prueba del LLM local]"

# Entrada del usuario con diseño verde
prompt_container = st.container()
with prompt_container:
    prompt = st.chat_input("typing...", key="user_input")

if prompt:
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <strong style='color: white;'>You</strong>
                <small style='color: #c8e6c9;'>{timestamp}</small>
            </div>
            <div style='color: white;'>{prompt}</div>
        """, unsafe_allow_html=True)
    
    st.session_state.messages.append({
        "role": "user", 
        "message": prompt, 
        "timestamp": timestamp
    })
    
    # Respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Obtener historial de chat
            chat_history = [
                {"role": m["role"], "content": m["message"]} 
                for m in st.session_state.messages
            ]
            
            # Generar respuesta usando el LLM
            bot_response = generate_response(prompt, chat_history)
            
            # Mostrar respuesta
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <strong style='color: #1a3e2f;'>🦖 RAGtor-UPV</strong>
                    <small style='color: #666;'>{timestamp}</small>
                </div>
                <div style='color: #333;'>{bot_response}</div>
            """, unsafe_allow_html=True)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "message": bot_response, 
        "timestamp": timestamp
    })

# Título principal con diseño verde
st.markdown("""
    <div style='background-color: #1a3e2f; padding: 20px; border-radius: 15px; margin-bottom: 30px;'>
        <p style='color: #c8e6c9; text-align: center; margin: 5px 0 0;'>Our chatbot works wonderfully and will help you with most of your questions. However, please remember that as with any AI, some answers might not be completely accurate. We trust that its assistance will be valuable!</p>
    </div>
""", unsafe_allow_html=True)