import streamlit as st
from datetime import datetime

st.title("🤖 ChatBot Mejorado")

# Inicializa el estado de sesión para almacenar mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos en el historial
for message in st.session_state.messages:
    role = message["role"]
    content = message["message"]
    timestamp = message.get("timestamp", "")
    
    with st.chat_message(role):
        st.markdown(f"**{role.capitalize()}** ({timestamp}):  \n{content}")

# Entrada del usuario
prompt = st.chat_input("Escribe tu mensaje...")

if prompt:
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(f"**Tú** ({timestamp}):  \n{prompt}")

    # Guardar mensaje en sesión
    st.session_state.messages.append(
        {"role": "user", "message": prompt, "timestamp": timestamp}
    )

    # Respuesta automática del bot
    bot_response = "Hola, soy un bot 🤖. ¿En qué puedo ayudarte?"

    with st.chat_message("assistant"):
        st.markdown(f"**Bot** ({timestamp}):  \n{bot_response}")

    # Guardar respuesta en sesión
    st.session_state.messages.append(
        {"role": "assistant", "message": bot_response, "timestamp": timestamp}
    )