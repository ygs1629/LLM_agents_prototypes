import streamlit as st
import os

# --- 1. CONFIGURACIÓN DE PÁGINA (TIENE QUE SER LO PRIMERO) ---
st.set_page_config(page_title="Profe Python AI", page_icon="🐍")

# --- 2. IMPORTS CON MANEJO DE ERRORES VISUAL ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.tools import tool
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
except ImportError as e:
    st.error(f"❌ Error CRÍTICO de Importación: {e}")
    st.error("Por favor, revisa el archivo requirements.txt")
    st.stop()

# --- 3. DEFINICIÓN DE HERRAMIENTAS Y AGENTE ---

# Herramientas
search = DuckDuckGoSearchRun()

@tool
def python_syntax_checker(code_snippet: str) -> str:
    """Analiza sintaxis básica de Python y devuelve feedback."""
    feedback = []
    if "print " in code_snippet and "(" not in code_snippet:
        feedback.append("Error: En Python 3 'print' necesita paréntesis.")
    if "def" in code_snippet and ":" not in code_snippet:
        feedback.append("Error: Faltan dos puntos ':' tras definir función.")
    if not feedback:
        return "Sintaxis básica correcta."
    return "Errores encontrados: " + " ".join(feedback)

tools = [search, python_syntax_checker]

# Prompt
system_instruction = """
Eres un Tutor de Python conciso.
1. Respuestas CORTAS (máximo 3 párrafos).
2. Ve al grano. Si hay código, explícalo en 2 frases antes de mostrarlo.
3. Si el usuario te da código, analízalo con 'python_syntax_checker'.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("placeholder", "{history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Función para inicializar el agente
def get_agent_chain(api_key):
    if not api_key:
        return None
    
    # Modelo
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=500
    )
    
    # Agente
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# Gestión de Memoria
if "memory_store" not in st.session_state:
    st.session_state.memory_store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.memory_store:
        st.session_state.memory_store[session_id] = ChatMessageHistory()
    return st.session_state.memory_store[session_id]

# --- 4. INTERFAZ GRÁFICA (UI) ---

st.title("🐍 Tu Tutor de Python Personal")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input("🔑 Google API Key", type="password", help="Pega aquí tu clave")
    
    st.divider()
    st.write("🛠️ **Zona de Código**")
    user_code = st.text_area("Pega tu código aquí:", height=150)
    
    if st.button("🗑️ Borrar Chat"):
        st.session_state.messages = []
        st.rerun()

# Inicializar Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Pega tu API Key y empecemos."}]

# Mostrar Mensajes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Lógica principal de ejecución
if user_input := st.chat_input("Escribe tu duda..."):
    
    if not api_key_input:
        st.warning("⚠️ ¡Necesito la API Key en la barra lateral!")
        st.stop()

    # Guardar y mostrar input usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Preparar consulta completa
    full_query = user_input
    if user_code:
        full_query = f"CÓDIGO:\n```python\n{user_code}\n```\nPREGUNTA:\n{user_input}"

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.status("🧠 Pensando...", expanded=True) as status:
            try:
                # 1. Obtener Agente
                agent_exec = get_agent_chain(api_key_input.strip())
                
                # 2. Configurar Memoria
                agent_with_history = RunnableWithMessageHistory(
                    agent_exec,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )
                
                # 3. Ejecutar
                response = agent_with_history.invoke(
                    {"input": full_query},
                    config={"configurable": {"session_id": "sesion_unica"}}
                )
                
                output_text = response["output"]
                status.update(label="¡Hecho!", state="complete", expanded=False)
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Algo falló: {e}")