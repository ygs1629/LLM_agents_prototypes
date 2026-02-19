from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

# --- 1. Herramientas ---
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

# --- 2. Prompt ---
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

# --- Función auxiliar (DEFINIDA FUERA para evitar errores) ---
def ensure_string_output(agent_result: dict) -> dict:
    output = agent_result.get('output')
    if not isinstance(output, str):
        agent_result['output'] = str(output)
    return agent_result

# --- 3. Inicialización del Agente ---
def get_agent_executor(api_key: str):
    # Validamos que la key no tenga espacios extraños
    clean_key = api_key.strip()
    
    chat = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash', 
        temperature=0.3,
        google_api_key=clean_key,
        max_output_tokens=400
    )
    
    agent = create_tool_calling_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Devolvemos el ejecutable + el filtro de texto
    return agent_executor | RunnableLambda(ensure_string_output)

# --- 4. Gestión de Memoria ---
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 5. Función Principal (Process Query) ---
def process_query(user_input, session_id, api_key):
    # 1. Obtenemos el cerebro
    try:
        agent_exec = get_agent_executor(api_key)
        if agent_exec is None:
            return "Error crítico: El agente no se pudo inicializar (NoneType)."
    except Exception as e:
        return f"Error de configuración del agente: {str(e)}"

    # 2. Le añadimos la memoria
    agent_with_history = RunnableWithMessageHistory(
        agent_exec,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # 3. Ejecutamos
    try:
        response = agent_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response['output']
        
    except Exception as e:
        return f"Ocurrió un error al procesar tu solicitud: {str(e)}"