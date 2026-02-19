import streamlit as st
import os
import functools
import json
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuración de la Página ---
st.set_page_config(page_title="AI News Writer Agent", page_icon="📝", layout="wide")

st.title("🤖 Generador de Artículos con Agentes AI")
st.markdown("Este sistema utiliza un flujo de **LangGraph** con agentes de Búsqueda, Esquematización y Redacción.")

# --- Inicialización de Estado para Fuentes ---
if "sources" not in st.session_state:
    st.session_state.sources = []

# --- Sidebar: Configuración de APIs y Fuentes ---
with st.sidebar:
    st.header("🔑 Configuración")
    
    google_api_key = st.text_input(
        "Google API Key (Gemini)", 
        type="password", 
        help="Introduce tu clave de Google AI Studio"
    )
    
    tavily_api_key = st.text_input(
        "Tavily API Key", 
        type="password",
        help = "Introduce tu clave del buscador tavily"
    )

    st.info("Asegúrate de pulsar 'Enter' tras pegar las claves.")
    
    st.markdown("---")
    st.header("🌐 Fuentes Consultadas")
    sources_container = st.container()
    
    if st.session_state.sources:
        with sources_container:
            for source in st.session_state.sources:
                st.markdown(f"🔹 [{source['url'][:30]}...]({source['url']})")
    else:
        sources_container.info("Aquí aparecerán los enlaces consultados.")

# --- Lógica del Grafo (Solo se define si hay Keys) ---

if google_api_key and tavily_api_key:
    # Configurar entorno
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    # 1. Definir Estado (Igual al original)
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # 2. Definir Herramientas (Igual al original)
    tools = [TavilySearchResults(max_results=5)]

    # 3. Helper para crear Agentes (Igual al original)
    def create_agent(llm, tools, system_message: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        if tools:
            return prompt | llm.bind_tools(tools)
        else:
            return prompt | llm

    # 4. Definir Prompts (CORREGIDOS PARA EVITAR ALUCINACIONES DE PYTHON)
    
    search_template = """Your job is to act as a research assistant.
    
    **LANGUAGE INSTRUCTION:**
    You must detect the language of the user's request (e.g., Spanish, English, French).
    Any explanation, output, or thought process you generate must be IN THE SAME LANGUAGE as the user's request.
    
    **INSTRUCTIONS:**
    1. Search the web for related news relevant to the user's topic using the provided tool.
    2. Once you have the search results, synthesize the key information into a clear summary.
    3. Do NOT output Python code or function calls as text. Use the tool directly if needed.
    
    **SAFETY & LOGIC CHECK:**
    Analyze the user input:
    1. If the user asks a specific question (e.g., "Who is Jeffrey Epstein?", "¿Qué pasó con las elecciones?", "Latest stats on AI"), this is VALID. PROCEED to search.
    2. If the user provides a topic that is just a SINGLE ISOLATED KEYWORD without context (e.g., just "Trump", "Messi", "War"), this is AMBIGUOUS.
    
    ONLY IF the topic is a single isolated keyword with no question or context, output exactly: 
    "STOP: [Explanation in user's language why the topic is too broad]."
    
    OTHERWISE, forward the summarized findings to the outliner node.
    """

    outliner_template = """Your job is to generate an outline for an article based on the provided search results/summary.
    
    **LANGUAGE INSTRUCTION:**
    Detect the language of the user's original request.
    Generate the outline IN THE SAME LANGUAGE as the user's request.
    
    **INSTRUCTION:**
    Review the previous messages for the search summary. Create a structured outline."""

    writer_template = """Your job is to write a full article based on the provided outline.

    **FORMATTING INSTRUCTION:**
    You must strictly use this format structure (keep the labels TITLE and BODY in English for parsing, but write the content in the target language):

    TITLE: <Insert Title Here in User's Language>
    BODY: <Insert Body Here in User's Language>

    **LANGUAGE INSTRUCTION:**
    Detect the language of the user's original request (Spanish, English, etc.).
    The content of the title and the body MUST BE written in that same language.

    NOTE: Do not copy the outline directly. Write a cohesive narrative."""

    # 5. Inicializar Modelo (CON SAFETY SETTINGS DESACTIVADOS)
    # Esto evita que el modelo devuelva bloqueos silenciosos (respuestas vacías)
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        temperature=1,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    # 6. Crear Agentes
    search_agent = create_agent(llm, tools, search_template)
    outliner_agent = create_agent(llm, [], outliner_template)
    writer_agent = create_agent(llm, [], writer_template)

    # 7. Definir Nodos (Añadimos metadatos 'sender' para la UI)
    def agent_node(state, agent, name):
        try:
            result = agent.invoke(state)
            # Aseguramos que el sender se pegue correctamente
            result.additional_kwargs["sender"] = name
            return {'messages': [result]}
        except Exception as e:
            # Captura de errores para que no se rompa el loop silenciosamente
            return {'messages': [AIMessage(content=f"Error en {name}: {str(e)}", additional_kwargs={"sender": name, "error": True})]}

    search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
    outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")
    tool_node = ToolNode(tools)

    # 8. Definir Aristas (Edges) - Lógica Modificada para STOP
    def should_search(state) -> Literal["tools", "outliner", END]:
        messages = state['messages']
        last_message = messages[-1]
        
        # Si hay tool calls y no hemos parado, vamos a herramientas
        if last_message.tool_calls and "STOP:" not in last_message.content:
            return "tools"
        
        # Si el modelo ha dicho STOP, terminamos el flujo aquí
        if "STOP:" in last_message.content:
            return END
            
        # Si no, seguimos al outliner (flujo normal)
        return "outliner"

    # 9. Construir Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("search")
    
    # Mapeo explícito de las salidas condicionales
    workflow.add_conditional_edges(
        "search", 
        should_search,
        {
            "tools": "tools",
            "outliner": "outliner",
            END: END
        }
    )
    
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    app = workflow.compile()

else:
    st.warning("⚠️ Por favor introduce tu Google API Key en el menú lateral para comenzar.")
    st.stop()

# --- Interfaz de Chat (Frontend) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial previo
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input del Usuario
user_input = st.chat_input("Escribe el tema del artículo...")

if user_input:
    # Reset de fuentes y UI
    st.session_state.sources = []
    sources_container.empty()
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    trace_logs = []
    final_response = None
    last_ai_message_content = None

    with st.status("🧠 Los agentes están trabajando...", expanded=True) as status:
        
        for event in app.stream(inputs, stream_mode="values"):
            message = event['messages'][-1]
            
            if isinstance(message, AIMessage):
                sender = message.additional_kwargs.get("sender", "System")
                content = message.content
                last_ai_message_content = content # Guardar por si acaso
                
                if sender == "Search Agent":
                    status.write("🔍 **Search Agent:** Analizando y buscando...")
                    trace_logs.append(f"### 🔍 Search Agent Output\n{content}")
                    
                    if "STOP:" in content:
                        final_response = content
                        status.write("🛑 **Search Agent:** Pregunta ambigua detectada.")
                    
                elif sender == "Outliner Agent":
                    status.write("📝 **Outliner Agent:** Estructurando contenido...")
                    trace_logs.append(f"### 📝 Outliner Agent Output\n{content}")
                    
                elif sender == "Writer Agent":
                    status.write("✍️ **Writer Agent:** Redactando artículo final...")
                    trace_logs.append(f"### ✍️ Writer Agent Output (Final)\n{content}")
                    final_response = content 
            
            elif message.type == "tool":
                 status.write("🛠️ **Tool:** Obteniendo datos externos...")
                 trace_logs.append(f"### 🛠️ Tool Output\n{message.content}")
                 
                 # Lógica para extraer URLs y ponerlas en el sidebar
                 try:
                     tool_data = json.loads(message.content)
                     if isinstance(tool_data, list) and len(tool_data) > 0:
                         st.session_state.sources = tool_data
                         with sources_container:
                             st.write("✅ Fuentes encontradas:")
                             for source in tool_data:
                                 url = source.get('url', '#')
                                 st.markdown(f"🔹 [{url[:35]}...]({url})")
                 except Exception:
                     pass

        # Fallback: Si terminamos y no hay final_response (y no fue STOP), usamos el último mensaje IA
        if not final_response and last_ai_message_content and "STOP:" not in last_ai_message_content:
             final_response = last_ai_message_content

        status.update(label="¡Proceso completado!", state="complete", expanded=False)

    # 3. Renderizar Output Final
    if final_response:
        
        # Limpieza de lista si viene envuelto
        if isinstance(final_response, list):
            clean_text = final_response[-1]
        else:
            clean_text = final_response
            
        # Caso 1: Se detuvo por ambigüedad
        if "STOP:" in clean_text:
            clean_text = clean_text.replace("STOP:", "🛑 **Aviso:**")
        
        # Caso 2: Es un artículo normal -> Limpiamos etiquetas TITLE/BODY
        else:
            clean_text = clean_text.replace("TITLE:", "# ").replace("BODY:", "")
            if "NOTE:" in clean_text:
                clean_text = clean_text.split("NOTE:")[0]

        with st.chat_message("assistant"):
            st.markdown(clean_text.strip()) 
            st.session_state.messages.append(AIMessage(content=clean_text))

    with st.expander("🕵️ Ver Trazabilidad"):
        for log in trace_logs:
            st.markdown(log)
            st.markdown("---")