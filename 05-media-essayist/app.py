import streamlit as st
import json
import uuid
import ast
from langchain_core.messages import HumanMessage, AIMessage
from agent import build_workflow

# --- Configuración de la Página ---
st.set_page_config(page_title="AI Philosophy Professor", page_icon="🦉", layout="wide")

st.title("🦉 El Catedrático AI: Análisis Filosófico de la Actualidad")
st.markdown("Propón un suceso o concepto. Nuestros agentes lo procesarán a través de **Ontología, Ética y Epistemología**.Esto es solo una aproximación, no se tiene tomar como una tesis categórica y contundente.")

# --- Inicialización de Estado ---
if "sources" not in st.session_state:
    st.session_state.sources = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 Credenciales")
    google_api_key = st.text_input("Google API Key (Gemini)", type="password")
    tavily_api_key = st.text_input("Tavily API Key (Búsqueda)", type="password")
    st.info("Pulsa 'Enter' tras introducir las claves.")
    
    st.markdown("---")
    st.header("📚 Bibliografía Empírica")
    sources_placeholder = st.empty()

    if st.session_state.sources:
        with sources_placeholder.container():
            for source in st.session_state.sources:
                st.markdown(f"🔹 [{source.get('url', '#')[:35]}...]({source.get('url', '#')})")
    else:
        sources_placeholder.info("Si el tema requiere noticias, aparecerán aquí.")

if not (google_api_key and tavily_api_key):
    st.warning("⚠️ Introduce tus claves API para despertar al Catedrático.")
    st.stop()

# --- Instanciar Grafo ---
app_workflow = build_workflow(google_api_key, tavily_api_key)

# --- Renderizar Chat ---
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- Input de Usuario ---
if user_input := st.chat_input("Plantea tu tesis o evento..."):
    st.session_state.sources = []
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    inputs = {"messages": [HumanMessage(content=user_input)]}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    trace_logs = []
    final_response = None

    with st.status("🏛️ El Claustro de Agentes está deliberando...", expanded=True) as status:
        for event in app_workflow.stream(inputs, config=config, stream_mode="values"):
            message = event['messages'][-1]
            
            if isinstance(message, AIMessage):
                sender = message.additional_kwargs.get("sender", "System")
                content = " ".join([str(i) for i in message.content]) if isinstance(message.content, list) else str(message.content)
                
                # Feedback de estado y logs internos    
                if sender == "Investigador":
                    status.write("📰 **Investigador:** Analizando premisas o buscando datos...")
                    trace_logs.append(f"### 📰 Investigador\n{content}")
                    
                    if "Premisa filosófica" in content and len(st.session_state.sources) == 0:
                        with sources_placeholder.container():
                            st.info("🧠 Análisis puramente teórico. No requiere fuentes empíricas de actualidad.")
                elif sender == "Ontología":
                    status.write("🏛️ **Ontología:** Estructurando la esencia del problema...")
                    trace_logs.append(f"### 🏛️ Ontología\n{content}")
                elif sender == "Ética":
                    status.write("⚖️ **Ética:** Evaluando tensiones morales...")
                    trace_logs.append(f"### ⚖️ Ética\n{content}")
                elif sender == "Epistemología":
                    status.write("🖋️ **Epistemología:** Sintetizando tesis final...")
                    trace_logs.append(f"### 🖋️ Epistemología\n{content}")
                    final_response = content 
                elif sender == "Tutor Casual":
                    status.write("💬 **Tutor:** Respondiendo de forma conversacional...")
                    trace_logs.append(f"### 💬 Tutor Casual\n{content}")
                    final_response = content
                
            elif message.type == "tool" or message.type == "function":
                status.write("🌐 **Buscador:** Fuentes empíricas recuperadas...")
                
                # Guardamos un log truncado para no saturar la trazabilidad
                content_str = str(message.content)
                trace_logs.append(f"### 🌐 Tool Output\n{content_str[:300]}...\n*(Datos completos procesados en memoria)*")
                
                # 1. PARSEO TODOTERRENO
                tool_data = None
                if isinstance(message.content, (list, dict)):
                    tool_data = message.content
                else:
                    try:
                        tool_data = json.loads(message.content)
                    except Exception:
                        try:
                            tool_data = ast.literal_eval(message.content)
                        except Exception:
                            pass
                
                # 2. EXTRACCIÓN SEGURA DE FUENTES
                sources_list = []
                if isinstance(tool_data, dict):
                    # Si es el formato nuevo {"results": [...]}
                    sources_list = tool_data.get("results", [])
                elif isinstance(tool_data, list):
                    # Si es el formato antiguo [...]
                    sources_list = tool_data
                
                # 3. FILTRADO (Solo nos quedamos con los que tienen URL válida)
                valid_sources = [s for s in sources_list if isinstance(s, dict) and 'url' in s]
                
                # 4. RENDERIZADO EN EL SIDEBAR
                if valid_sources:
                    st.session_state.sources = valid_sources
                    with sources_placeholder.container():
                        st.markdown("### ✅ Fuentes consultadas:")
                        for source in valid_sources:
                            # Sacamos el título, o usamos la URL si no hay título
                            title = source.get('title', source.get('url', 'Enlace externo'))
                            url = source.get('url', '#')
                            st.markdown(f"🔹 [{title[:45]}...]({url})")

        status.update(label="¡Deliberación completada!", state="complete", expanded=False)

    # --- Renderizado de Respuesta ---
    if final_response:
        clean_text = final_response.replace("TITLE:", "# ").replace("BODY:", "\n\n")
        # Limpieza de fallos residuales de IA
        if "NOTE:" in clean_text:
            clean_text = clean_text.split("NOTE:")[0]
        
        # Eliminamos cualquier cabecera extraña que la IA intente colar
        clean_text = clean_text.replace("---Análisis Ético", "").strip()

        with st.chat_message("assistant"):
            st.markdown(clean_text) 
            st.session_state.messages.append(AIMessage(content=clean_text))

    with st.expander("🔍 Ver Proceso Cognitivo (Trazabilidad)"):
        for log in trace_logs:
            st.markdown(log)
            st.markdown("---")