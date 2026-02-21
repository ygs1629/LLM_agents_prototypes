import streamlit as st
import uuid
import time
import fitz 
from agent import process_query

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLOps & Cloud Architect", page_icon="☁️", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente personal de MLops/Cloud. Pregunta tus dudas e intentaré ayudarte de la mejor forma posible"}]

# --- GENERADOR DE STREAMING VISUAL ---
def stream_text(text):
    """Simula el efecto máquina de escribir para una mejor UX"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)

st.title("☁️ AI MLOps & Cloud Architect")
st.caption("🚀 Diseña, optimiza y despliega tus modelos a producción aplicando y entendiendo las mejores prácticas de la industria.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input("🔑 Google API Key", type="password")
    
    st.divider()
    st.write("🐳 **Analizador de Código**")
    user_code = st.text_area("Pega un YAML o Dockerfile:", height=150)
    
    st.divider()
    st.write("📕 **Contexto Adicional (PDF)**")
    uploaded_pdf = st.file_uploader("Sube guías o documentación", type=["pdf"])
    
    st.divider()
    if st.button("🗑️ Iniciar Nueva Sesión", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "¡Mesa limpia! ¿Qué desplegamos ahora?"}]
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- RENDERIZAR CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- LÓGICA DE INTERACCIÓN ---
if user_input := st.chat_input("Ej: Audita mi código y compáralo con el PDF..."):
    
    if not api_key_input:
        st.warning("⚠️ Introduce tu API Key de Google.")
        st.stop()

    # 1. Indicadores visuales de lectura
    adjuntos = []
    pdf_text = ""
    
    if user_code:
        adjuntos.append("📄 Código (YAML/Dockerfile)")
    
    if uploaded_pdf is not None:
        adjuntos.append(f"📕 PDF ({uploaded_pdf.name})")
        # Extraer texto del PDF al vuelo
        with st.spinner("Leyendo PDF..."):
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            for page in doc:
                pdf_text += page.get_text()

    # Guardar y mostrar el mensaje del usuario con los chips visuales
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        if adjuntos:
            st.caption("📎 **Archivos adjuntos leídos:** " + " | ".join(adjuntos))

    # 2. Construir el prompt oculto
    full_query = user_input
    if user_code:
        full_query += f"\n\n[CÓDIGO ADJUNTO]:\n```\n{user_code}\n```"
    if pdf_text:
        # Limitamos un poco para evitar pasarnos, aunque Gemini soporta 1M tokens
        full_query += f"\n\n[CONTEXTO DEL PDF ADJUNTO]:\n{pdf_text[:100000]}"

    # 3. Llamada al Agente y Streaming
    with st.chat_message("assistant"):
        with st.spinner("🛠️ Analizando, validando y buscando..."):
            try:
                output_text = process_query(full_query, st.session_state.session_id, api_key_input)
            except Exception as e:
                output_text = f"Error crítico: {e}"
        
        # Efecto Streaming
        st.write_stream(stream_text(output_text))
        st.session_state.messages.append({"role": "assistant", "content": output_text})
        
        # 4. Botón de Descarga
        st.download_button(
            label="💾 Descargar esta respuesta (.md)",
            data=output_text,
            file_name="informe_mlops.md",
            mime="text/markdown"
        )