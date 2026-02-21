import sys
sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
import os
import tempfile
import pymupdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="RAG básico 🔍", layout="wide")
st.title("RAG sobre cualquier PDF que compartas")

# --- DISCLAIMER DE SEGURIDAD Y FUNCIONALIDAD ---
with st.expander("⚠️ Avisos", expanded=True):
    st.warning("""
    **🛡️ Privacidad de Datos:** Estás utilizando una API pública. **NO subas documentos con información confidencial**, datos personales, financieros o secretos comerciales. El proveedor de la API podría procesar y retener estos datos según sus políticas de uso.
    
    **🧠 Memoria Conversacional:** Este prototipo **no tiene memoria**. El asistente responde a cada pregunta de forma aislada. Si haces una pregunta de seguimiento, el modelo no recordará de qué estabais hablando en la pregunta anterior.
    """)

# --- BARRA LATERAL ---
st.sidebar.header("Configuración ⚙️")

# 1. Input para la API Key (Requisito del usuario)
api_key = st.sidebar.text_input("Introduce tu Groq API Key 🗝️:", type="password")

# 2. Configuración de los LLM
id_model = st.sidebar.selectbox(
    "Modelo:", 
    ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"],
    index=0)
temperature = st.sidebar.slider("Temperatura (recomendado valores más bajos):", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

# 3. Subida de archivo
uploaded_file = st.sidebar.file_uploader("Sube un archivo PDF", type="pdf")


# --- LÓGICA DE PROCESAMIENTO ---
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Feedback por paso
        with st.status("🧠 Procesando documento...", expanded=True) as status:
            st.write("📄 Leyendo PDF...")
            loader = PyMuPDFLoader(tmp_file_path)
            docs = loader.load()
            
            st.write("✂️ Dividiendo texto en fragmentos (chunks)...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            
            st.write("🧮 Generando base de datos vectorial (Embeddings)...")
            embeddings = get_embeddings_model()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            status.update(label="✅ Documento indexado y listo", state="complete", expanded=False)
            return vectorstore
            
    finally:
        os.remove(tmp_file_path)

# --- DETENCIÓN SI FALTAN REQUISITOS ---
if not api_key:
    st.info("👋 Por favor, introduce tu API Key de Groq en la barra lateral para comenzar.")
    st.stop()

if not uploaded_file:
    st.info("📂 Por favor, sube un documento PDF para analizar.")
    st.stop()

# --- INICIALIZACIÓN DEL SISTEMA RAG ---
try:
    # Inicializar LLM
    llm = ChatGroq(
        api_key=api_key,
        model=id_model,
        temperature=temperature,
        max_retries=2)

    if "vectorstore" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        st.session_state.vectorstore = process_pdf(uploaded_file)
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = [] 

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    # --- INTERFAZ DE CHAT ---
    st.divider()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! He leído tu documento. ¿Qué quieres saber sobre él?"}]

    # Mostrar mensajes previos
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario en la parte inferior
    if user_question := st.chat_input("Pregunta algo sobre el documento..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Prompt agnóstico y estricto
        system_prompt = """You are an expert data extractor and assistant. 
        Your ONLY job is to answer the user's question based strictly on the provided context.
        
        RULES:
        1. If the answer is not contained in the context, say exactly: "Lo siento, pero no encuentro esa información en el documento." Do not guess nor be complacent.
        2. Do not use robotic phrases like "Based on the text..." or "According to the context...". Just give the direct answer.
        3. Be concise, clear, and format your response beautifully using markdown (bolding, bullet points) if necessary.
        
        CONTEXT:
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")])

        chain_rag = (
            {"context": retriever, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser())

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Buscando en el documento..."):
                retrieved_docs = retriever.invoke(user_question) 
                response = chain_rag.invoke(user_question)
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()
                
                st.markdown(response)
                
                with st.expander("🔍 Ver fuentes del documento"):
                    for i, doc in enumerate(retrieved_docs):
                        st.caption(f"**Fragmento {i+1}** (Pág. {doc.metadata.get('page', 'N/A')}):")
                        st.write(f"_{doc.page_content}_")
                        st.divider()

        # Guardar respuesta en el historial visual
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Ocurrió un error: {e}")