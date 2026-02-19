import sys
# Configuración para evitar errores de codificación en Windows
sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="RAG Customer Support 🤖", layout="wide")
st.title("🤖 RAG sobre el pdf que subas...")

# --- BARRA LATERAL (Configuración) ---
st.sidebar.header("Configuración")

# 1. Input para la API Key (Requisito del usuario)
api_key = st.sidebar.text_input("Introduce tu Groq API Key:", type="password")

# 2. Configuración del Modelo (Igual que en el notebook)
id_model = st.sidebar.selectbox(
    "Modelo:", 
    ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"],
    index=0
)
temperature = st.sidebar.slider("Temperatura:", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

# 3. Subida de archivo (Reemplaza a files.upload())
uploaded_file = st.sidebar.file_uploader("Sube tu manual (PDF)", type="pdf")


# --- LÓGICA DE PROCESAMIENTO (Igual que 'Indexing Steps') ---

@st.cache_resource
def get_embeddings_model():
    """Carga y cachea el modelo de embeddings para no recargarlo en cada interacción."""
    # Usamos el mismo modelo que en tu notebook
    embedding_model_name = "BAAI/bge-large-en-v1.5"
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

def process_pdf(uploaded_file):
    """Procesa el PDF: Lo guarda, lo carga con PyMuPDF, lo divide y crea el índice FAISS."""
    
    # PyMuPDFLoader necesita un archivo en disco, así que creamos uno temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # 1. Cargar documento (PyMuPDFLoader)
        loader = PyMuPDFLoader(tmp_file_path)
        docs = loader.load()
        
        # 2. Split (RecursiveCharacterTextSplitter)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # 3. Embeddings & Vector Store (FAISS)
        embeddings = get_embeddings_model()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
        
    finally:
        # Limpieza del archivo temporal
        os.remove(tmp_file_path)

# --- INTERFAZ PRINCIPAL ---

if not api_key:
    st.info("👋 Por favor, introduce tu API Key de Groq en la barra lateral para comenzar.")
    st.stop()

if not uploaded_file:
    st.info("📂 Por favor, sube un documento PDF (ej. manual de usuario) para analizar.")
    st.stop()

# --- INICIALIZACIÓN DEL SISTEMA RAG ---

try:
    # Inicializar LLM
    llm = ChatGroq(
        api_key=api_key,
        model=id_model,
        temperature=temperature,
        max_retries=2
    )

    # Procesar documento solo si no está en cache o ha cambiado
    if "vectorstore" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Procesando documento... Esto puede tardar unos segundos (generando embeddings)..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.session_state.file_name = uploaded_file.name
        st.success("✅ Documento indexado correctamente.")

    # Configurar Retriever (Indexing Step 4)
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # --- CHAT INTERFACE ---
    
    # Campo de pregunta del usuario
    user_question = st.text_input("Pregunta sobre el documento:", placeholder="How do I change my password?")

    if user_question:
        # Definir Prompt (Improvements to the prompt section)
        system_prompt = """You are a helpful virtual assistant answering general questions about a company's services.
        Use the following bits of retrieved context to answer the question.
        If you don't know the answer, just say you don't know. Keep your answer concise. \n\n"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Question: {input}\n\n Context: {context}"),
            ]
        )

        # Crear Cadena (Chain Creation)
        chain_rag = (
            {"context": retriever, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # Ejecutar y mostrar
        with st.spinner("Generando respuesta..."):
            response = chain_rag.invoke(user_question)
            
            st.markdown("### Respuesta:")
            # Limpieza básica por si el modelo devuelve tags de pensamiento (como DeepSeek/R1)
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            
            st.markdown(response)
            
            # Opcional: Mostrar fuentes usadas (Debug)
            with st.expander("Ver contexto recuperado (Debug)"):
                docs = retriever.invoke(user_question)
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content[:200]}...")

except Exception as e:
    st.error(f"Ocurrió un error: {e}")