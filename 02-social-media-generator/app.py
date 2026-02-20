import sys
sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Publicaciones para redes sociales", page_icon="✍️", layout="wide")

# --- ESTADO DE LA SESIÓN ---
if "generated_content" not in st.session_state:
    st.session_state.generated_content = ""

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input("🔑 Groq API Key:", type="password")
    st.markdown("---")
    st.caption("Modelo: llama-3.3-70b-versatile")

if not api_key_input:
    st.warning("👋 Para comenzar, introduce tu API Key de Groq en la barra lateral.")
    st.stop()

# --- INICIALIZACIÓN DEL LLM ---
try:
    llm = ChatGroq(api_key=api_key_input,
        model="llama-3.3-70b-versatile",
        temperature=1.0,max_tokens=None,
        timeout=None,max_retries=2,)
    
except Exception as e:
    st.error(f"Error al conectar con Groq. Verifica tu API Key. Detalles: {e}")
    st.stop()

# --- FUNCIÓN DE GENERACIÓN ---
def generate_content(llm_model, params):
    system_prompt = """You are a world-class Social Media Manager and elite Copywriter.
    Your goal is to write HIGH-CONVERTING, PLATFORM-NATIVE content that stops the scroll. DRAW AS MUCH ATENTTION AS POSSIBLE.
    
    CRITICAL RULES (DO NOT IGNORE):
    1. NO WIKIPEDIA SYNDROME: DO NOT define concepts (e.g., never start with "Mental health is..."). Instead, speak directly to the user's pain points, emotions, or daily struggles.
    2. USE HOOKS: The first sentence must be a powerful hook, a controversial statement, or an engaging question.
    3. PLATFORM FORMATTING:
        - Instagram: Aesthetic spacing. Conversational. Natural emojis (don't overdo it).
        - Twitter / X: Maximum 280 characters. Short, punchy, bold statement. No cringe emojis.
        - LinkedIn: Professional but storytelling-based. Use short paragraphs.
    4. NO AI ROBOT TALK: Sound human, empathetic and engaging.
    5. Output ONLY the raw post content. No intro/outro remarks from you.
    """
    
    human_prompt = """
    Write a social media post using these parameters:
    - Topic: {topic}
    - Platform: {platform}
    - Tone: {tone}
    - Target Audience: {audience}
    - Length constraint: {length}
    - Call to Action: {cta}
    - Hashtags: {hashtags}
    - SEO Keywords to blend naturally: {keywords}
    """
    
    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),])
    
    chain = template | llm_model | StrOutputParser()
    return chain.invoke(params)

# --- CABECERA ---
st.title("✍️ Generador de contenido")
st.markdown("Crea borradores para tus redes sociales en segundos.")
st.divider()

# --- LAYOUT A DOS COLUMNAS ---
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📝 Configuración")
    
    with st.container(border=True):
        topic = st.text_input("🎯 Tema:", placeholder="Ej: burnout en el trabajo, tips para dormir...")
        
        c1, c2 = st.columns(2)
        with c1:
            platform = st.selectbox("📱 Plataforma:", ['Instagram', 'Twitter / X', 'LinkedIn', 'Facebook', 'Blog'])
            audience = st.selectbox("👥 Audiencia:", ['Público General', 'Jóvenes / Gen Z', 'Profesionales', 'Familias'])
        with c2:
            tone = st.selectbox("🎭 Tono:", ['Informativo', 'Inspirador', 'Humorístico / Informal', 'Directo / Agresivo'])
            length = st.selectbox("📏 Longitud:", ['Corto (+ directo)', 'Medio (Storytelling)', 'Largo (+ informativo)'])

        with st.expander("⚙️ Opciones SEO & Conversión"):
            keywords = st.text_input("🔑 Palabras clave (SEO):", placeholder="Ej: salud mental, terapia, autocuidado...")
            cta = st.checkbox("Incluir Call to Action (CTA)", value=True)
            hashtags = st.checkbox("Incluir Hashtags", value=True)

    # Botón dentro de la columna izquierda
    generar = st.button("🚀 Generar", type="primary", use_container_width=True)

with col_output:
    st.subheader("💡 La propuesta")
    
    if generar:
        if not topic:
            st.warning("⚠️ Escribe un tema para empezar.")
        else:
            params = {
                "topic": topic,
                "platform": platform,
                "tone": tone,
                "audience": audience,
                "length": length,
                "cta": "Include a strong Call to Action asking for comments or clicks." if cta else "Do NOT include a Call to Action.",
                "hashtags": "Include 3-5 highly relevant hashtags." if hashtags else "Do NOT include any hashtags.",
                "keywords": keywords if keywords else "None"
            }
            
            with st.spinner("🧠 Redactando la publicación..."):
                try:
                    # Llamamos al LLM y guardamos en estado
                    st.session_state.generated_content = generate_content(llm, params)
                except Exception as e:
                    st.error(f"Error de generación: {e}")
    
    # Mostramos el resultado siempre en esta columna
    if st.session_state.generated_content:
        with st.container(border=True):
            st.markdown(st.session_state.generated_content)
        
        # Desplegable oculto con el botón nativo de copiar
        with st.expander("📋 Haz clic aquí para copiar el texto generado 👉"):
            st.code(st.session_state.generated_content, language="markdown")