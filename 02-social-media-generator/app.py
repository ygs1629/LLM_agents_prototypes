import sys
sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuración de página (debe ser lo primero)
st.set_page_config(page_title="Content Generator 🤖", page_icon="🤖")

# --- BARRA LATERAL PARA API KEY ---
st.sidebar.header("Configuración")
api_key_input = st.sidebar.text_input("Introduce tu Groq API Key:", type="password")

if not api_key_input:
    st.info("👋 Para usar esta app, introduce tu API Key de Groq en la barra lateral.")
    st.stop()  # Detiene la ejecución hasta que haya clave

# --- CONFIGURACIÓN DEL LLM CON LA CLAVE DEL USUARIO ---
id_model = "llama-3.3-70b-versatile"

try:
    llm = ChatGroq(
        api_key=api_key_input, # Aquí usamos la clave que metió el usuario
        model=id_model,
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
except Exception as e:
    st.error(f"Error al conectar con Groq. Verifica tu API Key. Detalles: {e}")
    st.stop()

## Función de generación
def llm_generate(llm, prompt):
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a digital marketing expert specialized in SEO and persuasive copywriting."),
        ("human", "{prompt}"),
    ])
    
    chain = template | llm | StrOutputParser()
    res = chain.invoke({"prompt": prompt})
    return res

# --- INTERFAZ PRINCIPAL ---
st.title("Content generator")

topic = st.text_input("Topic:", placeholder="e.g., nutrition, mental health, routine check-ups...")
platform = st.selectbox("Platform:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'E-mail'])
tone = st.selectbox("Message tone:", ['Normal', 'Informative', 'Inspiring', 'Urgent', 'Informal'])
length = st.selectbox("Text length:", ['Short', 'Medium', 'Long'])
audience = st.selectbox("Target audience:", ['All', 'Young adults', 'Families', 'Seniors', 'Teenagers'])

col1, col2 = st.columns(2)
with col1:
    cta = st.checkbox("Include CTA")
with col2:
    hashtags = st.checkbox("Return Hashtags")

keywords = st.text_area("Keywords (SEO):", placeholder="Example: wellness, preventive healthcare...")

if st.button("Content generator"):
    if not topic:
        st.warning("Por favor, escribe un tema (Topic) primero.")
    else:
        prompt = f"""
        Write an SEO-optimized text on the topic '{topic}'.
        Return only the final text in your response and don't put it inside quotes.
        - Platform where it will be published: {platform}.
        - Tone: {tone}.
        - Target audience: {audience}.
        - Length: {length}.
        - {"Include a clear Call to Action." if cta else "Do not include a Call to Action."}
        - {"Include relevant hashtags at the end of the text." if hashtags else "Do not include hashtags."}
        {"- Keywords to include (for SEO): " + keywords if keywords else ""}
        """
        
        with st.spinner("Generando contenido..."):
            try:
                res = llm_generate(llm, prompt)
                st.markdown(res)
            except Exception as e:
                st.error(f"Ocurrió un error al generar: {e}")