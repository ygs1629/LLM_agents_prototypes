from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
import yaml
import subprocess
import tempfile

# --- 1. Herramientas MLOps ---
search = DuckDuckGoSearchRun()

@tool
def checkov_terraform_scanner(tf_content: str) -> str:
    """Ejecuta Checkov SAST scanner sobre código Terraform."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tf', delete=False) as f:
        f.write(tf_content)
        temp_path = f.name
        
    result = subprocess.run(['checkov', '-f', temp_path, '--output', 'json'], capture_output=True, text=True)
    
    return result.stdout 

@tool
def dockerfile_analyzer(dockerfile_content: str) -> str:
    """Analiza el contenido de un Dockerfile."""
    feedback = []
    content_lower = dockerfile_content.lower()
    if "latest" in content_lower: feedback.append("⚠️ Evita tags ':latest'.")
    if "root" in content_lower or "user " not in content_lower: feedback.append("🔒 Usa un usuario no-root.")
    if "apt-get update" in content_lower and "rm -rf" not in content_lower: feedback.append("🧹 Limpia la caché de apt.")
    return "Mejoras sugeridas:\n" + "\n".join(feedback) if feedback else "Dockerfile OK."

@tool
def yaml_syntax_validator(yaml_content: str) -> str:
    """Verifica sintaxis YAML básica."""
    try:
        yaml.safe_load(yaml_content)
        return "El YAML es sintácticamente válido."
    except yaml.YAMLError as exc:
        return f"Error de sintaxis en el YAML: {exc}"

@tool
def k8s_logic_analyzer(yaml_content: str) -> str:
    """Analiza manifiestos de Kubernetes en busca de vulnerabilidades y buenas prácticas."""
    feedback = []
    content = yaml_content.lower()
    if "runasuser: 0" in content: feedback.append("🚨 [K8s] Contenedor ejecutándose como root (runAsUser: 0).")
    if "resources:" not in content: feedback.append("⚠️ [K8s] Faltan resource requests/limits (CPU/Memoria).")
    if "networkpolicy" not in content: feedback.append("⚠️ [K8s] No se detectan NetworkPolicies.")
    return "Hallazgos en K8s:\n" + "\n".join(feedback) if feedback else "K8s OK."

@tool
def terraform_analyzer(tf_content: str) -> str:
    """Analiza código Terraform (HCL) multinube (AWS, GCP, Azure) en busca de brechas de seguridad."""
    feedback = []
    content = tf_content.lower()
    # Redes / AWS
    if "0.0.0.0/0" in content: feedback.append("🚨 [Red] CIDR abierto al público (0.0.0.0/0).")
    if "s3:*" in content or "resource\": \"*\"" in content: feedback.append("🚨 [IAM] Permisos excesivos (wildcards *).")
    # Azure
    if "allow_blob_public_access = true" in content: feedback.append("🚨 [Azure] Acceso público a blobs habilitado.")
    # GCP / Otros
    if "google_project_iam_binding" in content and "roles/owner" in content: feedback.append("🚨 [GCP] Rol de Owner asignado por código.")
    return "Hallazgos en Terraform:\n" + "\n".join(feedback) if feedback else "Terraform parece seguro."

tools = [search, dockerfile_analyzer, yaml_syntax_validator, k8s_logic_analyzer, terraform_analyzer]

# --- 2. Prompt del Especialista  ---
system_instruction = """
Eres un Tutor Experto en MLOps, Cloud y Despliegue a Producción. 
Tu misión es ayudar a Data Scientists e Ingenieros de IA (muy buenos en Python y Jupyter Notebooks, pero novatos en infraestructura) a llevar sus modelos a producción de forma segura y profesional.

REGLAS DE TUTORÍA:
1. ENFOQUE PEDAGÓGICO: Cuando detectes un error. NO te limites a regañar. Explica *por qué* es una mala práctica en el mundo real del Machine Learning y cómo solucionarlo.
2. NO ABRUMES (CERO FALSOS POSITIVOS): Limítate a corregir el código que te pasan basándote en estándares de la industria. Si el código es básico pero correcto para un MVP, señala el acierto y NO sobrecompliques configuraciones si no es conveniente.
3. CONTEXTO MLOPS: Relaciona los problemas de infraestructura con el ciclo de vida del ML (ej. "Sin tags en Terraform perderás el rastro de cuánto cuesta tu infraestructura de inferencia").

FORMATO DE RESPUESTA:
- Usa un tono alentador y profesional.
- Estructura tu feedback en: "✅ Lo que está bien", "⚠️ Áreas de mejora (con explicación)" y "🛠️ Cómo solucionarlo (código)".
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("placeholder", "{history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


def ensure_string_output(agent_result: dict) -> dict:
    output = agent_result.get('output')
    if not isinstance(output, str):
        agent_result['output'] = str(output)
    return agent_result

# --- 3. Inicialización del Agente ---
def get_agent_executor(api_key: str):
    clean_key = api_key.strip()
    chat = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash', 
        temperature=0.2, 
        google_api_key=clean_key,
        max_output_tokens=800
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor | RunnableLambda(ensure_string_output)

# --- 4. Gestión de Memoria ---
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 5. Función Principal ---
def process_query(user_input, session_id, api_key):
    try:
        agent_exec = get_agent_executor(api_key)
        agent_with_history = RunnableWithMessageHistory(
            agent_exec,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        response = agent_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response['output']
    except Exception as e:
        return f"Error al procesar: {str(e)}"