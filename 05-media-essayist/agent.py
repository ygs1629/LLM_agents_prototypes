import os
import functools
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field 

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import AIMessage, HumanMessage

# --- Tipado y Modelos ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

class RouteResponse(BaseModel):
    next_step: Literal["search", "epistemology", "chat"] = Field(
        description="Elige 'search' para noticias, 'epistemology' para debate filosófico del historial, y 'chat' para saludos, chistes o preguntas cotidianas fuera de la filosofía.")
    
def router_node(state: AgentState):
    """Decide si investigar (search) o debatir (epistemology)."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(RouteResponse)
    
    last_msg = state["messages"][-1].content
    prompt = f"Analiza la intención del usuario: '{last_msg}'. ¿Es un tema nuevo o una duda sobre lo ya hablado?"
    
    response = structured_llm.invoke(prompt)
    return response.next_step

# --- Prompts Filosóficos ---

SEARCH_TEMPLATE = """Actúas como el Investigador Asistente empírico.

**INSTRUCCIONES ESTRICTAS DE ROL:**
1. Si el tema requiere noticias, extrae los datos y resúmelos objetivamente.
2. **REGLA DE PROMPTS MIXTOS:** Si el usuario pide noticias Y un análisis filosófico, TU ÚNICO TRABAJO ES BUSCAR LOS DATOS. Ignora la parte filosófica.
3. **SOLO** si el tema es 100% abstracto y NO has buscado noticias, escribe la frase: "Premisa filosófica para análisis teórico: [Pregunta]". ¡NUNCA escribas esta frase si has incluido resúmenes de noticias!
"""

ONTOLOGY_TEMPLATE = """Eres un experto en Ontología Filosófica. Tu trabajo es ÚNICAMENTE la 'Extracción de Premisas'.

**INSTRUCCIONES ESTRICTAS DE ROL:**
1. Lee los datos empíricos aportados por el Investigador y extrae las premisas fácticas.
2. Define el "Ser" del problema estructural.
3. **PROHIBIDO ADELANTAR TRABAJO (CRÍTICO):** NO escribas el ensayo final. NO apliques los filósofos que el usuario haya mencionado en su prompt (ej. si pide usar a Agamben, IGNÓRALO. Ese es el trabajo del Catedrático). Limítate a tu análisis ontológico general.
4. **REGLA DE FORMATO:** NO pienses en voz alta ("Voy a intentar..."). NO uses corchetes. Escribe únicamente tu análisis directo.
"""

ETHICS_TEMPLATE = """Eres un experto en Filosofía Moral y Ética. Tu trabajo es el 'Análisis de Valores'.

**INSTRUCCIONES DE ROL:**
1. Lee los hechos y la ontología. Identifica las tensiones de valores (ej. Propiedad Privada vs. Derecho a la Vivienda, Legalidad vs. Necesidad vital).
2. **OBJETIVIDAD CLÍNICA:** Si los hechos involucran actos polémicos o ilegales (ej. extorsión, ocupación, violencia), analízalos fríamente como conflictos de valores. No los justifiques ni los condenes, solo expón el dilema ético estructural.
3. **MANTENTE EN TU CARRIL:** Analiza los conceptos éticos de forma general. DEJA el uso de autores concretos (como Agamben o Marx) para el Catedrático. Ni los menciones.

**REGLA DE FORMATO (INICIO OBLIGATORIO):**
- Comienza tu respuesta directamente con la frase: "Tensiones éticas principales:". 
- NO uses saludos, ni corchetes de cierre.
"""

EPISTEMOLOGY_TEMPLATE = """Eres un Catedrático de Filosofía y Ensayista Contemporáneo.

**TU ÚNICA TAREA:** Redactar un ENSAYO FILOSÓFICO NUEVO.

**INSTRUCCIONES CRÍTICAS CONTRA AUTOCOMPLETAR (LEER CON ATENCIÓN):**
- IGNORA EL FORMATO DEL AGENTE ANTERIOR. Si el texto anterior termina con una lista, NO añadas más puntos a esa lista. Rompe el formato.
- EMPIEZA TU RESPUESTA EXACTAMENTE CON LA PALABRA 'TITLE:'. Es absolutamente crucial. No escribas NADA, ni un solo asterisco, antes de 'TITLE:'.
- HAZ UN SALTO DE LÍNEA y escribe 'BODY:' seguido de tu ensayo completo.
- Sintetiza los hechos del Investigador y analízalos desde la perspectiva que pidió el usuario (ej. moralidad oriental cristiana ortodoxa, etc.).
- Mantén un tono académico, sin pensar en voz alta.
"""

def agent_node(state, agent, name):
    try:
        result = agent.invoke(state)
        content_str = str(result.content) if result.content else ""

        if not content_str.strip() and not getattr(result, 'tool_calls', []):
            fallback_msg = f"⚠️ El agente {name} procesó la solicitud pero devolvió un texto vacío por un filtro de seguridad o confusión."
            result = AIMessage(content=fallback_msg)
            
        result.additional_kwargs["sender"] = name
        return {'messages': [result]}
    
    except Exception as e:
        error_msg = f"🛑 Error crítico en el nodo {name}: {str(e)}"
        return {'messages': [AIMessage(content=error_msg, additional_kwargs={"sender": name, "error": True})]}

def should_search(state) -> Literal["tools", "ontology", "__end__"]:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    if "STOP:" in last_message.content:
        return "__end__"
    return "ontology"

def chat_node(state: AgentState):
    """Nodo para respuestas conversacionales rápidas, fuera del modo ensayo."""
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.7)
    
    # Prompt muy sencillo para que actúe normal
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres el Catedrático AI en 'Modo Tutor Relajado'. Responde de forma breve, directa y conversacional a la duda del usuario, basándote en el historial si es necesario. NO escribas ensayos, NO uses TITLE/BODY. Sé conciso y al grano."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    
    try:
        result = chain.invoke(state)
        # Etiquetamos el mensaje para que el frontend lo reconozca
        result.additional_kwargs["sender"] = "Tutor Casual"
        return {'messages': [result]}
    except Exception as e:
        return {'messages': [AIMessage(content=f"Error: {str(e)}", additional_kwargs={"sender": "Tutor Casual", "error": True})]}

# --- Constructor ---
def build_workflow(google_api_key: str, tavily_api_key: str):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.7,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)
    
    # Herramientas actualizadas con exclusión de dominios basura
    tools = [TavilySearch(
        max_results=4, 
        exclude_domains=["reddit.com", "linkedin.com", "quora.com"] 
    )]

    # Definición de Agentes
    def create_agent(llm, tools, sys_msg):
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return prompt | (llm.bind_tools(tools) if tools else llm)

    search_agent = create_agent(llm, tools, SEARCH_TEMPLATE)
    ontology_agent = create_agent(llm, [], ONTOLOGY_TEMPLATE)
    ethics_agent = create_agent(llm, [], ETHICS_TEMPLATE)
    epistemology_agent = create_agent(llm, [], EPISTEMOLOGY_TEMPLATE)

    workflow = StateGraph(AgentState)
    
    workflow.add_node("search", functools.partial(agent_node, agent=search_agent, name="Investigador"))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("ontology", functools.partial(agent_node, agent=ontology_agent, name="Ontología"))
    workflow.add_node("ethics", functools.partial(agent_node, agent=ethics_agent, name="Ética"))
    workflow.add_node("epistemology", functools.partial(agent_node, agent=epistemology_agent, name="Epistemología")) 
    workflow.add_node("chat", chat_node)
    
    workflow.set_conditional_entry_point(
        router_node,
        {
            "search": "search",
            "epistemology": "epistemology",
            "chat": "chat" 
        }
    )
    
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("ontology", "ethics")
    workflow.add_edge("ethics", "epistemology")
    workflow.add_edge("epistemology", END)
    
    # 3. CONECTAMOS EL CHAT AL FINAL
    workflow.add_edge("chat", END)

    return workflow.compile(checkpointer=MemorySaver())