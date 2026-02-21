import json
import statistics
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agent import process_query  

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# --- 1. ESTRUCTURA AVANZADA DEL JUEZ (ENFOQUE TUTOR) ---
class EvaluationResult(BaseModel):
    correct_issues: int = Field(description="Número de conceptos o problemas correctamente identificados y explicados.")
    missed_issues: list[str] = Field(description="Conceptos esperados (Ground Truth) que el tutor NO mencionó.")
    hallucinated_issues: list[str] = Field(description="Alucinaciones: conceptos avanzados innecesarios, inventos o regaños que abruman al estudiante.")
    severity_accuracy: float = Field(description="De 0.0 a 1.0 indicando si el tutor acertó en la importancia del tema para producción.")
    capacidad_explicacion: float = Field(description="De 0.0 a 1.0. Evalúa la calidad pedagógica: ¿Explicó el 'por qué'? ¿Fue empático y claro? ¿Usó buen formato?")

# --- 2. JUEZ LLM ---
def get_judge():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    return llm.with_structured_output(EvaluationResult)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un Juez Evaluador de Modelos de IA. 
    Evalúas a un Tutor de MLOps que revisa código de Data Scientists novatos en infraestructura.
    
    CÓMO EVALUAR:
    1. Revisa si el tutor abordó los 'Expected Issues' (conceptos/herramientas clave).
    2. PENALIZA en 'hallucinated_issues' si el tutor abruma al estudiante con exigencias innecesarias de DevOps puro, si recomienda cosas destructivas o si es condescendiente.
    3. Premia con un alto 'capacidad_explicacion' si el tutor explica el POR QUÉ de las cosas usando un tono alentador y un formato claro (listas, markdown)."""),
    ("human", """
    CONSULTA DEL ESTUDIANTE: {user_query}
    CÓDIGO A EVALUAR: {input_code}
    CONCEPTOS ESPERADOS (GROUND TRUTH): {expected_issues}
    
    RESPUESTA DEL TUTOR:
    {agent_response}
    """)
])

# --- 3. LÓGICA DE PUNTUACIÓN MATEMÁTICA (50% PESO A LA EXPLICACIÓN) ---
def calculate_final_score(eval_data: EvaluationResult):
    total_expected = eval_data.correct_issues + len(eval_data.missed_issues)
    recall = eval_data.correct_issues / total_expected if total_expected > 0 else 1.0 
    
    total_predicted = eval_data.correct_issues + len(eval_data.hallucinated_issues)
    precision = eval_data.correct_issues / total_predicted if total_predicted > 0 else (1.0 if len(eval_data.hallucinated_issues) == 0 else 0.0)
    
    # Fórmula ponderada orientada a la pedagogía
    score = (
        0.20 * recall +
        0.20 * precision +
        0.10 * eval_data.severity_accuracy +
        0.50 * eval_data.capacidad_explicacion
    ) * 10
    
    return round(score, 2), recall

# --- 4. BUCLE DE EJECUCIÓN ---
def run_evaluation_suite(api_key: str, num_runs: int = 5):
    with open('test_cases.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    judge_llm = get_judge()
    judge_chain = judge_prompt | judge_llm
    
    # Almacenamiento de métricas globales
    global_scores = []
    global_hallucinations = []
    global_capacidad_explicacion = []
    
    total_concepts_expected = 0
    total_concepts_covered = 0
    
    for test in dataset:
        print(f"\n🚀 Evaluando Clase: {test['id']}")
        test_scores = []
        
        # Conceptos a cubrir en este test particular
        concepts_in_test = len(test.get("expected_issues", []))
        total_concepts_expected += (concepts_in_test * num_runs)

        for i in range(num_runs):
            print(f"   🔄 Run {i+1}/{num_runs}...", end=" ", flush=True)
            
            agent_response = process_query(
                user_input=test['user_query'] + f"\n\n```\n{test['input_code']}\n```",
                session_id=f"eval_{test['id']}_{i}",
                api_key=api_key
            )
            
            eval_result = judge_chain.invoke({
                "user_query": test["user_query"],
                "input_code": test["input_code"],
                "expected_issues": json.dumps(test.get("expected_issues", [])),
                "agent_response": agent_response
            })
            
            final_score, run_recall = calculate_final_score(eval_result)
            
            # Recolección de datos
            test_scores.append(final_score)
            global_scores.append(final_score)
            global_hallucinations.append(len(eval_result.hallucinated_issues))
            global_capacidad_explicacion.append(eval_result.capacidad_explicacion)
            total_concepts_covered += eval_result.correct_issues
                
            print(f"Nota: {final_score}/10 | Alucinaciones: {len(eval_result.hallucinated_issues)}")

    # --- 5. PANEL DE CONTROL GLOBAL DEL TUTOR ---
    if global_scores:
        global_avg = statistics.mean(global_scores)
        global_variance = statistics.stdev(global_scores) if len(global_scores) > 1 else 0.0
        
        # Cálculos de los nuevos índices
        stability_index = max(0.0, 100.0 - (global_variance * 10))
        alucinaciones_promedio = statistics.mean(global_hallucinations) if global_hallucinations else 0.0
        conceptos_cubiertos_pct = (total_concepts_covered / total_concepts_expected * 100) if total_concepts_expected > 0 else 100.0
        explicacion_media = statistics.mean(global_capacidad_explicacion) * 10 # Pasado a escala sobre 10

        print("\n" + "="*65)
        print("🎓 MÉTRICAS DE CALIDAD DEL TUTOR MLOPS 🎓")
        print("="*65)
        print(f"   Nota Media Global:         {global_avg:.2f} / 10")
        print(f"   Desviación Típica Total:   ±{global_variance:.2f}")
        print("-" * 65)
        print(f"   ⚖️ Índice de Estabilidad:                {stability_index:.1f}%")
        print(f"   🚨 Alucinaciones (por respuesta):        {alucinaciones_promedio:.2f}")
        print(f"   🎯 Cantidad de Conceptos Cubiertos:      {conceptos_cubiertos_pct:.1f}%")
        print(f"   💡 Capacidad de Explicación (PVS):       {explicacion_media:.2f} / 10")
        print("="*65)

if __name__ == "__main__":
    run_evaluation_suite(api_key=api_key, num_runs=5)