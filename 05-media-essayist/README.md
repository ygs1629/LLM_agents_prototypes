# AI Multi-Agent Philosophy Professor

## ☁️ Overview

A full-stack, Multi-Agent web application powered by LangGraph designed to analyze current events through a deep philosophical lens. The app acts as an academic faculty, routing user queries through specialized AI agents (Researcher, Ontologist, Ethicist, and Epistemologist) to synthesize real-time news into profound, university-level essays. It also features a parallel "Casual Tutor" node for follow-up discussions and dialectical debate.

👉 **[Try the app on Streamlit](https://media-essayist.streamlit.app/)**

## 💾 Data

Instead of a static dataset or traditional database, this project utilizes real-time internet data via the Tavily Search API. It operates dynamically, pulling empirical facts from current news to ground the theoretical analysis (Retrieval-Augmented Generation). The application seamlessly handles dynamic JSON/List payloads from the web and formats them into a live bibliography.

## 🎯 Objective

To architect a robust, stateful Multi-Agent system capable of deep reasoning. The main challenge was to enforce strict role segregation ("Lane Keeping") and prevent cascading failures across the graph, ensuring each agent performs its specific task (e.g., empirical data retrieval vs. moral analysis) without overstepping, hallucinating, or breaking the conversational state.

## 🛠️ Techniques

- **Multi-Agent Orchestration**: Implementation of a LangGraph StateGraph to manage the sequential and conditional flow of data between multiple distinct AI personas.
- **Semantic Routing**: An LLM-powered Router (Evaluator) that classifies user intents to direct traffic efficiently (e.g., triggering the full research pipeline for news vs. the casual node for simple chat).
- **Retrieval-Augmented Generation (RAG)**: Live web search integration to ground abstract philosophical analysis in empirical reality.
- **Advanced Prompt Engineering**: Utilizing hard context breaks, negative prompting, and anti-metatext rules to prevent "Role Breakout" (agents stealing tasks) and "Text Continuation Leaking" (the autocomplete effect).
- **Stateful Memory**: Utilizing LangGraph's MemorySaver to maintain deep conversational context across multiple turns and tangential questions.

## 📊 Evaluation & Architectural Stability

Because this is a complex architectural project, evaluation focused on system stability, behavioral control, and edge-case resilience rather than pure quantitative metrics. During development, several critical LLM behaviors were identified and eradicated:

| Architectural Challenge | Solution Implemented | Impact / Insight |
|------------------------|---------------------|------------------|
| **Role Breakout** (Agents doing others' jobs) | Strict "Lane Keeping" instructions | 🚀 Agents successfully learned to ignore parts of the prompt not relevant to their specific role |
| **The "Autocomplete" Effect** | Hard context breaks and strict formatting triggers (TITLE:) | 🎯 Eradicated formatting hallucinations. The final agent stopped "finishing the sentences" of previous agents |
| **Silent Tool Call Failures** | Enhanced agent_node fallback logic for API empty responses | ⚖️ Stabilized the RAG pipeline. The graph now seamlessly handles tool execution without crashing |
| **State Amnesia** | Integration of MemorySaver thread configurations | 💡 The system successfully proved its ability to recall deep historical context turns later without losing the LangGraph state |

## 🧰 Skills and Tools

- **Languages**: Python
- **Generative AI & Orchestration**: LangGraph, LangChain, Google Gemini 2.5 Flash, Tavily Search API
- **Validation & Logic**: Pydantic (Structured Outputs), Abstract Syntax Trees (ast)
- **Web Deployment**: Streamlit



## ⚠️ Known Limitations

While the Multi-Agent architecture is highly resilient, it inherits some limitations from current LLM behaviors:

- **Safety Filter Collisions**: When analyzing raw, sensitive news (e.g., crime, geopolitical conflicts) through an ethical lens, the underlying LLM's safety filters can sometimes trigger, resulting in empty outputs despite aggressive prompt handling.
- **Context Loss on Extreme Ambiguity**: If a user asks highly vague follow-up questions using ambiguous pronouns, the LLM might lose the specific contextual anchor despite the state memory, defaulting to generic answers.
- **Fact-Checking (Hallucination Risk)**: The system lacks a definitive "Self-Correction" loop. There is a slight risk that the Epistemologist agent might slightly distort the Researcher's empirical data to fit the requested philosophical narrative.

## 🔮 Future Improvements

To make this enterprise-ready, future iterations would implement:

- An intermediate **"LLM-as-a-judge"** node to fact-check the final essay against the initial Tavily sources before displaying it to the user
- **Export functionality** (PDF/Markdown) for the generated essays
- Isolating LangGraph thread memories by specific philosophical topics to prevent cross-contamination in long, multi-topic sessions

---

**Note**: The web application is configured in Spanish. The description here is in English for portfolio presentation.