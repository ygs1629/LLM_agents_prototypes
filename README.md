# 🤖 Generative AI & LLM Agents Prototypes

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-MultiAgent-orange.svg)](https://langchain.com/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

This repository serves as a progressive playground in the context of Generative AI and LLM agents applied in my master's degree in AI. It documents the evolution from basic Prompt Engineering and API integrations, through Retrieval-Augmented Generation (RAG) and tool calling, culminating in stateful, Multi-Agent architectures.

---

## 🏗️ Architectural Progression & Key Focus

The core engineering approach across these projects is building **demo versions of AI systems**. These prototypes focus heavily on mitigating the most common pitfalls of modern LLMs:

* **Preventing Hallucinations:** Using strict grounding in local vector databases (RAG) and dynamic web search (Tavily).
* **Mitigating "Role Breakout":** Using advanced Prompt Engineering (Negative constraints, Anti-metatext rules) to ensure AI agents stay in their designated lanes.
* **Evaluation Frameworks:** Implementing *LLM-as-a-Judge* patterns and structured Pydantic outputs to mathematically validate model performance (Recall, Precision, Formatting).
* **State Management:** Transitioning from stateless LCEL chains to cyclical, memory-persistent graphs using `LangGraph`.

---

## 📂 Project Index

Below is a summary of the 5 prototypes included in this repository. **Click on each project's folder for more details.**

### 1. [AI Musical Psychoanalyst](./01-clustering-llm/)
* **Focus:** Hybridizing Unsupervised ML with GenAI.
* **Stack:** Scikit-Learn (PCA, GMM), Spotify API, Gemini 2.0 Flash.
* **Description:** A full-stack app that extracts mathematical audio features from a user's favorite tracks, clusters them using GMM and uses an LLM to deliver a sarcastic, psychoanalytical "roast" of their personality without leaking raw metrics.

### 2. [AI Social Media Copywriter](./02-social-media-generator/)
* **Focus:** LangChain Expression Language (LCEL) & State Management.
* **Stack:** LangChain Core, Groq API (Llama-3-70b), Streamlit.
* **Description:** A lightweight app to cure the "blank page syndrome." It utilizes strict negative constraints to force engaging, hook-driven copywriting tailored to specific social networks, in an attempt to avoid a robotic tone.

### 3. [Basic RAG: Zero-Hallucination PDF Assistant](./03-basic-rag/)
* **Focus:** Vectorization, Embeddings & Semantic Search.
* **Stack:** FAISS, HuggingFace (`all-MiniLM-L6-v2`), PyMuPDF, Groq API.
* **Description:** A localized RAG pipeline that ingests technical PDFs. It demonstrates strict anti-hallucination prompting, forcing the model to answer *only* based on the retrieved context or admit ignorance.

### 4. [AI MLOps & Cloud Architect Tutor](./04-tutor-agent/)
* **Focus:** Tool Calling, AgentExecutor & LLM-as-a-Judge.
* **Stack:** LangChain Agents, DuckDuckGo Search, Custom Python Heuristics, Gemini.
* **Description:** An autonomous MLOps tutor that audits infrastructure code (Docker, K8s, Terraform and the main three cloud providers). This project heavily emphasizes quantitative evaluation, using a secondary LLM to score the primary agent's pedagogical value and precision across a Golden Dataset of 20 edge cases.

### 5. [AI Multi-Agent Philosophy Professor](./05-media-essayist/)
* **Focus:** LangGraph, Multi-Agent Orchestration & Real-time RAG.
* **Stack:** LangGraph, Tavily Search API, MemorySaver.
* **Description:** A stateful Multi-Agent system that routes user queries through the imitation of an academic faculty (Researcher, Ontologist, Ethicist, Epistemologist). It retrieves live news (if needed to respond to the user's query) and synthesizes it into deep philosophical essays while maintaining strict "Lane Keeping" to prevent cascading failures across the graph.

---

## ⚙️ Quick Start

All projects in this repository are deployed as independent Streamlit applications. To run any of them locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/ygs1629/LLM_agents_prototypes.git
   cd LLM_agents_prototypes
   ```
2. Navigate to the desired folder:
   ```bash
    cd 05-multi-agent-philosophy
   ```

3. Install the dependencies (preferably in a dedicated environment):
    ```bash
    pip install -r requirements.txt
    ```

4. Run the streamlit app:
    ``` bash
    streamlit run app.py
    ```

--- 
**Note**: You will need to provide your own free API Keys for Gemini, Groq, Spotify, or Tavily depending on the project, entered directly in the app's UI. The web application is configured in Spanish. The description here is in English for portfolio presentation.