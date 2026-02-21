# ☁️ AI MLOps & Cloud Architect Tutor

## ☁️ Overview
A full-stack, Multi-Agent web application designed to mentor Data Scientists and Machine Learning Engineers transitioning their models from *Jupyter Notebooks* to production environments. The app acts as an expert MLOps tutor, auditing infrastructure code (Kubernetes, CI/CD, GCP, AWS, Azure, etc) and providing pedagogical, contextualized feedback to bridge the gap between Data Science and DevOps.

👉 **[Try the app on Streamlit](https://mlops-agent-tutor.streamlit.app/)** 

## 💾 Data
Instead of a traditional database, this project utilizes a **Golden Dataset** (`test_cases.json`) consisting of 20 stratified, real-world edge cases. This dataset serves as the *Ground Truth* for evaluating the AI. It covers domains such as Docker optimization for massive PyTorch images, K8s OOMKilled debugging, FastAPI Event Loop blocking, and Terraform cloud provisioning.

## 🎯 Objective
To build an autonomous, tool-calling AI Agent capable of contextual reasoning, and to rigorously validate its performance using an advanced **"LLM-as-a-Judge"** evaluation framework. The goal is to ensure the model acts as an empathetic, educational tutor rather than a strict, hallucinating security auditor.

## 🛠️ Techniques
* **Agentic AI & Tool Calling:** Implementation of a LangChain `AgentExecutor` equipped with custom heuristic Python tools (`dockerfile_analyzer`, `yaml_validator`, `terraform_analyzer`) and live web search (`DuckDuckGoSearchRun`). 
* **Prompt Engineering:** Persona adoption, strict formatting constraints, and negative prompting to prevent the "Know-it-all" syndrome (False Positives) and handle malicious Prompt Injections safely.
* **LLM-as-a-Judge Evaluation:** Automated cross-validation using **Pydantic** structured outputs to force a secondary LLM (Gemini 2.0 Flash) to mathematically score the Agent (Gemini 2.5 Flash) based on recall, precision and pedagogical value. 

## 📊 Evaluation
The agent's performance was evaluated across 5 runs per test case (100 total inferences) to measure stability and accuracy. During development, a massive shift in Prompt Engineering was applied—moving from a "Strict DevSecOps Auditor" persona to an "Empathetic MLOps Tutor". 

This alignment drastically improved all metrics, entirely eradicating hallucinations and boosting the pedagogical quality of the responses:

| Evaluation Metric | V1 (Strict Auditor) | V2 (Pedagogical Tutor) | Impact / Insight |
| :--- | :---: | :---: | :--- |
| **Global Average Score** | 7.03 / 10 | **9.13 / 10** | 🚀 Massive leap in reasoning and format adherence. |
| **Stability Index** | 56.1% | **93.3%** | ⚖️ Surpassed the >85% threshold required for production. |
| **Hallucinations (per run)**| 4.30 | **0.17** | 🚨 "False Positives" eradicated. The agent stopped inventing unnecessary requirements. |
| **Concept Coverage Rate** | 100.0% | **100%+** | 🎯 The agent not only found the required issues but proactively identified and explained secondary optimizations. |
| **Pedagogical Value (PVS)**| *N/A* | **8.80 / 10** | 💡 The model successfully learned to explain the *why* behind the infrastructure, rather than just pointing out errors. |

## 🧰 Skills and Tools
* **Languages:** Python
* **Generative AI & Orchestration:** LangChain, Google Gemini 1.5 Flash (Agent), Gemini 2.0 Flash (Judge)
* **Validation & Metrics:** Pydantic, Python `statistics`
* **Web Deployment:** Streamlit

## ⚠️ Known Limitations
While the Agent's reasoning and the LLM-as-a-Judge framework are robust, the custom tools currently rely on basic Python string matching and heuristics. 

If a user inputs highly obfuscated or complex configuration files, the heuristic tools might falsely report the code as "Safe". Thanks to aggressive Prompt Engineering, the LLM is currently trained to distrust its own tools and perform a secondary deep-dive analysis, catching what the regex misses. 

**Future Improvements:**
To make this enterprise-ready, future iterations would replace these basic Python functions with industry-standard CLI linters and Static Application Security Testing (SAST) tools. Integrating tools like **Checkov** (for Terraform), **Trivy** (for Docker vulnerabilities), and **Infracost** (for FinOps) executed via Python's `subprocess` module would provide the LLM with professional-grade JSON reports to interpret, bridging the gap between an educational prototype and a production-grade DevSecOps pipeline.

In addition, it would be a big improvement if there was a human expert in the evaluation part, in order to ensure the quatilty of the ground truth dataset as well as the LLM-as-a-judge qualifications.

---
*Note: The web application is configured in Spanish to cater to a specific user base. The description here is in English for portfolio presentation.*