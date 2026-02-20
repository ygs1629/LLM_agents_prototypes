# ✍️ AI social media copywriter: the blank page killer

## ☁️ Overview
A lightweight, LangChain-powered web application designed to generate platform-specific social media drafts. Using the Llama-3-70b model via the Groq API, it takes simple topics and translates them into formatted content ready for platforms like LinkedIn, X (Twitter), and Instagram. It serves as a practical portfolio prototype demonstrating API integration and UI state management.

👉 **[Try the app here on Streamlit](https://social-media-lcel.streamlit.app/)**

## 🎯 Objective
To demonstrate a simple yet effective implementation of the LangChain Expression Language (LCEL) with advanced prompt engineering, resulting in a functional marketing tool.

## 🛠️ Techniques
* **LangChain Orchestration (LCEL):** Seamlessly chaining `ChatPromptTemplate`, `ChatGroq`, and `StrOutputParser` for modular and efficient text generation.
* **Advanced Prompt Engineering:** Implementation of strict negative constraints (e.g., preventing the "Wikipedia Syndrome" of defining concepts) to force engaging, hook-driven copywriting instead of robotic text.
* **State Management:** Utilizing Streamlit's `session_state` to ensure generated outputs persist across UI interactions without reloading.

## 🧰 Skills and tools
* **Languages:** Python
* **Frameworks:** LangChain Core, Streamlit
* **LLM Provider:** Groq API (Llama-3-70b-versatile, free version)
* **Concepts:** LCEL, Prompt Engineering, UI/UX Layout Design

## 🚧 Known limitations
While this prototype effectively cures the "blank page syndrome", it faces inherent limitations of static LLM architectures. Therefore, the next evolutionary steps would be towards handling:

1. **The brand voice bottleneck (few-shot prompting):** Currently, the app relies on generalized tones ("Informative", "Humorous"). The natural "AI Robot Talk" bias is hard to eliminate entirely. 
   * *Next Step:* Implement a dynamic **few-shot prompting** feature, by allowing users to input up to 3 of their past successful posts, enabling the LLM to clone their exact tone pattern.
2. **Real-time blindness and token math (evolving into LangGraph):** The static LLM cannot access today's news for trendy events and it also inherently struggles with strict character limits (like X's 280 chars) because LLMs compute tokens, not letters.
   * *Next Step:* Evolve the architecture from a linear LCEL chain to a cyclical Multi-Agent system using **LangGraph**. This would integrate external tools (like web search) and a validation loop (an agent that checks the character count and rewrites the tweet if it exceeds 280 chars before showing it to the user).

---

*Note: The web application is in Spanish. The description here is in English for portfolio presentation.*