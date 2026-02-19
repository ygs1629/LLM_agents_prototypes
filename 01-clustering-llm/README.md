# 🧠 AI Musical Psychoanalyst

## ☁️ Overview
A full-stack data science web application that analyzes your musical taste and delivers a sarcastic, psychoanalytical "roast" of your personality. By extracting the mathematical audio features of your favorite songs, the app groups your taste into a distinct "musical profile" and uses gen AI to judge your personality preferences. 

👉 **[Try the app on Streamlit](https://ai-psycho.streamlit.app/)**

## 💾 Data
The project utilizes a database of ~80.000 tracks. The data pipeline attempts dynamic extraction via the **Spotify API** and features a robust, automated fallback mechanism to a historical **Kaggle Dataset** to bypass recent Spotify API deprecations (Error 403). Extracted features include *danceability, energy, valence, acousticness, and instrumentalness*.

## 🎯 Objective
To build a resilient end-to-end data pipeline that combines unsupervised ML (to objectively cluster musical profiles) with prompt engineering (to translate data into a natural-language, humorous user experience).

## 🛠️ Techniques
* **Data Engineering:** Automated API extraction, dynamic static-fallback routing, and data caching.
* **Unsupervised Machine Learning:** Dimensionality reduction using **PCA** and clustering via **Gaussian Mixture Models (GMM)**.
* **Prompt Engineering:** Implementation of negative constraints and persona adoption to prevent data leakage and force a specific sarcastic tone.

## 📊 Evaluation
* **Clustering (GMM):** Evaluated using the **Bayesian Information Criterion (BIC)** to find the optimal number of components, backed by **Silhouette Scores** to measure cluster separation.
* **Generative AI:** Evaluated qualitatively based on strict prompt adherence (ensuring zero raw-metric leakage into the output) and tone consistency across different musical clusters.

## 🧰 Skills and Tools
* **Languages:** Python
* **Machine Learning:** Scikit-Learn (Pipelines, PCA, GMM), Pandas, NumPy
* **Generative AI:** Google Gemini 2.0 Flash API (free)
* **Web Deployment:** Streamlit
* **Version Control & APIs:** Git, Spotify API (Spotipy), KaggleHub

## ⚠️ Known Limitations (The *Nino Bravo* Anomaly)
While the unsupervised clustering and GenAI pipeline work seamlessly, the model relies purely on mathematical audio features (ignoring metadata like genre or release year). This leads to hilarious but insightful edge cases. 

For example, a classic 1970s romantic ballad with massive vocal power and dense orchestration (like Nino Bravo's *"Un Beso y Una Flor"*) triggers a high `energy` and low `acousticness` score. The GMM mathematically groups this into the **"Aggressive / Heavy"** cluster, and the "blind" LLM proceeds to roast the user as an angry, system-hating internet troll instead of a hopeless romantic. 

**Future Improvements:**

This highlights the inherent limitations of raw acoustic features lacking cultural context. Future iterations would require Hybrid Modeling—injecting categorical metadata (e.g., Spotify's `genre` tags or release decade) as contextual weights for both the GMM and the LLM prompt to fine-tune the psychoanalysis. A possible solution to this would be to define an autonomous AI Agent tasked with searching and retrieving that specific cultural metadata, enriching the context when it is not natively found in the API data or the fallback dataset.

---
*Note: The web application is in Spanish. The description here is in English for portfolio presentation.*