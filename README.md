# 🧠 Diabetes Research Assistant

![Architecture Diagram](search4cure-architecture.drawio.png)

https://search4cure-diabetes-941266239729.us-central1.run.app/ 

![sample image](image.png)

![image2](image-1.png))


**Deployment:**
Inside the backend directory execute:
```
gcloud run deploy search4cure-diabetes \
  --source . \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --memory=4Gi \
  --platform=managed \
  --allow-unauthenticated \
  --env-vars-file env.yaml
```

or

```streamlit run main.py```


**Diabetes Research Assistant** is a Streamlit-based application that leverages LLM agents to help researchers, students, and practitioners discover and explore scientific publications focused on the use of machine learning methods in diabetes prediction, treatment, and care management.

This tool provides:
- Summarized insights from research papers  
- Mentions of core topics or methods  
- Links to related or cited articles  
- A conversational interface to interactively query a corpus of academic papers  

---

## 🚀 Features

- 🔍 **Semantic Search**: Ask natural language questions like “What are recent ML methods used in diabetes care?” and get summarized paper insights.
- 📑 **Detailed Summaries**: Extracts summaries, topic mentions, and reference links from specific pages of papers.
- 🧠 **LLM Agent Integration**: Uses an AI agent (e.g., LangChain with OpenAI) to reason over indexed papers and return relevant context-based results.
- 🖥️ **User-Friendly Interface**: A clean, interactive UI built with [Streamlit](https://streamlit.io).

---



