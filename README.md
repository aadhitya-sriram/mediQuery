# **MediQuery: Medical Chatbot**

MediQuery is an AI-powered medical assistant chatbot that provides **accurate, evidence-based healthcare responses** in a conversational manner. The chatbot leverages advanced language models and Pinecone for vector-based similarity search, supporting contextual, interactive, and user-friendly communication.

---

## **Features**

- **Medical Assistance**: Provides accurate and evidence-based responses to medical queries.
- **NER Extraction**: Identifies medical entities (e.g., diseases, symptoms, treatments) using a pre-trained NER model (`numind/NuExtract-tiny-v1.5`).
- **Context-Aware Responses**: Uses Pinecone for vector similarity search to retrieve relevant context from the knowledge base.
- **Session Memory**: Maintains conversation history and summarizes ongoing sessions.
- **Streamlit Frontend**: A clean, user-friendly chat interface for querying the chatbot.

---

## **Technologies Used**

- **Frontend**: [Streamlit](https://streamlit.io/) for the interactive chat interface.
- **Backend**:
  - **Pinecone**: Vector database for similarity search.
  - **Transformers**: Pre-trained NER and language models.
  - **LangChain**: For building memory and workflow graphs.
- **Model**: `llama3-8b-8192` for generating responses.
