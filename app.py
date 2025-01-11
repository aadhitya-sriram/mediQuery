import streamlit as st
from chat import App

if 'app_instance' not in st.session_state:
    st.session_state.app_instance = App(
        system_prompt = """
You are a medical assistant providing clear, accurate, and evidence-based healthcare responses in a natural and user-friendly manner.
Instructions:
- Focus on clarity, accuracy, and user safety.
- Evaluate if the provided context is required first, and then use it to provide a response.
- **Never** mention the context provided in your response.
- If a query is not related to healthcare, politely inform the user and avoid answering unrelated topics.
""",
        template = """{
            "Medical": {
                "Disease": [],
                "Symptom": [],
                "Medication": [],
                "Treatment": [],
                "Procedure": [],
                "Psychological factor": [],
                "Allergen": [],
                "Health Condition": [],
                "Risk Factor": [],
                "Dosage": [],
                "Side Effect": [],
                "Health Metric": [],
                "Medical Terminology": []
            }
        }"""

    )
    st.session_state.messages = []
    st.session_state.summary = ""

app = st.session_state.app_instance

st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Chat with Medical Assistant")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

user_query = st.chat_input("Ask a medical question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        response = app.chat(user_query)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})