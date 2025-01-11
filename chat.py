import os
import re
import json
import torch
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

class State(MessagesState):
    summary: str

class App:

    def __init__(self, system_prompt, template):
        load_dotenv()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ner_model_name = "numind/NuExtract-tiny-v1.5"
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.model = ChatGroq(model="llama3-8b-8192")
        self.system_prompt = system_prompt
        self.template = template
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.load_knowledge_base()
        self.load_ner()
        self.set_memory()
        print(f"Using device: {self.device}")
        self.query_response(self.system_prompt)

    def set_memory(self):
        self.memory = MemorySaver()
        self.workflow = StateGraph(State)
        self.workflow.add_node("conversation", self.call_model)
        self.workflow.add_node(self.summarize_conversation)
        self.workflow.add_edge(START, "conversation")
        self.workflow.add_conditional_edges("conversation", self.should_continue,)
        self.workflow.add_edge("summarize_conversation", END)
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": "4"}}

    def load_ner(self):
        self.ner_model = AutoModelForCausalLM.from_pretrained(self.ner_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device).eval()
        self.ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name, trust_remote_code=True)

    def load_knowledge_base(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX"))
        self.knowledge_base = PineconeVectorStore(embedding=self.embeddings, index=index)

    def store_past_context(self, page_content):
        document = Document(page_content=page_content)
        self.past_context.add_documents([document])

    def predict_ner(self, texts, batch_size=1, max_length=10_000, max_new_tokens=4_000):
        template = json.dumps(json.loads(self.template), indent=4)
        prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]
        outputs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_encodings = self.ner_tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(self.ner_model.device)
                pred_ids = self.ner_model.generate(**batch_encodings, max_new_tokens=max_new_tokens, pad_token_id = self.ner_tokenizer.eos_token_id)
                outputs += self.ner_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        prediction = [output.split("<|output|>")[1] for output in outputs][0]
        return json.loads(prediction)

    def is_medical_query(self, query):
        entities = self.predict_ner([query])["Medical"]
        medical_entities = [i for i in entities.items() if i[1]]
        if (len(medical_entities) > 0):
            return True
        return False

    def call_model(self, state):
        summary = state.get("summary", "")
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def should_continue(self, state):
        messages = state["messages"]
        if len(messages) > 2:
            return "summarize_conversation"
        return END

    def summarize_conversation(self, state):
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.model.invoke(messages)
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def query_response(self, input_msg):
        input_message = HumanMessage(content=input_msg)
        self.app.invoke({"messages": [input_message]}, self.config)

    def chat(self, query):
        if self.is_medical_query(query):
            context = self.knowledge_base.similarity_search(query)[0].page_content
            query = f"Context: {context}\n\n{query}"

        self.query_response(query)
        return self.app.get_state(self.config).values["messages"][-1].content
