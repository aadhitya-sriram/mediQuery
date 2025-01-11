from dotenv import load_dotenv
import time
import os

load_dotenv()

# from langchain_community.document_loaders import PyPDFLoader
# file_path = "Documents/medical.pdf"
# loader = PyPDFLoader(file_path)
# docs = loader.load()
# print(len(docs))


# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
# all_splits = text_splitter.split_documents(docs)
# print(len(all_splits))


from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("mediquery")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)


# ids = vector_store.add_documents(documents=all_splits)



results = vector_store.similarity_search("What is asthma?")
print(results[0])


def load_pine():
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("mediquery")
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)


# def query_pine(query):
#     results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
#     print(results[0])





#### MISC

        # if os.path.isdir("./knowledge_base"):
        #     self.knowledge_base = FAISS.load_local(
        #         "knowledge_base",
        #         self.embeddings,
        #         allow_dangerous_deserialization=True
        #     )
        # else:
        #     self.knowledge_base = FAISS(
        #         embedding_function=self.embeddings,
        #         index=self.faiss_index,
        #         docstore=InMemoryDocstore(),
        #         index_to_docstore_id={},
        #     )
        #     loader = PyPDFLoader(r'Documents/medical.pdf')
        #     documents = loader.load()
        #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        #     chunks = text_splitter.split_documents(documents)
        #     uuids = [str(uuid4()) for _ in range(len(chunks))]
        #     self.knowledge_base.add_documents(documents=chunks, ids=uuids)
        #     self.knowledge_base.save_local("knowledge_base")