from langchain_chroma import Chroma
from config.models import embeddings


vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="healthcare-documents",
    persist_directory="./chroma_langchain_db",
    )   

def get_retriever():
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever