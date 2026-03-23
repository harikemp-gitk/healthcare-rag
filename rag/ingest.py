from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   
from config.models import embeddings
from langchain_chroma import Chroma
import glob

file_paths = glob.glob("data/*.pdf")

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()
    
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="healthcare-documents",
        persist_directory="./chroma_langchain_db",
        )
    vector_store.add_documents(documents)
    return vector_store


if __name__ == "__main__":
    file_paths = glob.glob("data/*.pdf")
    all_documents = []

    for file_path in file_paths:
        documents = load_pdf(file_path)
        docs=split_text(documents)
        all_documents.extend(docs)
    print(f"Total documents after splitting: {len(all_documents)}")
    vector_store = create_vector_store(all_documents)
    print("Documents have been ingested and stored in the vector database.")
