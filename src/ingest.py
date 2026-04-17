import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_PATH = "data/mediwaste_info.txt"
CHROMA_PATH = "chroma_db"

def ingest_data():
    print(f"Loading data from {DATA_PATH}...")
    # Load the document
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()

    print("Splitting text into chunks...")
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    print("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    # Using HuggingFace Embeddings (which uses sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Creating vector store at {CHROMA_PATH}...")
    # Create the vector store
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print("Ingestion complete! The Chroma database has been created and saved.")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} does not exist. Please create it first.")
    else:
        ingest_data()
