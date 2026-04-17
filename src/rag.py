import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# For local LLM (Ollama)
# from langchain_community.llms import Ollama
# For Google Gemini (uncomment if you want to use Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

CHROMA_PATH = "chroma_db"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mediwaste_info.txt"


def load_authoritative_rules():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Authoritative rules file not found: {DATA_PATH}")
    return DATA_PATH.read_text(encoding="utf-8")

def get_rag_chain():
    # 1. Load the database and embeddings
    print("Loading vector database and embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    authoritative_rules = load_authoritative_rules()

    # 2. Prepare the retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 3. Setup the Prompt Template
    template = """You are a helpful and knowledgeable AI assistant for the MediWaste AI project.
    Your job is to answer questions about medical waste segregation and how to use the MediWaste website.
    The "Authoritative Rules" section is the highest-priority source of truth and must override any other context.
    For any waste-bin question, answer strictly from the Authoritative Rules.
    Never mention any bin color other than red, yellow, green, or blue.
    Never use outside medical-waste knowledge to invent or suggest another bin color.
    If a waste item is not defined in the Authoritative Rules, say that it is not defined in the current MediWaste mapping.
    Use the supporting retrieved context only when it does not conflict with the Authoritative Rules.
    Do not add prefatory phrases such as "Based on the provided content" or "According to the provided context".
    Keep your answer clear, concise, and professional.
    Output only the answer text.

    Authoritative Rules:
    {rules}

    Supporting Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # 4. Setup the LLM
    # If the user has Ollama installed locally and running `llama3` or `mistral`
    # print("Initializing LLM (Ollama: llama3)...")
    # llm = Ollama(model="llama3")

    # If you prefer Google Gemini, uncomment the below lines and set GOOGLE_API_KEY environment variable
    print("Initializing LLM (Google Gemini)...")
    import os
    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY environment variable not set. Gemini API will fail.")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # 5. Create the Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "rules": lambda _: authoritative_rules,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def run_assistant():
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: {CHROMA_PATH} does not exist. Please run ingest.py first to create the knowledge base.")
        return

    print("\n=======================================================")
    print("Welcome to MediWaste AI Assistant!")
    print("Ask me any question about medical waste segregation.")
    print("Type 'exit' or 'quit' to stop.")
    print("=======================================================\n")

    chain = get_rag_chain()

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not query.strip():
            continue

        print("\nAI Assistant is thinking...")
        try:
            # We use invoke
            response = chain.invoke(query)
            print(f"\nMediWaste AI:\n{response}")
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Ensure your chosen LLM (like Ollama) is running locally.")

if __name__ == "__main__":
    run_assistant()
