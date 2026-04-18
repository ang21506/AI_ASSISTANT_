import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag import get_rag_chain

app = FastAPI(title="MediWaste RAG AI Assistant")

# Configure CORS so the frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

# Initialize the chain globally so it doesn't reload the models on each request
print("Initializing RAG chain...")
try:
    chain = get_rag_chain()
    print("RAG chain initialized successfully.")
except Exception as e:
    print(f"Error initializing RAG chain: {e}")
    chain = None

@app.get("/")
def read_root():
    return {"status": "online", "message": "MediWaste AI Assistant is Ready"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not chain:
        raise HTTPException(status_code=500, detail="RAG chain not initialized. Check server logs.")
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        response = chain.invoke(query)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Important: Run on a different port than the main backend (which is 8000)
    uvicorn.run(app, host="0.0.0.0", port=8002)
