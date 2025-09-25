# ./backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.rag_pipeline import ask_question
from backend.compare import compare_cars

app = FastAPI(
    title="Automobile Knowledge Assistant",
    description="A RAG-powered assistant for automotive engineering and car comparisons",
    version="1.0.0"
)

# -------------------------------
# Request schema
# -------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  # number of documents to retrieve (default 3)

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Automobile Knowledge Assistant API 🚗"}

@app.post("/ask")
def ask_question_endpoint(request: QueryRequest):
    """
    Ask a question using the RAG pipeline.
    Returns a clean answer and optionally source documents.
    """
    try:
        answer, sources = ask_question(request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "answer": answer,
            "sources": [s.metadata.get("source", "") for s in sources]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}
