# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Import your modules
from rag_pipeline import ask_question
from compare import compare_cars

app = FastAPI(title="Automobile Knowledge Assistant")

# Request models
class QueryRequest(BaseModel):
    question: str

class CompareRequest(BaseModel):
    car1: str
    car2: str
    features: List[str] = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_rag(req: QueryRequest):
    try:
        answer = ask_question(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
def compare(req: CompareRequest):
    try:
        result = compare_cars(req.car1, req.car2, req.features)
        return {"car1": req.car1, "car2": req.car2, "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
