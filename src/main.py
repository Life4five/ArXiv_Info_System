# main.py
import uvicorn
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Импорт из соседнего файла
from rag_pipeline import ScienceRAG

rag_system = None

class QueryRequest(BaseModel):
    text: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    process_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    print("⏳ Запуск... Грузим модели...")
    rag_system = ScienceRAG() # Параметры по умолчанию уже настроены в классе
    yield
    print("🛑 Остановка...")

app = FastAPI(title="Science RAG API", lifespan=lifespan)

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="System loading...")
    
    start = time.time()
    answer = rag_system.answer(request.text, top_k=request.top_k)
    duration = time.time() - start
    
    return QueryResponse(query=request.text, answer=answer, process_time=round(duration, 2))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)