# main.py
import uvicorn
import time
import logging
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

from rag_pipeline import ScienceRAG

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

rag_system = None

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, str]] = []
    process_time: float

class HealthResponse(BaseModel):
    status: str
    system_ready: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    logger.info("⏳ Starting RAG system...")
    try:
        rag_system = ScienceRAG()
        logger.info("✅ RAG system ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG: {e}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down...")
    # Очистка ресурсов при необходимости
    if rag_system:
        # Здесь можно добавить cleanup методы
        pass

app = FastAPI(
    title="Science RAG API",
    description="Retrieval-Augmented Generation for Scientific Papers",
    version="1.0.0",
    lifespan=lifespan
)

# CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Science RAG API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Проверка статуса системы"""
    return HealthResponse(
        status="ok" if rag_system else "loading",
        system_ready=rag_system is not None
    )

@app.post("/ask", response_model=QueryResponse, tags=["RAG"])
def ask_question(request: QueryRequest):
    """
    Основной endpoint для вопросов.
    
    - **text**: Вопрос пользователя
    - **top_k**: Количество источников (1-20)
    """
    if not rag_system:
        raise HTTPException(
            status_code=503, 
            detail="System is still loading. Please try again in a few seconds."
        )

    try:
        start = time.time()
        result = rag_system.answer(request.text, top_k=request.top_k)
        duration = time.time() - start
        
        logger.info(f"Query processed in {duration:.2f}s: '{request.text[:50]}...'")
        
        return QueryResponse(
            query=request.text,
            answer=result["answer"],
            sources=result["sources"],
            process_time=round(duration, 2)
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
