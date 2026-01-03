
"""
api.py - FastAPI REST API for RAG system

WHAT IT DOES:
Exposes RAG system as HTTP REST API
Endpoints:
- POST /query: Ask question and get answer
- GET /health: Check if system is active
- GET /stats: Database statistics

INSTALLATION:
pip install fastapi uvicorn

USAGE:
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

TEST:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does attention work?"}'
"""



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
from pathlib import Path
from contextlib import asynccontextmanager

from .rag import AcademicRAG


rag_system: Optional[AcademicRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager – handles startup and shutdown.
    FastAPI 0.93+ replacement for on_event startup/shutdown.
    """
    global rag_system

    print("\n" + "=" * 60)
    print("RAG API STARTUP")
    print("=" * 60)

    try:
        rag_system = AcademicRAG(
            chroma_dir="chroma_db",
            model="llama3.1:8b",
            top_k=5,
        )
        rag_system.load_vectorstore()

        print("API is ready!")
        print("Swagger docs: http://localhost:8000/docs")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"Error during initialization: {e}")
        print("   The server will start, but the /query endpoint will not work!")

    yield  # server running

    print("\nShutting down API...")


# ============= FASTAPI APP =============

app = FastAPI(
    title="Academic RAG API",
    description="API REST for RAG system on academic pdfs",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= PYDANTIC MODELS =============

class QueryRequest(BaseModel):
    
    question: str = Field(..., description="Question to ask the RAG system")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How does the attention mechanism works?",
                "top_k": 5
            }
        }
        
class Source(BaseModel):
    filename: str
    page: Optional[int] = None
    similarity_score: float
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    contexts_count: int
    processing_time: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The attention mechanism is...",
                "sources": ["attention.pdf", "Vision trans.pdf"],
                "contexts_count": 5,
                "processing_time": 2.3
            }
        }
        

class HealthResponse(BaseModel):

    status: str
    ollama_connected: bool
    vectordb_loaded: bool
    chunks_count: int
    
class StatsResponse(BaseModel):

    total_chunks: int
    total_pdfs: int
    model: str
    collection_name: str

#  ============= ENDPOINTS =============

@app.get("/", tags=["Root"])
async def root():
    
    return {
        "name": "Academic RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }
    
    
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
        
    chunks_count = 0
    if rag_system.vectorstore:
        try:
            chunks_count = rag_system.vectorstore._collection.count()
        except:
            chunks_count = 0
            
    
    ollama_ok = True
    try:
        import requests
        response = requests.get(f"{rag_system.ollama_url}/api/tags", timeout=2)
        ollama_ok = response.status_code == 200
    except:
        ollama_ok = False
    
    return HealthResponse(
        status="healthy" if ollama_ok and chunks_count > 0 else "degraded",
        ollama_connected=ollama_ok,
        vectordb_loaded=rag_system.vectorstore is not None,
        chunks_count=chunks_count
    )
    
    
    
@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    
    if not rag_system or not rag_system.vectorstore:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    chunks_count = rag_system.vectorstore._collection.count()
    
    
    pdfs = set()
    try:
        sample = rag_system.vectorstore.similarity_search("", k=100)
        for doc in sample:
            source = doc.metadata.get('source', '')
            if source:
                pdfs.add(Path(source).name)
    except:
        pass
    
    return StatsResponse(
        total_chunks=chunks_count,
        total_pdfs=len(pdfs),
        model=rag_system.model,
        collection_name="academic_papers"
    )

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check server logs."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="The question cannot be empty"
        )
        
    start_time = time.time()
    
    try:
        result = rag_system.query(
            question=request.question,
            k=request.top_k
        )
        
        processing_time = time.time() - start_time
        
        sources_names = [Path(src).name for src in result['sources']]
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources_names,
            contexts_count=len(result['contexts']),
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while querying: {str(e)}"
        )
        
@app.post("/query/detailed", tags=["RAG"])
async def query_rag_detailed(request: QueryRequest):

    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")
    
    start_time = time.time()
    
    try:
        result = rag_system.query(request.question, k=request.top_k)
        processing_time = time.time() - start_time
        
        # Formatta contexts per output
        contexts_formatted = []
        for ctx in result['contexts']:
            contexts_formatted.append({
                'rank': ctx['rank'],
                'content': ctx['content'][:200] + "...",  # Preview
                'source': Path(ctx['metadata'].get('source', 'unknown')).name,
                'page': ctx['metadata'].get('page', '?'),
                'similarity_score': round(ctx['similarity_score'], 3)
            })
        
        return {
            'answer': result['answer'],
            'contexts': contexts_formatted,
            'sources': [Path(src).name for src in result['sources']],
            'processing_time': round(processing_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    import uvicorn
    
    print(" Starting FastAPI server...")
    print(" API available at: http://localhost:8000")
    print(" Interactive docs: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop\n")

    
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  
    )
