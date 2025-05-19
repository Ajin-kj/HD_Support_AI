from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.AIsearch.agent_service import run_agent_query
from pydantic import BaseModel
from typing import List, Optional
from app.AIsearch.Support_service import search_support
from app.AIsearch.Metrics_service import search_metrics
from app.utils.logger import logger

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["*"] for all origins (not recommended in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 1

@app.post("/Support_search")
def search_handler(request: QueryRequest):
    results = search_support(request.query, request.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No matching documents found.")
    return results

@app.post("/Metrics_search")
def metrics_handler(request: QueryRequest):
    results = search_metrics(request.query, request.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No matching metrics found.")
    return results

class AgentRequest(BaseModel):
    query: str
    session_id: str  # New: required session identifier

@app.post("/ai_agent_query")
def ai_agent_handler(request: AgentRequest):
    try:
        result = run_agent_query(request.query, request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")