from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
