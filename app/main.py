from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.AIsearch.Support_service import search_support
from app.AIsearch.Metrics_service import search_metrics
from app.utils.logger import logger

app = FastAPI()

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
