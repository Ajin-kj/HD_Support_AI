import requests
import json
import logging
from app.embedding.generator import generate_query_embedding
from app.config import Config

logger = logging.getLogger(__name__)

def search_support(query: str, k: int = 1):
    """Search support-index for top k matching documents."""
    query_embedding = generate_query_embedding(query)
    if not query_embedding:
        return []

    endpoint = f"https://{Config.SEARCH_SERVICE_NAME}.search.windows.net/indexes/{Config.SUPPORT_SEARCH_INDEX_NAME}/docs/search?api-version={Config.SEARCH_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": Config.SEARCH_API_KEY}
    payload = {
        "vectorQueries": [{"kind": "vector", "vector": query_embedding, "fields": "contentVector", "k": k}],
        "select": "id,filename,lastmodified,content"
    }

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json().get("value", [])
        else:
            logger.error(f"Search failed: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []
