import logging
from langchain_openai import AzureOpenAIEmbeddings
from app.config import Config

logger = logging.getLogger(__name__)

def generate_query_embedding(query: str):
    """Generate embedding for a query string."""
    try:
        embeddings_model = AzureOpenAIEmbeddings(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_deployment=Config.DEPLOYMENT_NAME,
            api_version=Config.API_VERSION
        )
        cleaned = query.replace("\n", " ").replace("\r", " ").strip()
        cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8")
        embedding = embeddings_model.embed_query(cleaned)
        return embedding
    except Exception as e:
        logger.error(f"❌ Query embedding error: {e}")
        return []
