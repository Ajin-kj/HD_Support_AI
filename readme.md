
````markdown
# 🚀 FastAPI Azure AI Search POC

This is a Proof-of-Concept (PoC) FastAPI server that integrates with Azure OpenAI and Azure AI Search for support and metrics document search using semantic vector queries.

---

## ✅ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
````

---

### 2. Create & Activate Python Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Create `.env` File

Create a `.env` file in the root directory and add your Azure credentials:

```ini
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
DEPLOYMENT_NAME=your_deployment_name
API_VERSION=your_api_version
SUPPORT_SEARCH_INDEX_NAME=your_search_service_name
METRICS_SEARCH_INDEX_NAME=your_search_service_name
SEARCH_INDEX_NAME=your_search_index_name
SEARCH_API_KEY=your_search_api_key
SEARCH_API_VERSION=your_search_api_version

```

---

### 5. Run the FastAPI Server

```bash
uvicorn app.main:app --reload
```

* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📦 API Endpoints

### `POST /Support_search`

Search support documents.

**Request Body**:

```json
{
  "query": "explain cloud infrastructure",
  "top_k": 3
}
```

---

### `POST /Metrics_search`

Search metrics-related documents.

**Request Body**:

```json
{
  "query": "US revenue metrics Q1",
  "top_k": 2
}
```

---

## 🔧 Notes

* Swagger UI is automatically available when running with `--reload`.
* Ensure Azure resources (OpenAI + Search) are correctly configured.
* Replace placeholders in `.env` with actual Azure credentials before running.


```
