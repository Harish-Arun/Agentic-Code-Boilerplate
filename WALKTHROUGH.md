# NNP-AI Boilerplate - Walkthrough

## Summary

Created a complete, containerized, config-driven boilerplate for the Manual Payments AI Platform. The system is designed as "lego blocks" - each component is independently deployable and swappable.

---

## What Was Built

### 4 Containerized Services

| Container | Technology | Port | Purpose |
|-----------|------------|------|---------|
| **Frontend** | React + Vite | 3000 | OPS Portal UI |
| **API Service** | FastAPI | 8000 | REST API |
| **Agents** | LangGraph | 8001 | Workflow orchestration |
| **MCP Tools** | FastMCP | 8002 | Tool server |

---

## Project Structure

```
nnp-ai/
├── docker-compose.yml              # Orchestrates all 4 containers
├── .env.example                    # Environment template
├── config/
│   └── app_config.yaml             # Central config (swappable components)
│
├── shared/                         # Shared Python module
│   ├── config/loader.py            # Pydantic config loader
│   ├── models/schemas.py           # Shared data models
│   └── adapters/
│       ├── database.py             # SQLite/Postgres/Mongo adapters
│       └── llm.py                  # Gemini/OpenAI/Azure adapters
│
├── frontend/                       # React+Vite OPS Portal
│   ├── src/pages/DocumentList.jsx
│   ├── src/pages/DocumentReview.jsx
│   └── Dockerfile
│
├── api-service/                    # FastAPI REST API
│   ├── main.py
│   ├── routers/documents.py
│   ├── routers/processing.py
│   └── Dockerfile
│
├── agents/                         # LangGraph Workflows
│   ├── main.py
│   ├── graph/workflow.py
│   ├── graph/nodes/extraction.py
│   ├── graph/nodes/signature_detection.py
│   ├── graph/nodes/verification.py
│   └── Dockerfile
│
└── mcp-tools/                      # FastMCP Tool Server
    ├── main.py
    ├── tools/ocr_tool.py
    ├── tools/pdf_utils.py
    ├── tools/signature_provider.py
    └── Dockerfile
```

---

## Swappability Examples

### 1. Swap Database

```yaml
# config/app_config.yaml
database:
  type: "postgres"  # Change from "sqlite"
```

No code changes needed - the adapter factory handles it.

### 2. Swap LLM Provider

```yaml
# config/app_config.yaml  
llm:
  provider: "gemini"  # Default provider
  gemini:
    api_key: "${GEMINI_API_KEY}"
```

### 3. Toggle Features

```yaml
# config/app_config.yaml
agents:
  enabled:
    - extraction
    # - signature_detection  # Comment out to disable
    # - verification
```

### 4. Customize Prompts

```yaml
# config/app_config.yaml
prompts:
  extraction: |
    Your custom extraction prompt...
```

---

## How to Test

### Quick Start
```bash
cd nnp-ai
cp .env.example .env
docker-compose up -d
```

### Verify Services
```bash
# Check all containers are running
docker-compose ps

# Test API health
curl http://localhost:8000/health

# Test Agents health  
curl http://localhost:8001/health

# Test MCP Tools health
curl http://localhost:8002/health
```

### Access UI
- Open http://localhost:3000
- View Document List
- Click on a document to see Review screen

---

## Key Design Decisions

1. **Gemini-powered pipeline** - All services use Gemini for extraction, detection, and verification

2. **Config-driven** - Single YAML file controls all swappable components

3. **Adapter pattern** - Abstract base classes with multiple implementations for DB and LLM

4. **Shared module** - Common code (models, config, adapters) shared across containers

5. **Independent containers** - Each service has its own Dockerfile and requirements

---

## Next Steps

1. Copy `.env.example` to `.env` and configure
2. Run `docker-compose up -d` to start all services
3. Test the flow end-to-end with a real PDF
4. Swap adapters as needed:
   - PostgreSQL/MongoDB for persistence
   - OpenAI or Azure for alternative LLM providers
