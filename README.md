# NNP-AI - Manual Payments AI Platform

A modular, containerized, config-driven boilerplate for intelligent document processing with agentic AI.

## 🚀 Quick Start

```bash
# Clone and navigate
cd nnp-ai

# Copy environment template
cp .env.example .env

# Start all services with Docker
docker-compose up -d

# Check status
docker-compose ps
```

**Access Points:**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Agents API: http://localhost:8001
- MCP Tools: http://localhost:8002

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React+Vite)                     │
│                     :3000 - OPS Portal                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   API Service (FastAPI)                      │
│                    :8000 - REST API                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Agents (LangGraph)                          │
│                :8001 - Workflow Orchestration                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  MCP Tools (FastMCP)                         │
│              :8002 - Tool Server (OCR, PDF, Sig)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
nnp-ai/
├── docker-compose.yml      # Container orchestration
├── .env.example            # Environment template
├── config/
│   └── app_config.yaml     # Central configuration
│
├── shared/                 # Shared Python package
│   ├── config/             # Config loader
│   ├── models/             # Pydantic schemas
│   └── adapters/           # Pluggable DB/LLM adapters
│
├── frontend/               # Container 1: OPS Portal
├── api-service/            # Container 2: FastAPI
├── agents/                 # Container 3: LangGraph
└── mcp-tools/              # Container 4: FastMCP
```

---

## 🔌 Swappability (Lego Blocks)

### Change Database
```yaml
# config/app_config.yaml
database:
  type: "postgres"  # sqlite, postgres, mongo
```

### Change LLM Provider
```yaml
# config/app_config.yaml
llm:
  provider: "gemini"  # gemini, openai, azure
  gemini:
    api_key: "${GEMINI_API_KEY}"
```

### Enable/Disable Agents
```yaml
# config/app_config.yaml
agents:
  enabled:
    - extraction
    - signature_detection
    # - verification  # Disabled
```

### Customize Prompts
```yaml
# config/app_config.yaml
prompts:
  extraction: |
    Your custom extraction prompt here...
```

---

## 🐳 Container Commands

```bash
# Build all containers
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api-service
docker-compose logs -f agents

# Stop all services
docker-compose down

# Rebuild single service
docker-compose up -d --build api-service
```

---

## 🧪 Local Development

### API Service
```bash
cd api-service
pip install -r requirements.txt
python main.py
```

### Agents
```bash
cd agents
pip install -r requirements.txt
python main.py
```

### MCP Tools
```bash
cd mcp-tools
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 📚 API Endpoints

### Documents
- `GET /documents` - List all documents
- `POST /documents` - Create document
- `GET /documents/{id}` - Get document
- `PATCH /documents/{id}` - Update document
- `DELETE /documents/{id}` - Delete document
- `GET /documents/{id}/status-history` - Get document status timeline
- `GET /documents/{id}/operation` - Get latest software operation event
- `GET /documents/operations/list` - List software operation events

### Processing
- `POST /process/document` - Start processing
- `GET /process/status/{id}` - Get status
- `POST /process/rerun/{id}` - Re-run processing

### Agents
- `POST /run` - Run full workflow
- `POST /run/extraction` - Run extraction only
- `POST /run/signature` - Run signature only

---

## 📝 License

MIT
