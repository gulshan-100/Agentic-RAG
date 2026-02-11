# 🤖 Multi-Agent RAG System

A truly multi-agentic AI Retrieval-Augmented Generation system where each agent is an independent entity with its own LLM instance, specialized configuration, and autonomous decision-making capabilities.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Create a `.env` file:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### Run
```bash
streamlit run app.py
```


## 🏗️ Architecture

### Multi-Agent System
Eight independent agents, each with dedicated LLM instance and specialized configuration:

| Agent | Model | Temperature | Role |
|-------|-------|-------------|------|
| **SecurityGuard** | GPT-4o | 0.1 | Threat detection & input validation |
| **QueryOptimizer** | GPT-4o | 0.4 | NLP-based query optimization |
| **DocumentRetriever** | GPT-3.5-turbo | 0.0 | Vector similarity search |
| **AnswerGenerator** | GPT-4o | 0.3 | Deep reasoning & answer synthesis |
| **GroundingValidator** | GPT-4o | 0.1 | Fact-checking & validation |
| **QualityEvaluator** | GPT-4o | 0.2 | Metacognitive quality assessment |
| **OutputGuard** | GPT-4o | 0.1 | Safety validation & filtering |
| **MemoryManager** | GPT-3.5-turbo | 0.0 | Conversation history management |

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  SecurityGuard      │
                    │  GPT-4o (T=0.1)     │
                    │  Threat Detection   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │    Safe?            │
                ┌───┤                     ├───┐
            YES │   └─────────────────────┘   │ NO
                │                             ▼
                ▼                      ❌ Query Blocked
        ┌─────────────────────┐
        │  QueryOptimizer      │
        │  GPT-4o (T=0.4)      │◄───────┐ Refinement
        │  Query Enhancement   │        │ Loop (Max 2x)
        └──────────┬───────────┘        │
                   │                     │
                   ▼                     │
        ┌─────────────────────┐         │
        │  DocumentRetriever   │         │
        │  GPT-3.5 (T=0.0)     │         │
        │  FAISS Vector Search │         │
        └──────────┬───────────┘         │
                   │                     │
                   ▼                     │
        ┌─────────────────────┐         │
        │  AnswerGenerator     │         │
        │  GPT-4o (T=0.3)      │         │
        │  Answer Synthesis    │         │
        └──────────┬───────────┘         │
                   │                     │
                   ▼                     │
        ┌─────────────────────┐         │
        │  GroundingValidator  │         │
        │  GPT-4o (T=0.1)      │         │
        │  Fact Checking       │         │
        └──────────┬───────────┘         │
                   │                     │
        ┌──────────┴──────────┐          │
        │   Grounded?         │          │
    ┌───┤                     ├───┐      │
YES │   └─────────────────────┘   │ NO   │
    │                             ▼      │
    ▼                      ❌ Invalid     │
┌─────────────────────┐                  │
│  QualityEvaluator    │                  │
│  GPT-4o (T=0.2)      │                  │
│  Quality Assessment  │                  │
└──────────┬───────────┘                  │
           │                              │
┌──────────┴──────────┐                   │
│  Quality OK?        │                   │
├─────────┬───────────┤                   │
│ YES     │ REFINE    │───────────────────┘
│         └───────────┘
▼
┌─────────────────────┐
│  OutputGuard         │
│  GPT-4o (T=0.1)      │
│  Safety Validation   │
└──────────┬───────────┘
           │
┌──────────┴──────────┐
│      Safe?          │
├───────┬─────────────┤
│ YES   │ NO          │
│       ▼             │
│   ❌ Blocked        │
▼                     
┌─────────────────────┐
│  MemoryManager      │
│  GPT-3.5 (T=0.0)    │
│  History Update     │
└──────────┬──────────┘
           │
           ▼
    ┌─────────────┐
    │   ✅ USER   │
    │  RESPONSE   │
    └─────────────┘
```

**Agent Communication Flow:**
```
Iteration 0:
  QueryOptimizer ────────► DocumentRetriever: "Iteration 0 - Optimized query: [query]"
  DocumentRetriever ─────► AnswerGenerator: "Retrieved 5 documents"
  AnswerGenerator ───────► GroundingValidator: "Iteration 0 - Generated answer: [preview]"
  QualityEvaluator ──────► QueryOptimizer: "Requesting refinement" (if needed)

Iteration 1 (if refinement needed):
  QueryOptimizer ────────► DocumentRetriever: "Iteration 1 - Optimized query: [refined query]"
  DocumentRetriever ─────► AnswerGenerator: "Retrieved 5 documents"
  AnswerGenerator ───────► GroundingValidator: "Iteration 1 - Generated answer: [improved preview]"
  QualityEvaluator ──────► OutputGuard: "Answer approved"
  
OutputGuard ─────────────► MemoryManager: "Output approved"
```
**Example: Query & Answer Evolution Across Iterations**
```
User Question: "How many job openings are there?"

Iteration 0:
├─ Query: "total number of job openings vacancies positions available"
├─ Answer: "The total number of vacancies is 1,815 for Constables..."
└─ Evaluator: Requests refinement for broader context

Iteration 1:
├─ Query: "aggregate count employment opportunities job vacancies open positions"
├─ Answer: "1,815 vacancies as specified in the government notification..."
└─ Evaluator: Approved ✓
```
### Inter-Agent Communication
Agents coordinate via message passing:
```
QueryOptimizer → DocumentRetriever: "Optimized query ready"
DocumentRetriever → AnswerGenerator: "Retrieved 5 documents"
AnswerGenerator → GroundingValidator: "Generated answer preview"
QualityEvaluator → QueryOptimizer: "Requesting refinement" (if needed)
OutputGuard → MemoryManager: "Output approved"
```

### Key Features
- **Independent LLM Instances**: Each agent has its own `ChatOpenAI` object
- **Inter-Agent Communication**: Message passing system for coordination
- **Autonomous Decision-Making**: Each agent evaluates independently
- **Iterative Refinement**: Up to 2 automatic refinement loops with query evolution tracking
- **Performance Tracking**: Real-time metrics per agent
- **Full Transparency**: See queries and answers evolve across iterations

## 🛠️ Tech Stack

### Core
- **Python 3.8+**
- **Streamlit** - Web interface
- **LangGraph** - Multi-agent workflow orchestration
- **LangChain** - LLM framework

### AI/ML
- **OpenAI GPT-4o** - Advanced reasoning agents
- **OpenAI GPT-3.5-turbo** - Lightweight utility agents
- **OpenAI Embeddings** - text-embedding-3-large
- **FAISS** - Vector similarity search

### Document Processing
- **PyPDF** - PDF documents
- **python-docx** - Word documents
- **python-pptx** - PowerPoint presentations
- **openpyxl** - Excel spreadsheets

## 📖 Usage

1. Upload documents (PDF, DOCX, PPTX, TXT, XLSX)
2. Wait for vector store processing
3. Ask questions about your documents
4. View agent metrics and communications in real-time
5. See how queries and answers evolve across refinement iterations

## ⚙️ Configuration

Edit `config.py` for:
- Agent-specific models and temperatures
- Embedding model and chunk size
- Retrieval parameters (K=5, MMR search)
- Max refinement iterations (default: 2)

## 📄 License

MIT License