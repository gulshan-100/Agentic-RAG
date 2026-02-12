# 🤖 Multi-Agent RAG System

![Multi-Agent RAG Demo](image.png)

A truly multi-agentic AI Retrieval-Augmented Generation system where each agent is an independent entity with its own LLM instance, specialized configuration, and autonomous decision-making capabilities.

## 🚀 Quick Start

### Prerequisites
For full multimodal capabilities, install system dependencies:

**Windows:**
1. **Poppler** (for PDF image extraction)
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases/
   - Extract and add `bin/` folder to your system PATH
2. **Tesseract OCR** (optional, for image text extraction)
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki

**Mac:**
```bash
brew install poppler tesseract
```

**Linux:**
```bash
sudo apt-get install poppler-utils tesseract-ocr
```

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
- **🎯 Multimodal Intelligence**: Advanced image and table understanding

## 🖼️ Multimodal RAG Capabilities

### Fast CLIP-Based Image Processing ⚡
The system now uses **CLIP (Contrastive Language-Image Pre-Training)** for lightning-fast image embeddings:

**Why CLIP?**
- ⚡ **Speed**: ~50-200ms per image (vs 5-15 seconds with GPT-4 Vision)
- 💰 **Cost**: No API costs - runs locally
- 🎯 **Quality**: State-of-the-art multimodal embeddings
- 🔍 **Search**: Direct image-text similarity in shared embedding space

**Processing Pipeline:**
1. **Extraction**: Uses PyMuPDF (fitz) to extract images from PDFs
2. **Filtering**: Skips small images (logos/icons) - only processes images ≥150px
3. **CLIP Embedding**: Local CLIP model generates image embeddings (~100ms per image)
4. **Optional Captions**: GPT-4 Vision can add detailed descriptions (enable via UI checkbox)

**Configuration:**
```python
# config.py
ENABLE_MULTIMODAL = True  # Enable image processing
USE_CLIP_EMBEDDINGS = True  # Fast CLIP embeddings (default)
USE_GPT_VISION_CAPTIONS = False  # Optional detailed captions (slower)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # Fast & accurate
```

**Performance Comparison:**
| Mode | Speed per Image | API Cost | Quality |
|------|----------------|----------|---------|
| CLIP Only | 50-200ms | Free | Excellent |
| GPT-4 Vision | 5-15 seconds | ~$0.03 | Detailed Captions |
| CLIP + GPT-4V | ~5-15 seconds | ~$0.03 | Best of Both |

**Example:**
```
User Query: "What does the system architecture diagram show?"

Retrieved Context:
[🖼️ IMAGE with CLIP embedding]
[IMAGE 1 on page 3]: png, 800x600px. Context: Multi-agent architecture 
settings listed.

Answer: Based on the architecture diagram on page 3, the system uses 
8 specialized agents...
```

### Complete Table Extraction
The system extracts **complete raw tables** without summarization for precise data access:

**Pipeline:**
1. **Extraction**: 
   - Excel files: Pandas reads all sheets with full data preservation
   - PDFs: Pattern matching detects pipe/tab-separated tables
2. **Structured Format**: Tables converted to readable text format:
   ```
   Table: Sales_Data
   Columns: Product, Q1, Q2, Q3, Q4, Total
   Product: Widget A; Q1: 150; Q2: 200; Q3: 175; Q4: 225; Total: 750
   Product: Widget B; Q1: 120; Q2: 180; Q3: 165; Q4: 195; Total: 660
   ```
3. **Metadata**: Stores column names, row count, data types
4. **No Summarization**: Complete tables embedded as-is for accurate value extraction
5. **Direct Querying**: Answers extract specific values from full table data

**Example:**
```
User Query: "What were Widget A sales in Q3?"

Retrieved Context:
[📋 TABLE]
Table: Sales_Data
Columns: Product, Q1, Q2, Q3, Q4, Total
Product: Widget A; Q1: 150; Q2: 200; Q3: 175; Q4: 225; Total: 750
Product: Widget B; Q1: 120; Q2: 180; Q3: 165; Q4: 195; Total: 660

Answer: According to the Sales_Data table, Widget A had 175 sales in Q3.
```

**Benefits:**
- **Precision**: Exact values extracted from complete data, not summaries
- **Flexibility**: Can answer any query about the table without information loss
- **Accuracy**: No summarization = no data loss or distortion

### Multimodal Query Understanding
The **DocumentRetriever** and **AnswerGenerator** agents are enhanced with multimodal awareness:

- **Query Detection**: Automatically detects when users ask about "images", "diagrams", "tables", "charts"
- **Content Tagging**: Retrieved chunks tagged with 📋 TABLE or 🖼️ IMAGE indicators
- **Specialized Responses**: AnswerGenerator uses different strategies for text vs. table vs. image questions
- **Confidence Boosting**: Retrieval confidence increased when multimodal content is found

### Processing Statistics
After document upload, view detailed multimodal metrics:
- **Images Extracted**: Raw images pulled from PDFs
- **Images Captioned**: Number of GPT-4 Vision API calls
- **Multimodal Embeddings**: Images prepared for multimodal embedding models
- **Tables Extracted**: Complete raw tables preserved
- **Docs with Tables/Images**: Count of documents containing structured data

**Why This Approach:**
1. **Tables**: No summarization = no information loss, enables precise value extraction
2. **Images**: Multimodal embeddings allow visual similarity search beyond text captions
3. **Cost-Efficient**: One-time processing (caption/extract per image/table, not per query)
4. **Accuracy**: Complete data preserved for exact answers

## 🛠️ Tech Stack

### Core
- **Python 3.8+**
- **Streamlit** - Web interface
- **LangGraph** - Multi-agent workflow orchestration
- **LangChain** - LLM framework

### AI/ML
- **OpenAI GPT-4o** - Advanced reasoning + Vision capabilities
- **OpenAI GPT-3.5-turbo** - Lightweight utility agents
- **OpenAI Embeddings** - text-embedding-3-large
- **FAISS** - Vector similarity search

### Document Processing
- **PyPDF** - PDF text extraction
- **PyMuPDF (fitz)** - PDF image extraction
- **pdf2image** - PDF to image conversion
- **Pillow (PIL)** - Image processing
- **python-docx** - Word documents
- **python-pptx** - PowerPoint presentations
- **openpyxl** - Excel spreadsheets
- **pandas** - Structured data processing

### Multimodal Processing
- **GPT-4 Vision (gpt-4o)** - Image understanding and captioning
- **PyMuPDF** - Image extraction from PDFs
- **Pillow** - Image format handling
- **Base64 encoding** - Image transmission to vision API
- **Multimodal embeddings** - Visual + text embedding support

## 📖 Usage

### ⚡ Fast Mode vs 🤖 AI Mode

**Fast Mode (Default - Recommended):**
- ✅ Quick processing: ~5-10 seconds for typical documents
- ✅ Extracts: Text, tables, document structure
- ❌ Skips: Image captioning
- **Use when**: You need quick results or documents don't have important images

**AI Mode (Multimodal):**
- ✅ Intelligent: GPT-4 Vision captions every image
- ✅ Extracts: Everything (text, tables, images with descriptions)
- ⏱️ Slower: Adds ~10-30 seconds per page with images
- **Use when**: Images contain important information (diagrams, charts, screenshots)

**To enable AI Mode**: Check the **🖼️ Images** checkbox before clicking "Process Files"

### Steps:

1. Upload documents (PDF, DOCX, PPTX, TXT, XLSX)
2. **Choose processing mode**: 
   - ⚡ Fast Mode (uncheck 🖼️ Images) - Quick text & tables
   - 🤖 AI Mode (check 🖼️ Images) - Full multimodal intelligence
3. Wait for vector store processing
   - Text chunks embedded with OpenAI text-embedding-3-large
   - Tables: Complete raw data extracted and embedded
   - Images (AI mode): GPT-4 Vision captions + multimodal embedding preparation
4. Ask questions about your documents
   - Regular text questions
   - Table queries: "What's the value in column X?"
   - Image queries (AI mode): "What does the diagram show?"
5. View agent metrics and communications in real-time
6. See how queries and answers evolve across refinement iterations

### Multimodal Embeddings (Advanced)
Images are stored with both text captions and base64-encoded data for future multimodal embedding:
- Access via `processor.get_multimodal_image_documents()`
- Returns list of dicts with: base64 image, GPT-4V caption, metadata, context
- Can be embedded with models like CLIP, ImageBind, or OpenAI's multimodal embeddings
- Enables visual similarity search beyond text-based retrieval

## ⚙️ Configuration

Edit `config.py` for:
- Agent-specific models and temperatures
- Embedding model and chunk size
- Retrieval parameters (K=5, MMR search)
- Max refinement iterations (default: 2)
- **Multimodal settings**: `ENABLE_MULTIMODAL`, `MIN_IMAGE_SIZE`, `IMAGE_CAPTION_DETAIL`
- **Table settings**: `ENABLE_TABLE_EXTRACTION`, `TABLE_SUMMARIZATION` (False for raw data)

## 📄 License

MIT License