# 🤖 Agentic RAG System

A sophisticated multi-agent Retrieval-Augmented Generation system with Streamlit interface. Upload documents and ask questions - the system processes your queries through 8 specialized AI agents for intelligent, accurate responses.

## 🌟 Features

- **Multi-Agent Architecture**: 8 AI agents working together (input validation, query optimization, document retrieval, answer generation, quality evaluation, safety checks)
- **Iterative Refinement**: System automatically improves answers through self-evaluation (up to 2 iterations)
- **Multiple Document Types**: PDF, DOCX, PPTX, TXT, XLSX support
- **FAISS Vector Search**: High-performance semantic similarity matching
- **Safety Controls**: Input/output filtering and content validation
- **Web Interface**: Easy-to-use Streamlit interface with chat functionality

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```
*Get your API key from: https://platform.openai.com/api-keys*

### 3. Run the Application
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

## 📖 How to Use

1. **Upload Documents**: Use the sidebar to upload files or load from a folder
2. **Wait for Processing**: The system builds a vector database from your documents
3. **Ask Questions**: Type questions about your documents in the chat
4. **Get Smart Answers**: The system processes through all agents and may refine answers automatically

## 📁 Architecture Overview

**Core Components:**
- **Streamlit Interface**: Web-based user interface for document upload and chat
- **Agent System**: Eight specialized AI agents for different processing tasks
- **Workflow Orchestration**: LangGraph coordinates agent execution and decision flow
- **Document Processing**: Multi-format document loading and text chunking
- **Vector Database**: FAISS-powered semantic search and retrieval
- **Configuration Management**: Centralized settings and environment handling

## ⚙️ Configuration

Edit `config.py` to customize:
- **Models**: Change OpenAI models (default: GPT-4o, text-embedding-3-large)
- **Chunk Size**: Document splitting parameters (default: 1000 chars)
- **Retrieval**: Number of documents to retrieve (default: 5)
- **Iterations**: Max refinement loops (default: 2)

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key
- Internet connection (for OpenAI API calls)

## 🐛 Troubleshooting

**API Key Issues**: Ensure OpenAI API key is properly configured in environment variables

**Dependency Issues**: Verify all required packages are installed correctly

**Document Loading Problems**: Check that file formats are supported and files are accessible

**Performance Issues**: Large documents may require longer processing time for vector database creation

## 📋 Example Questions

- "What is the main topic of these documents?"
- "Summarize the key requirements"
- "How do I apply for this position?" 
- "What are the important dates mentioned?"

## 🔄 How It Works

The system processes your questions through 8 AI agents:

1. **Input Guard** - Validates query safety
2. **Query Rewriter** - Optimizes for better retrieval
3. **Document Retriever** - Finds relevant content using FAISS
4. **Answer Generator** - Creates responses based on context
5. **Grounding Check** - Validates accuracy against sources
6. **Answer Evaluator** - Assesses quality and triggers refinement if needed
7. **Output Guard** - Final safety validation
8. **Memory Manager** - Updates conversation history

If the answer needs improvement, the system automatically refines the query and tries again (up to 2 iterations).

## 📄 License

This project is open source under the MIT License.

---

**Ready to experience intelligent document Q&A with autonomous AI agents!** 🚀