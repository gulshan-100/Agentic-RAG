"""
Streamlit Web Interface for Agentic RAG System
A user-friendly interface for the multi-agent RAG application
"""

import streamlit as st
import os
from dotenv import load_dotenv
from utils import DocumentProcessor, VectorStoreManager, validate_file_type, format_file_size, display_document_stats
from workflow import create_workflow
from agents import create_initial_state

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "workflow" not in st.session_state:
        st.session_state.workflow = None
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if "vector_manager" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager()

def main():
    """Main application function"""
    initialize_session_state()
    
    # App header
    st.title("🤖 Agentic RAG System")
    st.markdown("""
    A sophisticated multi-agent Retrieval-Augmented Generation system with autonomous decision-making, 
    multi-step reasoning, and quality validation.
    """)
    
    # Check API key setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("""
        ⚠️ **OpenAI API key not configured**
        
        Please set your API key in one of these ways:
        1. Create a `.env` file with: `OPENAI_API_KEY=your-key-here`
        2. Set environment variable: `export OPENAI_API_KEY='your-key'`
        
        Get your API key from: https://platform.openai.com/api-keys
        """)
        st.stop()
    
    # Sidebar for document management
    create_sidebar()
    
    # Main interface
    if st.session_state.vectorstore_ready:
        chat_interface()
    else:
        welcome_screen()

def create_sidebar():
    """Create the sidebar for document management"""
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Tab selection
        tab1, tab2 = st.tabs(["Upload Files", "Load Folder"])
        
        with tab1:
            upload_files_interface()
        
        with tab2:
            load_folder_interface()
        
        # System status
        st.header("🔧 System Status")
        display_system_status()
        
        # Workflow visualization
        if st.expander("🔄 View Workflow"):
            if st.session_state.workflow:
                st.text(st.session_state.workflow.get_workflow_visualization())

def upload_files_interface():
    """Interface for uploading files"""
    st.subheader("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'pptx', 'txt', 'xlsx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, PPTX, TXT, XLSX"
    )
    
    if uploaded_files:
        # Display file information
        st.write("📄 Selected Files:")
        for file in uploaded_files:
            size = format_file_size(len(file.getvalue()))
            st.write(f"• {file.name} ({size})")
        
        if st.button("🚀 Process Files"):
            process_uploaded_files(uploaded_files)

def load_folder_interface():
    """Interface for loading documents from folder"""
    st.subheader("Load from Folder")
    
    folder_path = st.text_input(
        "Folder Path",
        placeholder="Enter path to folder containing documents"
    )
    
    if folder_path and st.button("📁 Load Folder"):
        process_folder(folder_path)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and build vector store"""
    with st.spinner("Processing uploaded files..."):
        # Load documents
        st.info("📖 Loading documents...")
        docs = st.session_state.doc_processor.load_uploaded_files(uploaded_files)
        
        if not docs:
            st.error("❌ No documents could be loaded")
            return
        
        # Chunk documents
        st.info("✂️ Chunking documents...")
        chunks = st.session_state.doc_processor.chunk_documents(docs)
        
        # Build vector store
        st.info("🔍 Building vector store...")
        success = st.session_state.vector_manager.build_vectorstore(chunks)
        
        if success:
            # Create workflow
            st.session_state.workflow = create_workflow(
                st.session_state.vector_manager.get_retriever()
            )
            
            st.session_state.vectorstore_ready = True
            st.session_state.documents_loaded = True
            
            # Display statistics
            display_document_stats(docs, chunks)
            
            st.success("✅ Documents processed successfully! You can now start asking questions.")
            st.rerun()
        else:
            st.error("❌ Failed to process documents")

def process_folder(folder_path):
    """Process documents from a folder"""
    if not os.path.exists(folder_path):
        st.error("❌ Folder does not exist")
        return
    
    with st.spinner("Processing folder..."):
        # Load documents
        st.info("📖 Loading documents from folder...")
        docs = st.session_state.doc_processor.load_documents_from_folder(folder_path)
        
        if not docs:
            st.error("❌ No supported documents found in folder")
            return
        
        # Chunk documents
        st.info("✂️ Chunking documents...")
        chunks = st.session_state.doc_processor.chunk_documents(docs)
        
        # Build vector store
        st.info("🔍 Building vector store...")
        success = st.session_state.vector_manager.build_vectorstore(chunks)
        
        if success:
            # Create workflow
            st.session_state.workflow = create_workflow(
                st.session_state.vector_manager.get_retriever()
            )
            
            st.session_state.vectorstore_ready = True
            st.session_state.documents_loaded = True
            
            # Display statistics
            display_document_stats(docs, chunks)
            
            st.success("✅ Documents processed successfully! You can now start asking questions.")
            st.rerun()
        else:
            st.error("❌ Failed to process documents")

def display_system_status():
    """Display current system status"""
    if st.session_state.documents_loaded:
        st.success("✅ Documents Loaded")
    else:
        st.warning("⏳ No Documents Loaded")
    
    if st.session_state.vectorstore_ready:
        st.success("✅ Vector Store Ready")
    else:
        st.warning("⏳ Vector Store Not Ready")
    
    if st.session_state.workflow:
        st.success("✅ Workflow Initialized")
    else:
        st.warning("⏳ Workflow Not Ready")

def welcome_screen():
    """Display welcome screen when no documents are loaded"""
    st.markdown("""
    ## 🚀 Welcome to Agentic RAG System
    
    This advanced AI system uses **multi-agent architecture** to provide intelligent, context-aware responses to your questions.
    
    ### 🌟 Key Features:
    - **8-Agent Pipeline**: Input validation, query optimization, retrieval, generation, evaluation, and safety
    - **Iterative Refinement**: Self-improving answers through autonomous evaluation
    - **Multi-Format Support**: PDF, DOCX, PPTX, TXT, XLSX files
    - **FAISS Vector Search**: High-performance semantic similarity matching
    - **Safety Controls**: Input/output filtering and grounding validation
    
    ### 📋 Getting Started:
    1. **Upload documents** or **load from folder** using the sidebar
    2. **Wait for processing** to complete (vector store creation)
    3. **Start asking questions** about your documents
    
    ### 🤖 How It Works:
    The system uses a sophisticated multi-agent workflow that processes your questions through multiple AI agents, each specialized for specific tasks like security validation, query optimization, document retrieval, answer generation, and quality evaluation.
    """)
    
    # Example questions
    st.markdown("""
    ### 💡 Example Questions You Can Ask:
    - *"What is the main topic of these documents?"*
    - *"Summarize the key points"*
    - *"What are the requirements mentioned?"*
    - *"How do I apply for this position?"*
    - *"What are the important dates?"*
    """)

def chat_interface():
    """Main chat interface for asking questions"""
    st.header("💬 Ask Questions About Your Documents")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        process_question(question)
    
    # Clear chat button
    if st.session_state.chat_history and st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def display_chat_history():
    """Display the conversation history"""
    for i, message in enumerate(st.session_state.chat_history):
        if message.startswith("User: "):
            with st.chat_message("user"):
                st.write(message[6:])  # Remove "User: " prefix
        elif message.startswith("Assistant: "):
            with st.chat_message("assistant"):
                st.write(message[11:])  # Remove "Assistant: " prefix

def process_question(question):
    """Process user question through the agentic workflow"""
    # Add user message to chat
    with st.chat_message("user"):
        st.write(question)
    
    # Process through workflow
    with st.chat_message("assistant"):
        with st.spinner("🤖 Processing through agentic workflow..."):
            # Create initial state
            initial_state = create_initial_state(question, st.session_state.chat_history)
            
            # Run workflow
            result = st.session_state.workflow.invoke(initial_state)
            
            # Display answer
            answer = result["answer"]
            st.write(answer)
            
            # Display workflow information in expander
            with st.expander("🔍 Workflow Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Iterations", result.get("iteration_count", 0))
                    st.write("**Rewritten Query:**")
                    st.write(result.get("rewritten_question", "N/A"))
                
                with col2:
                    st.write("**Evaluation Feedback:**")
                    st.write(result.get("evaluation_feedback", "No feedback"))
                    
                if result.get("iteration_count", 0) > 0:
                    st.info(f"✨ Answer was refined {result['iteration_count']} time(s) for better quality!")
    
    # Update session state
    st.session_state.chat_history = result["chat_history"]

if __name__ == "__main__":
    main()