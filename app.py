"""
Streamlit Web Interface for Multi-Agent RAG System
A user-friendly interface for the truly multi-agentic AI RAG application
"""

import streamlit as st
import os
from dotenv import load_dotenv
from config import Config
from utils import DocumentProcessor, VectorStoreManager, validate_file_type, format_file_size, display_document_stats
from workflow import create_workflow
from agents import create_initial_state

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Agent RAG System",
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
        st.session_state.vector_manager = VectorStoreManager(st.session_state.doc_processor)

def main():
    """Main application function"""
    initialize_session_state()
    
    # App header
    st.title("🤖 Multi-Agent RAG System")
    st.markdown("""
    A truly multi-agentic AI Retrieval-Augmented Generation system where each agent is 
    an **independent entity** with its own GPT model instance, specialized configuration, and 
    autonomous decision-making capabilities. **Powered by OpenAI GPT models for optimal performance.**
    """)
    
    
    # Check OpenAI API key setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("""
        ⚠️ **OpenAI API key not configured**
        
        Please set your API key in the .env file:
        `OPENAI_API_KEY=your-key-here`
        
        Get your API key from: https://platform.openai.com/api-keys
        """)
        st.stop()
    
    # Ollama setup check (commented out - using GPT models for better performance)
    # try:
    #     import requests
    #     response = requests.get("http://localhost:11434/api/tags", timeout=5)
    #     if response.status_code != 200:
    #         raise Exception("Ollama not responding")
    #     
    #     # Check if llama3:latest is available
    #     models = response.json().get("models", [])
    #     llama3_available = any(model.get("name") == "llama3:latest" for model in models)
    #     
    #     if not llama3_available:
    #         st.error("""
    #         ⚠️ **llama3:latest model not found in Ollama**
    #         
    #         Please run: `ollama pull llama3:latest`
    #         """)
    #         st.stop()
    #         
    # except Exception as e:
    #     st.error("""
    #     ⚠️ **Ollama not running or not accessible**
    #     
    #     Please ensure Ollama is running on http://localhost:11434
    #     Run: `ollama serve` or start Ollama desktop application
    #     """)
    #     st.stop()
    
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
        
        # Multi-Agent Metrics
        if st.expander("📊 Agent Performance Metrics"):
            if st.session_state.workflow:
                metrics = st.session_state.workflow.get_agent_metrics()
                for agent_name, agent_metrics in metrics.items():
                    st.markdown(f"**{agent_name.replace('_', ' ').title()}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Calls", agent_metrics["calls"])
                    with col2:
                        st.metric("Success", agent_metrics["successes"])
                    with col3:
                        success_rate = agent_metrics["success_rate"] * 100
                        st.metric("Rate", f"{success_rate:.1f}%")
                    st.divider()
            else:
                st.info("Process documents first to see agent metrics")

def upload_files_interface():
    """Interface for uploading files"""
    st.subheader("Upload Documents")
    
    # Multimodal processing toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'pptx', 'txt', 'xlsx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, PPTX, TXT, XLSX"
        )
    with col2:
        enable_multimodal = st.checkbox(
            "🖼️ Images (CLIP)",
            value=Config.ENABLE_MULTIMODAL,
            help="Fast CLIP-based image embeddings (~50-200ms per image, no API costs)"
        )
        Config.ENABLE_MULTIMODAL = enable_multimodal
        
        # Advanced: GPT-4 Vision captions (optional)
        if enable_multimodal:
            use_gpt_vision = st.checkbox(
                "📝 Detailed Captions",
                value=Config.USE_GPT_VISION_CAPTIONS,
                help="Add GPT-4 Vision descriptions (slower: +5-15s per image, API costs apply)"
            )
            Config.USE_GPT_VISION_CAPTIONS = use_gpt_vision
    
    # Show processing mode info
    if enable_multimodal:
        if Config.USE_GPT_VISION_CAPTIONS:
            st.info("🤖 **AI Mode**: CLIP embeddings + GPT-4 Vision captions (intelligent but slower)")
        else:
            st.success("⚡ **Fast Multimodal**: CLIP embeddings only (~100ms per image)")
    else:
        st.success("⚡ **Text Only**: Tables & text only (fastest)")
    
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
    # Show processing mode
    if Config.ENABLE_MULTIMODAL:
        if Config.USE_GPT_VISION_CAPTIONS:
            st.warning("🤖 **AI Mode**: CLIP + GPT-4 Vision processing. Expect 5-15s per image for detailed captions.")
        else:
            st.success("⚡ **Fast CLIP Mode**: Processing images with local CLIP model (~100ms per image).")
    else:
        st.success("⚡ **Fast Mode Active**: Processing text & tables only.")
    
    # Load documents
    st.info("📖 Loading documents...")
    docs = st.session_state.doc_processor.load_uploaded_files(uploaded_files)
        
    if not docs:
        st.error("❌ No documents could be loaded")
        return
    
    # Chunk documents
    with st.spinner("✂️ Chunking documents..."):
        st.info("✂️ Chunking documents...")
        chunks = st.session_state.doc_processor.chunk_documents(docs)
    
    # Build vector store
    with st.spinner("🔍 Building vector store..."):
        st.info("🔍 Building vector store...")
        success = st.session_state.vector_manager.build_vectorstore(chunks)
    
    if success:
        # Create workflow
        st.session_state.workflow = create_workflow(
            st.session_state.vector_manager.get_retriever()
        )
        
        st.session_state.vectorstore_ready = True
        st.session_state.documents_loaded = True
        
        # Display statistics with preprocessing info
        display_document_stats(docs, chunks, st.session_state.doc_processor)
        
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
            
            # Display statistics with preprocessing info
            display_document_stats(docs, chunks, st.session_state.doc_processor)
            
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
    
    # Show streaming status
    from config import Config
    if Config.ENABLE_STREAMING:
        st.info("⚡ Streaming Enabled (Low Latency)")

def welcome_screen():
    """Display welcome screen when no documents are loaded"""
    st.markdown("""
    ## 🚀 Welcome to Multi-Agent RAG System
    
    This is a **truly multi-agentic AI system** where each agent is an independent entity with its own LLM instance, specialized configuration, and autonomous decision-making capabilities.
    
    ### 🌟 Multi-Agent Architecture:
    - **8 Independent Agents**: Each with dedicated LLM, custom temperature, and specialized role
    - **Inter-Agent Communication**: Agents send messages and coordinate decisions
    - **Autonomous Decision-Making**: Each agent makes independent decisions based on its expertise
    - **Performance Tracking**: Real-time metrics for each agent's success rate
    - **Iterative Refinement**: QualityEvaluator autonomously decides when refinement is needed
    
    ### 🤖 The Agent Team:
    1. **SecurityGuard** (GPT-4o, T=0.1) - Threat detection specialist
    2. **QueryOptimizer** (GPT-4o, T=0.4) - NLP optimization expert
    3. **DocumentRetriever** (GPT-3.5, T=0.0) - Vector search specialist
    4. **AnswerGenerator** (GPT-4o, T=0.3) - Deep reasoning synthesizer
    5. **GroundingValidator** (GPT-4o, T=0.1) - Fact-checking agent
    6. **QualityEvaluator** (GPT-4o, T=0.2) - Metacognitive assessor
    7. **OutputGuard** (GPT-4o, T=0.1) - Final safety validator
    8. **MemoryManager** (GPT-3.5, T=0.0) - Context manager
    
    ### 📋 Getting Started:
    1. **Upload documents** or **load from folder** using the sidebar
    2. **Wait for processing** to complete (vector store creation)
    3. **Start asking questions** - watch agents collaborate!
    4. **View agent metrics** in the sidebar to see performance
    
    ### ✨ What Makes This Multi-Agentic:
    - Each agent has its **own LLM instance** (not shared)
    - Agents use **different models and temperatures** optimized for their role
    - Agents **communicate with each other** via message passing
    - Each agent **tracks its own performance** and decision history
    - Agents make **autonomous decisions** independent of other agents
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
    """Process user question through the multi-agent workflow with streaming"""
    # Add user message to chat
    with st.chat_message("user"):
        st.write(question)
    
    # Process through workflow with intelligent streaming
    with st.chat_message("assistant"):
        # Create initial state
        initial_state = create_initial_state(question, st.session_state.chat_history)
        
        # Phase 1: Pre-processing and answer generation (non-streaming for quality check)
        with st.spinner("🤖 Multi-agent system processing..."):
            workflow = st.session_state.workflow
            
            # Execute security and query optimization
            state = initial_state
            state = workflow.agent_system.get_agent("security_guard").execute(state)
            
            # Check if blocked
            if "blocked" in state.get("answer", "").lower() or "⚠" in state.get("answer", ""):
                st.warning(state["answer"])
                return
            
            state = workflow.agent_system.get_agent("query_optimizer").execute(state)
            state = workflow.agent_system.get_agent("document_retriever").execute(state)
            
            # Generate initial answer (non-streaming for quality evaluation)
            answer_generator = workflow.agent_system.get_agent("answer_generator")
            state = answer_generator.execute(state)
            
            # Validate and check quality
            state = workflow.agent_system.get_agent("grounding_validator").execute(state)
            state = workflow.agent_system.get_agent("quality_evaluator").execute(state)
            
            # Refinement loop if needed (before streaming to user)
            max_refinements = 2
            refinement_count = 0
            while state.get("needs_refinement", False) and refinement_count < max_refinements:
                refinement_count += 1
                # Re-optimize and retrieve
                state = workflow.agent_system.get_agent("query_optimizer").execute(state)
                state = workflow.agent_system.get_agent("document_retriever").execute(state)
                
                # Re-generate answer
                state = answer_generator.execute(state)
                
                # Re-validate
                state = workflow.agent_system.get_agent("grounding_validator").execute(state)
                state = workflow.agent_system.get_agent("quality_evaluator").execute(state)
            
            # Final safety check
            state = workflow.agent_system.get_agent("output_guard").execute(state)
            state = workflow.agent_system.get_agent("memory_manager").execute(state)
        
        # Phase 2: Stream the FINAL answer to user
        # First, display retrieved images if any
        retrieved_images = state.get("retrieved_images", [])
        if retrieved_images:
            st.write("🖼️ **Retrieved Images:**")
            cols = st.columns(min(len(retrieved_images), 3))  # Max 3 columns
            for idx, img_data in enumerate(retrieved_images[:6]):  # Max 6 images
                col_idx = idx % 3
                with cols[col_idx]:
                    try:
                        import base64
                        import io
                        from PIL import Image
                        
                        # Decode base64 image
                        image_bytes = base64.b64decode(img_data['base64'])
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Display image with caption
                        st.image(image, 
                               caption=f"Page {img_data['metadata'].get('page', 'N/A')}: {img_data['caption'][:100]}...",
                               use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not display image: {str(e)}")
        
        # Then stream the text answer
        answer_placeholder = st.empty()
        final_answer = state["answer"]
        
        # Simulate streaming effect for the final answer
        displayed_text = ""
        words = final_answer.split()
        
        for i, word in enumerate(words):
            displayed_text += word + " "
            # Update display with cursor
            answer_placeholder.markdown(displayed_text + "█")
            # Small delay for streaming effect (adjust as needed)
            import time
            time.sleep(0.02)  # 20ms delay per word
        
        # Remove cursor and show final answer
        answer_placeholder.markdown(final_answer)
        
        # Display multi-agent workflow information in expander
        with st.expander("🔍 Multi-Agent Workflow Details"):
            tab1, tab2, tab3 = st.tabs(["Query Processing", "Agent Confidence", "Agent Communications"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Refinement Iterations", state.get("iteration_count", 0))
                    st.write("**Original Query:**")
                    st.write(question)
                
                with col2:
                    st.write("**Optimized Query:**")
                    st.write(state.get("rewritten_question", "N/A"))
                    st.write("**Evaluation Feedback:**")
                    st.write(state.get("evaluation_feedback", "No feedback"))
                
                if state.get("iteration_count", 0) > 0:
                    st.info(f"✨ Answer was autonomously refined {state['iteration_count']} time(s) by QualityEvaluator agent!")
            
            with tab2:
                st.write("**Agent Confidence Scores:**")
                confidence_scores = state.get("confidence_scores", {})
                if confidence_scores:
                    for agent, score in confidence_scores.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(score, text=agent)
                        with col2:
                            st.write(f"{score:.2%}")
                else:
                    st.info("No confidence scores available")
            
            with tab3:
                st.write("**Inter-Agent Communications:**")
                communications = state.get("agent_communications", [])
                if communications:
                    for msg in communications:
                        st.markdown(f"**{msg['from']}** → **{msg['to']}**: {msg['message']}")
                else:
                    st.info("No inter-agent communications recorded")
    
    # Update session state
    st.session_state.chat_history = state["chat_history"]

if __name__ == "__main__":
    main()