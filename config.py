"""
Configuration file for Multi-Agent RAG Application
Contains all settings, constants, and agent-specific configurations
"""

import os
from typing import List, Dict

class Config:
    """Configuration class for the multi-agent RAG application"""
    
    # OpenAI Configuration - Support both environment and Streamlit secrets
    @classmethod
    def get_openai_key(cls):
        """Get OpenAI API key from environment or Streamlit secrets"""
        # Try environment variable first
        key = os.getenv("OPENAI_API_KEY")
        
        # Try Streamlit secrets for cloud deployment
        if not key:
            try:
                import streamlit as st
                key = st.secrets.get("OPENAI_API_KEY")
            except:
                pass
        
        return key
    
    OPENAI_API_KEY = get_openai_key.__func__()
    
    # Agent-Specific Model Configurations
    AGENT_CONFIGS = {
        "security_guard": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "description": "Security-focused agent with adversarial thinking"
        },
        "query_optimizer": {
            "model": "gpt-4o",
            "temperature": 0.4,
            "description": "Query optimization with creative NLP expertise"
        },
        "document_retriever": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "description": "Vector search and document retrieval specialist"
        },
        "answer_generator": {
            "model": "gpt-4o",
            "temperature": 0.3,
            "description": "Deep reasoning for answer synthesis"
        },
        "grounding_validator": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "description": "Fact-checking and grounding validation"
        },
        "quality_evaluator": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "description": "Metacognitive quality assessment"
        },
        "output_guard": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "description": "Final safety and content filtering"
        },
        "memory_manager": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "description": "Conversation history management"
        }
    }
    
    # Base Model Configuration (fallback)
    EMBEDDING_MODEL = "text-embedding-3-large"
    CHAT_MODEL = "gpt-4o"
    TEMPERATURE = 0.2
    
    # Document Processing Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration  
    SEARCH_TYPE = "mmr"
    RETRIEVAL_K = 5
    
    # Multi-Agent Workflow Configuration
    MAX_ITERATIONS = 2
    ENABLE_AGENT_COMMUNICATION = True
    TRACK_AGENT_METRICS = True
    
    # Agent Performance Thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence for accepting outputs
    REFINEMENT_CONFIDENCE_THRESHOLD = 0.75  # Below this, consider refinement
    
    # Safety Configuration
    BLOCKED_PATTERNS = [
        "ignore previous instructions",
        "jailbreak",
        "override safety",
        "bypass filters"
    ]
    
    BANNED_WORDS = [
        "confidential",
        "private data",
        "classified",
        "sensitive information"
    ]
    
    # Supported File Types
    SUPPORTED_FILE_TYPES = [
        ".pdf",
        ".docx", 
        ".pptx",
        ".txt",
        ".xlsx"
    ]
    
    # Streamlit Configuration
    APP_TITLE = "🤖 Multi-Agent RAG System"
    APP_DESCRIPTION = """
    A truly multi-agentic AI Retrieval-Augmented Generation system where each agent is 
    an independent entity with its own LLM instance, specialized configuration, and 
    autonomous decision-making capabilities. Features inter-agent communication, 
    performance tracking, and coordinated workflow orchestration.
    """
    
    # File Upload Configuration
    MAX_FILE_SIZE = 200  # MB
    UPLOAD_FOLDER = "uploaded_documents"
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate if OpenAI API key is set"""
        key = cls.get_openai_key()
        return key and not key.startswith("your-")
    
    @classmethod
    def get_upload_path(cls) -> str:
        """Get or create upload folder path"""
        if not os.path.exists(cls.UPLOAD_FOLDER):
            os.makedirs(cls.UPLOAD_FOLDER)
        return cls.UPLOAD_FOLDER

# Environment setup avoid exposing API keys
def setup_environment():
    """Setup environment variables if not set"""
    key = Config.get_openai_key()
    if not key or key.startswith("your-"):
        print("⚠️  Warning: Please set your OPENAI_API_KEY in environment variables, .env file, or Streamlit secrets")
        print("   You can set it by: export OPENAI_API_KEY='your-actual-api-key'")
        print("   For Streamlit Cloud: Add OPENAI_API_KEY to your app secrets")
        return False
    return True