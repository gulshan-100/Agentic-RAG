"""
Configuration file for Agentic RAG Application
Contains all settings, constants, and environment variables
"""

import os
from typing import List

class Config:
    """Configuration class for the RAG application"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    
    # Model Configuration
    EMBEDDING_MODEL = "text-embedding-3-large"
    CHAT_MODEL = "gpt-4o"
    TEMPERATURE = 0.2
    
    # Document Processing Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration  
    SEARCH_TYPE = "mmr"
    RETRIEVAL_K = 5
    
    # Workflow Configuration  
    MAX_ITERATIONS = 2
    
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
    APP_TITLE = "🤖 Agentic RAG System"
    APP_DESCRIPTION = """
    A sophisticated multi-agent Retrieval-Augmented Generation system that demonstrates 
    true agentic behavior through autonomous decision-making, multi-step reasoning, and quality validation.
    """
    
    # File Upload Configuration
    MAX_FILE_SIZE = 200  # MB
    UPLOAD_FOLDER = "uploaded_documents"
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate if OpenAI API key is set"""
        return cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith("your-")
    
    @classmethod
    def get_upload_path(cls) -> str:
        """Get or create upload folder path"""
        if not os.path.exists(cls.UPLOAD_FOLDER):
            os.makedirs(cls.UPLOAD_FOLDER)
        return cls.UPLOAD_FOLDER

# Environment setup avoid exposing API keys
def setup_environment():
    """Setup environment variables if not set"""
    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY.startswith("your-"):
        print("⚠️  Warning: Please set your OPENAI_API_KEY in environment variables or .env file")
        print("   You can set it by: export OPENAI_API_KEY='your-actual-api-key'")
        return False
    return True