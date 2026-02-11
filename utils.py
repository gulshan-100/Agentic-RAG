"""
Utility functions for document processing and vector store management
"""

import os
import pandas as pd
import streamlit as st
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentProcessor:
    """Handle document loading and processing operations"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """Load documents from a folder"""
        docs = []
        
        if not os.path.exists(folder_path):
            st.error(f"Folder {folder_path} does not exist")
            return docs
            
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            docs.extend(self.load_single_document(file_path))
            
        return docs
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its file type"""
        docs = []
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
            elif file_extension == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                
            elif file_extension == ".pptx":
                loader = UnstructuredPowerPointLoader(file_path)
                docs = loader.load()
                
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
            elif file_extension == ".xlsx":
                df = pd.read_excel(file_path)
                content = df.to_string(index=False)
                docs = [Document(
                    page_content=content,
                    metadata={"source": os.path.basename(file_path), "type": "excel"}
                )]
                
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            st.error(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            
        return docs
    
    def load_uploaded_files(self, uploaded_files) -> List[Document]:
        """Load documents from Streamlit uploaded files"""
        docs = []
        upload_folder = "uploaded_documents"
        
        # Create upload folder if it doesn't exist
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_folder, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_docs = self.load_single_document(file_path)
            docs.extend(file_docs)
            
            # Clean up temporary file
            try:
                os.remove(file_path)
            except:
                pass
                
        return docs
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not docs:
            return []
        return self.splitter.split_documents(docs)

class VectorStoreManager:
    """Handle vector store operations"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings with proper error handling"""
        try:
            # Try multiple sources for API key
            api_key = None
            
            # Try environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            
            # Try Streamlit secrets for cloud deployment
            if not api_key:
                try:
                    api_key = st.secrets.get("OPENAI_API_KEY")
                except:
                    pass
            
            # Try from Config class
            if not api_key:
                from config import Config
                api_key = Config.OPENAI_API_KEY
            
            if not api_key or api_key.startswith("your-"):
                st.error("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")
                st.info("For Streamlit Cloud: Go to your app settings and add OPENAI_API_KEY to secrets")
                self.embeddings = None
                return False
                
            # Set environment variable for OpenAI client
            os.environ["OPENAI_API_KEY"] = api_key
                
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=api_key
            )
            
            # Test the embeddings with a simple query
            test_result = self.embeddings.embed_query("test")
            if test_result:
                return True
            else:
                st.error("Failed to validate embeddings")
                self.embeddings = None
                return False
                
        except Exception as e:
            st.error(f"❌ Failed to initialize embeddings: {str(e)}")
            st.info("Please check your OpenAI API key configuration")
            self.embeddings = None
            return False
    
    def build_vectorstore(self, chunks: List[Document]) -> bool:
        """Build FAISS vector store from document chunks"""
        if not chunks:
            st.error("No document chunks to process")
            return False
        
        # Check if embeddings are properly initialized
        if self.embeddings is None:
            st.error("❌ Embeddings not initialized. Cannot build vector store.")
            if not self._initialize_embeddings():
                return False
            
        try:
            with st.spinner("Building vector store..."):
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5}
                )
            return True
            
        except Exception as e:
            st.error(f"Failed to build vector store: {str(e)}")
            return False
    
    def get_retriever(self):
        """Get the retriever instance"""
        return self.retriever
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vectorstore:
            return []
            
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []

def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    supported_extensions = [".pdf", ".docx", ".pptx", ".txt", ".xlsx"]
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in supported_extensions

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def display_document_stats(docs: List[Document], chunks: List[Document]):
    """Display document statistics in Streamlit"""
    if docs:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents Loaded", len(docs))
        
        with col2:
            st.metric("Text Chunks Created", len(chunks))
        
        with col3:
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            st.metric("Total Characters", f"{total_chars:,}")