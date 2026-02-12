"""
Utility functions for document processing and vector store management
Includes advanced data engineering, preprocessing, and multimodal RAG capabilities
"""

import os
import re
import io
import base64
import hashlib
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Optional
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
import unicodedata

# Multimodal imports
try:
    from pdf2image import convert_from_path
    import fitz  # PyMuPDF
    from PIL import Image
    from openai import OpenAI
    MULTIMODAL_AVAILABLE = True
    
    # CLIP for fast image embeddings (NEW!)
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        CLIP_AVAILABLE = True
    except ImportError:
        CLIP_AVAILABLE = False
        print("⚠️ CLIP not available. Install: pip install transformers torch torchvision")
        
except ImportError:
    MULTIMODAL_AVAILABLE = False
    CLIP_AVAILABLE = False
    st.warning("⚠️ Multimodal dependencies not installed. Image extraction disabled. Run: pip install pdf2image pymupdf Pillow openai")

# Load environment variables
load_dotenv()

class CLIPEmbeddings:
    """
    Unified CLIP embeddings for both text and images
    This creates a shared embedding space for true multimodal retrieval
    """
    def __init__(self, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        
    def embed_documents(self, texts):
        """Embed multiple text documents using CLIP text encoder"""
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query text using CLIP text encoder"""
        return self._embed_text(text)
    
    def embed_image(self, image_data: bytes):
        """Embed image using CLIP image encoder (for manual image embedding)"""
        try:
            # Load image from bytes  
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # Convert to list
                embedding = image_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            print(f"⚠️ Error generating CLIP image embedding: {str(e)}")
            return [0.0] * 512  # Return zero vector as fallback
    
    def _embed_text(self, text):
        """Generate CLIP embedding for text"""
        try:
            # Process text with CLIP
            inputs = self.clip_processor(text=[text], return_tensors="pt", truncation=True)
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate text embedding
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Convert to list
                embedding = text_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            print(f"⚠️ Error generating CLIP text embedding: {str(e)}")
            return [0.0] * 512  # Return zero vector as fallback

# Load environment variables

class DocumentProcessor:
    """Handle document loading and processing operations with advanced preprocessing"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        self.preprocessing_stats = {
            "total_docs": 0,
            "cleaned_chars": 0,
            "tables_extracted": 0,
            "images_found": 0,
            "images_extracted": 0,
            "images_captioned": 0,
            "multimodal_embeddings": 0,
            "duplicates_removed": 0
        }
        
        # Store image data for multimodal embedding
        self.image_documents = []
        
        # Initialize CLIP model for fast image embeddings (NEW!)
        self.clip_model = None
        self.clip_processor = None
        if CLIP_AVAILABLE:
            try:
                from config import Config
                if Config.USE_CLIP_EMBEDDINGS:
                    print("Loading CLIP model for fast image processing...")
                    self.clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME)
                    self.clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        self.clip_model = self.clip_model.to("cuda")
                        print("✅ CLIP model loaded on GPU")
                    else:
                        print("✅ CLIP model loaded on CPU")
            except Exception as e:
                print(f"⚠️ Could not load CLIP model: {str(e)}")
                # Set to None to avoid errors later
                self.clip_model = None
                self.clip_processor = None
                
        if MULTIMODAL_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                from config import Config
                if Config.USE_GPT_VISION_CAPTIONS:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key.startswith("sk-"):
                        self.openai_client = OpenAI(api_key=api_key)
                        print("✅ GPT-4 Vision initialized for detailed captions")
                    else:
                        st.warning("⚠️ Invalid OPENAI_API_KEY format (should start with 'sk-')")
            except Exception as e:
                st.warning(f"⚠️ Could not initialize OpenAI client for vision: {str(e)}")
        
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
        """Load a single document with advanced preprocessing"""
        docs = []
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()
                # Extract tables and clean text
                docs = self._process_pdf_documents(raw_docs, file_path)
                
            elif file_extension == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                raw_docs = loader.load()
                docs = self._process_text_documents(raw_docs, file_path, "docx")
                
            elif file_extension == ".pptx":
                loader = UnstructuredPowerPointLoader(file_path)
                raw_docs = loader.load()
                docs = self._process_text_documents(raw_docs, file_path, "pptx")
                
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
                raw_docs = loader.load()
                docs = self._process_text_documents(raw_docs, file_path, "txt")
                
            elif file_extension == ".xlsx":
                docs = self._process_excel_with_tables(file_path)
                
            else:
                st.warning(f"Unsupported file type: {file_extension}")
            
            # Apply data quality validation
            docs = self._validate_and_filter_documents(docs)
            self.preprocessing_stats["total_docs"] += len(docs)
                
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
        """Split documents into chunks with deduplication"""
        if not docs:
            return []
        
        # Split documents
        chunks = self.splitter.split_documents(docs)
        
        # Remove duplicate chunks based on content similarity
        chunks = self._deduplicate_chunks(chunks)
        
        return chunks
    
    # ==================== Data Preprocessing Methods ====================
    
    def _clean_text(self, text: str) -> str:
        """Advanced text cleaning and normalization"""
        if not text:
            return ""
        
        original_length = len(text)
        
        # 1. Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # 2. Remove excessive whitespace while preserving structure
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'\t+', ' ', text)  # Tabs to single space
        
        # 3. Fix common OCR/extraction errors
        text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)  # Fix broken words
        
        # 4. Remove page numbers and headers/footers patterns
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        
        # 5. Clean special characters but preserve important punctuation
        text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'-]', '', text)
        
        # 6. Remove URLs and email addresses (optional, can preserve if needed)
        # text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
        # text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # 7. Strip leading/trailing whitespace
        text = text.strip()
        
        cleaned_chars = original_length - len(text)
        self.preprocessing_stats["cleaned_chars"] += cleaned_chars
        
        return text
    
    def _process_pdf_documents(self, docs: List[Document], file_path: str) -> List[Document]:
        """Process PDF documents with intelligent table and image extraction"""
        processed_docs = []
        total_pages = len(docs)
        
        # Check if multimodal processing is available and enabled
        from config import Config
        multimodal_enabled = (MULTIMODAL_AVAILABLE and Config.ENABLE_MULTIMODAL)
        use_clip = (Config.USE_CLIP_EMBEDDINGS and 
                   self.clip_model is not None and 
                   self.clip_processor is not None)
        use_gpt_vision = (Config.USE_GPT_VISION_CAPTIONS and 
                         self.openai_client is not None)
        
        if not multimodal_enabled and total_pages > 0:
            st.info("ℹ️ Multimodal processing disabled (set ENABLE_MULTIMODAL=True in config.py)")
        elif multimodal_enabled and not use_clip and not use_gpt_vision:
            st.warning("⚠️ Multimodal enabled but no model available (CLIP or GPT-4 Vision)")
        elif multimodal_enabled and use_clip:
            st.success("⚡ Fast CLIP-based image processing enabled (~50-200ms per image)")
        
        for i, doc in enumerate(docs):
            # Show progress for large PDFs
            if Config.SHOW_PROCESSING_PROGRESS and total_pages > 5 and i % 5 == 0:
                st.info(f"📄 Processing page {i+1}/{total_pages}...")
            
            # Clean text content
            cleaned_text = self._clean_text(doc.page_content)
            
            # Detect tables in content
            has_table, table_info = self._detect_table_in_text(cleaned_text)
            
            # Detect images (placeholder detection based on common patterns)
            has_images = self._detect_images_in_text(cleaned_text)
            
            page_num = doc.metadata.get("page", i)
            
            # Enhanced metadata
            metadata = {
                "source": os.path.basename(file_path),
                "type": "pdf",
                "page": page_num,
                "has_table": has_table,
                "has_images": has_images,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split())
            }
            
            # ===== INTELLIGENT IMAGE EXTRACTION & CAPTIONING =====
            image_content = ""
            if multimodal_enabled:
                try:
                    # Extract images from this specific page
                    images_data = self._extract_images_from_pdf(file_path, page_num)
                    
                    if images_data:
                        # Limit images per page for performance
                        max_images = getattr(Config, 'MAX_IMAGES_PER_PAGE', 5)
                        if len(images_data) > max_images:
                            st.warning(f"⚠️ Page {page_num+1} has {len(images_data)} images. Processing first {max_images} only.")
                            images_data = images_data[:max_images]
                        
                        metadata["has_images"] = True
                        metadata["image_count"] = len(images_data)
                        image_captions = []
                        
                        # Show progress for image processing
                        if Config.SHOW_PROCESSING_PROGRESS and len(images_data) > 0:
                            if use_clip:
                                st.info(f"⚡ Processing {len(images_data)} images on page {page_num+1} with CLIP (fast!)...")
                            elif use_gpt_vision:
                                st.info(f"🖼️ Processing {len(images_data)} images on page {page_num+1} with GPT-4 Vision (slow)...")
                        
                        for img_idx, img_data in enumerate(images_data):
                            try:
                                context = cleaned_text[:500]  # Surrounding text as context
                                
                                # Option 1: Fast CLIP embeddings (preferred)
                                if use_clip:
                                    # Decode base64 to bytes for CLIP
                                    image_bytes = base64.b64decode(img_data["base64"])
                                    clip_embedding = self._generate_clip_embedding(image_bytes, context)
                                    
                                    # Generate simple description (no API call)
                                    caption = f"Image {img_data['index']+1}: {img_data['format']}, {img_data['width']}x{img_data['height']}px"
                                    if context:
                                        caption += f". Context: {context[:100]}..."
                                    
                                    # For unified CLIP: create Document objects that will be added to main vector store
                                    if Config.USE_UNIFIED_CLIP_EMBEDDINGS:
                                        # Create image document for vector store
                                        image_doc = Document(
                                            page_content=f"[IMAGE] {caption}. Surrounding text: {context[:200]}",
                                            metadata={
                                                "source": os.path.basename(file_path),
                                                "type": "image",
                                                "page": img_data['page'],
                                                "image_index": img_data['index'],
                                                "format": img_data['format'],
                                                "width": img_data['width'],
                                                "height": img_data['height'],
                                                "is_image": True,
                                                "has_clip_embedding": True
                                            }
                                        )
                                        # Add to processed docs (will be embedded with CLIP and added to vector store)
                                        processed_docs.append(image_doc)
                                
                                # Option 2: Detailed GPT-4 Vision captions (optional, slow)
                                elif use_gpt_vision:
                                    caption = self._generate_image_caption(img_data["base64"], context)
                                    clip_embedding = None
                                
                                # Option 3: Fallback - simple text description
                                else:
                                    caption = self._generate_text_for_image(img_data, context)
                                    clip_embedding = None
                                
                                image_captions.append(
                                    f"[IMAGE {img_data['index']+1} on page {img_data['page']}]: {caption}"
                                )
                                
                                # Store image data for multimodal embedding
                                image_doc_metadata = {
                                    "source": os.path.basename(file_path),
                                    "type": "image",
                                    "page": img_data['page'],
                                    "image_index": img_data['index'],
                                    "format": img_data['format'],
                                    "width": img_data['width'],
                                    "height": img_data['height'],
                                    "parent_page": page_num,
                                    "is_multimodal": True,
                                    "has_clip_embedding": clip_embedding is not None
                                }
                                
                                self.image_documents.append({
                                    "base64": img_data["base64"],
                                    "caption": caption,
                                    "clip_embedding": clip_embedding,  # Store CLIP embedding
                                    "metadata": image_doc_metadata,
                                    "context": cleaned_text[:500]
                                })
                                self.preprocessing_stats["multimodal_embeddings"] += 1
                                
                            except Exception as img_error:
                                st.warning(f"⚠️ Could not process image {img_idx+1} on page {page_num+1}: {str(img_error)}")
                                continue
                        
                        # Append image descriptions to content for better retrieval
                        if image_captions:
                            image_content = "\n\n--- IMAGES IN THIS PAGE ---\n" + "\n\n".join(image_captions)
                
                except Exception as e:
                    st.warning(f"⚠️ Could not process images on page {page_num+1}: {str(e)}")
                    import traceback
                    st.error(f"Debug: {traceback.format_exc()[:500]}")
            
            # Combine text + image captions
            final_content = cleaned_text + image_content
            
            if has_table:
                metadata["table_info"] = table_info
                self.preprocessing_stats["tables_extracted"] += 1
            
            if has_images:
                self.preprocessing_stats["images_found"] += 1
            
            processed_docs.append(Document(
                page_content=final_content,
                metadata=metadata
            ))
        
        return processed_docs
    
    def _process_text_documents(self, docs: List[Document], file_path: str, doc_type: str) -> List[Document]:
        """Process text-based documents with cleaning"""
        processed_docs = []
        
        for doc in docs:
            cleaned_text = self._clean_text(doc.page_content)
            
            metadata = {
                "source": os.path.basename(file_path),
                "type": doc_type,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split())
            }
            
            processed_docs.append(Document(
                page_content=cleaned_text,
                metadata=metadata
            ))
        
        return processed_docs
    
    def _process_excel_with_tables(self, file_path: str) -> List[Document]:
        """Process Excel files with intelligent table summarization"""
        docs = []
        
        try:
            # Read all sheets
            xls = pd.ExcelFile(file_path)
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Convert to structured text format
                table_text = self._dataframe_to_text(df, sheet_name)
                
                metadata = {
                    "source": os.path.basename(file_path),
                    "type": "excel",
                    "sheet": sheet_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "has_table": True,
                    "table_columns": list(df.columns)
                }
                
                # Store complete raw table (no summarization)
                docs.append(Document(
                    page_content=table_text,
                    metadata=metadata
                ))
                
                self.preprocessing_stats["tables_extracted"] += 1
        
        except Exception as e:
            st.error(f"Error processing Excel file: {str(e)}")
        
        return docs
    
    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Convert DataFrame to well-structured text for embedding"""
        text_parts = [f"Table: {sheet_name}\n"]
        
        # Add column headers
        text_parts.append("Columns: " + ", ".join(df.columns.tolist()))
        text_parts.append("\n")
        
        # Add data rows in a readable format
        for idx, row in df.iterrows():
            row_text = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value):  # Skip NaN values
                    row_text.append(f"{col}: {value}")
            
            if row_text:
                text_parts.append("; ".join(row_text))
        
        return "\n".join(text_parts)
    
    def _detect_table_in_text(self, text: str) -> Tuple[bool, str]:
        """Detect if text contains table-like structures"""
        # Look for patterns indicating tables
        table_indicators = [
            r'\|[\s\w]+\|',  # Pipe-separated tables
            r'\t[\w\s]+\t',  # Tab-separated
            r'(?:\w+\s+){3,}(?:\d+\s+){2,}',  # Multiple columns of data
        ]
        
        for pattern in table_indicators:
            matches = re.findall(pattern, text)
            if len(matches) >= 3:  # At least 3 rows that look like table data
                return True, f"Detected {len(matches)} table rows"
        
        return False, ""
    
    def _detect_images_in_text(self, text: str) -> bool:
        """Detect references to images in extracted text"""
        image_indicators = [
            r'\[image:\s*\w+\]',
            r'\[fig(?:ure)?\s*\d+\]',
            r'see\s+(?:image|figure|diagram)',
            r'\[IMAGE\]',
            r'\[CHART\]'
        ]
        
        for pattern in image_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    # ==================== Multimodal RAG Methods ====================
    
    def _extract_images_from_pdf(self, pdf_path: str, page_num: int = None) -> List[Dict]:
        """
        Extract images from PDF using PyMuPDF (fitz)
        Returns list of image data with metadata
        """
        if not MULTIMODAL_AVAILABLE:
            return []
        
        images_data = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            pages_to_process = [page_num] if page_num is not None else range(len(pdf_document))
            
            for page_idx in pages_to_process:
                page = pdf_document[page_idx]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Skip very small images (likely logos/icons)
                        from config import Config as CfgCheck
                        min_size = getattr(CfgCheck, 'MIN_IMAGE_SIZE', 150)
                        if image.width < min_size or image.height < min_size:
                            continue
                        
                        # Convert to base64 for GPT-4 Vision
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        images_data.append({
                            "page": page_idx + 1,
                            "index": img_index,
                            "format": image_ext,
                            "width": image.width,
                            "height": image.height,
                            "base64": img_str,
                            "size_kb": len(image_bytes) / 1024
                        })
                        
                        self.preprocessing_stats["images_extracted"] += 1
            
            pdf_document.close()
            
        except Exception as e:
            st.warning(f"⚠️ Error extracting images from PDF: {str(e)}")
        
        return images_data
    
    def _generate_image_caption(self, image_base64: str, context: str = "") -> str:
        """
        Generate descriptive caption for image using GPT-4 Vision
        
        Args:
            image_base64: Base64 encoded image
            context: Surrounding text context from the document
        
        Returns:
            Generated caption/description
        """
        if not self.openai_client or not MULTIMODAL_AVAILABLE:
            return "[Image - captioning unavailable]"
        
        try:
            prompt = """Analyze this image and provide a detailed description covering:
1. Main subject/content
2. Key elements, objects, or data shown
3. Any text visible in the image
4. Purpose or information conveyed

Keep the description concise but informative (2-4 sentences)."""

            if context:
                prompt += f"\n\nContext from document: {context[:200]}"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # GPT-4 with vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "low"  # 'low' = faster, 'high' = better quality
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,  # Reduced for faster responses
                temperature=0.3,
                timeout=15  # 15 second timeout to prevent hanging
            )
            
            caption = response.choices[0].message.content.strip()
            self.preprocessing_stats["images_captioned"] += 1
            
            return caption
            
        except Exception as e:
            st.warning(f"⚠️ Error generating image caption: {str(e)}")
            return "[Image - caption generation failed]"
    
    def _generate_clip_embedding(self, image_data: bytes, text_context: str = None) -> Optional[List[float]]:
        """
        Generate CLIP embedding for image (FAST: ~50-200ms per image)
        
        Args:
            image_data: Raw image bytes
            text_context: Optional text context to enhance embedding
        
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not self.clip_model or not self.clip_processor or not CLIP_AVAILABLE:
            return None
        
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to list
                embedding = image_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            print(f"⚠️ Error generating CLIP embedding: {str(e)}")
            return None
    
    def _generate_text_for_image(self, image_metadata: dict, context: str = "") -> str:
        """
        Generate descriptive text for image (for text-only embeddings if CLIP unavailable)
        
        Args:
            image_metadata: Image metadata with format, dimensions, etc.
            context: Surrounding text context
        
        Returns:
            Descriptive text string
        """
        description = f"Image: {image_metadata.get('format', 'Unknown')} format, "
        description += f"{image_metadata.get('width', 0)}x{image_metadata.get('height', 0)} pixels"
        
        if context:
            description += f". Context: {context[:200]}"
        
        return description
    
    def get_multimodal_image_documents(self) -> List[Dict]:
        """
        Get stored image documents for multimodal embedding
        
        Returns:
            List of dictionaries containing:
            - base64: Base64 encoded image
            - caption: GPT-4 Vision generated caption
            - metadata: Image metadata
            - context: Surrounding text context
        """
        return self.image_documents
    
    def clear_image_cache(self):
        """Clear cached image documents"""
        self.image_documents = []
        self.preprocessing_stats["multimodal_embeddings"] = 0
    
    def _validate_and_filter_documents(self, docs: List[Document]) -> List[Document]:
        """Validate document quality and filter out poor content"""
        valid_docs = []
        
        for doc in docs:
            # Quality checks
            text = doc.page_content
            
            # 1. Minimum content length
            if len(text.strip()) < 50:
                continue
            
            # 2. Check for meaningful content (not just special characters)
            word_count = len(re.findall(r'\w+', text))
            if word_count < 10:
                continue
            
            # 3. Check content is not just numbers or gibberish
            alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
            if alpha_ratio < 0.3:  # At least 30% alphabetic characters
                continue
            
            valid_docs.append(doc)
        
        return valid_docs
    
    def _deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove duplicate or highly similar chunks"""
        if not chunks:
            return chunks
        
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            # Create a normalized hash of content
            normalized = re.sub(r'\s+', ' ', chunk.page_content.lower().strip())
            content_hash = hash(normalized)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
            else:
                self.preprocessing_stats["duplicates_removed"] += 1
        
        return unique_chunks
    
    def get_preprocessing_stats(self) -> Dict:
        """Return preprocessing statistics"""
        return self.preprocessing_stats

class VectorStoreManager:
    """Handle vector store operations"""
    
    def __init__(self, doc_processor=None):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.doc_processor = doc_processor  # Link to document processor for image access
        self._initialize_embeddings()
    
    def set_document_processor(self, doc_processor):
        """Link document processor for image retrieval"""
        self.doc_processor = doc_processor
    
    def _initialize_embeddings(self):
        """Initialize embeddings - either unified CLIP or OpenAI text embeddings"""
        try:
            from config import Config
            
            # Option 1: Unified CLIP embeddings for both text and images (RECOMMENDED)
            if (Config.USE_UNIFIED_CLIP_EMBEDDINGS and 
                self.doc_processor and 
                self.doc_processor.clip_model is not None):
                print("✅ Using unified CLIP embeddings for text AND images")
                self.embeddings = CLIPEmbeddings(self.doc_processor.clip_model, self.doc_processor.clip_processor)
                return True
            
            # Option 2: Traditional OpenAI text embeddings (text only) 
            print("📝 Using OpenAI text embeddings (text only)")
            
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
                # Store reference to document processor for image retrieval
                if self.doc_processor:
                    self.vectorstore._source_processor = self.doc_processor
                    
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",  # Changed from MMR to cosine similarity
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

def display_document_stats(docs: List[Document], chunks: List[Document], processor=None):
    """Display comprehensive document and preprocessing statistics"""
    if docs:
        st.subheader("📊 Document Processing Statistics")
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Documents Loaded", len(docs))
        with col2:
            st.metric("✂️ Text Chunks Created", len(chunks))
        with col3:
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            st.metric("🔤 Total Characters", f"{total_chars:,}")
        
        # Data Engineering Stats
        if processor and hasattr(processor, 'get_preprocessing_stats'):
            stats = processor.get_preprocessing_stats()
            
            st.divider()
            st.subheader("🛠️ Data Preprocessing & Engineering")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🧹 Cleaned Characters", f"{stats.get('cleaned_chars', 0):,}")
            with col2:
                st.metric("📋 Tables Extracted", stats.get('tables_extracted', 0))
            with col3:
                st.metric("🖼️ Images Detected", stats.get('images_found', 0))
            with col4:
                st.metric("🔄 Duplicates Removed", stats.get('duplicates_removed', 0))
            
            # Multimodal Intelligence Stats
            if stats.get('images_extracted', 0) > 0 or stats.get('images_captioned', 0) > 0 or stats.get('multimodal_embeddings', 0) > 0:
                st.divider()
                st.subheader("🤖 Multimodal AI Processing")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📸 Images Extracted", stats.get('images_extracted', 0))
                with col2:
                    st.metric("🏷️ Images Captioned (GPT-4V)", stats.get('images_captioned', 0))
                with col3:
                    st.metric("🎨 Multimodal Embeddings", stats.get('multimodal_embeddings', 0))
        
        # Document Quality Metrics
        st.divider()
        st.subheader("✅ Data Quality Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            doc_types = {}
            for doc in docs:
                doc_type = doc.metadata.get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            st.write("**Document Types:**")
            for dtype, count in doc_types.items():
                st.write(f"- {dtype}: {count}")
        
        with col2:
            avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
            st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars")
            
            total_words = sum(len(chunk.page_content.split()) for chunk in chunks)
            st.metric("Total Words", f"{total_words:,}")
        
        with col3:
            tables_count = sum(1 for doc in docs if doc.metadata.get('has_table', False))
            images_count = sum(1 for doc in docs if doc.metadata.get('has_images', False))
            st.metric("Docs with Tables", tables_count)
            st.metric("Docs with Images", images_count)