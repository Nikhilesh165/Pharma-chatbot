"""
Vector store implementation for HPL pharmaceutical data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HPLVectorStore:
    """Vector store for HPL pharmaceutical data."""
    
    def __init__(self, processed_data_dir: str, embeddings_dir: str):
        """
        Initialize the vector store.
        
        Args:
            processed_data_dir: Directory containing processed JSON files
            embeddings_dir: Directory to save FAISS index
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        self.vector_store = None
    
    def load_documents(self) -> List[Document]:
        """Load and prepare documents from processed data."""
        documents = []
        
        # Process each JSON file in the processed data directory
        for file_path in self.processed_data_dir.glob('*.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract product information
            product_name = data.get('name', '')
            product_url = data.get('url', '')
            
            # Process prescribing information sections
            if 'prescribing_info' in data and 'sections' in data['prescribing_info']:
                for section_name, content in data['prescribing_info']['sections'].items():
                    if content:
                        # Create metadata for the section
                        metadata = {
                            'product_name': product_name,
                            'product_url': product_url,
                            'section': section_name,
                            'source': str(file_path)
                        }
                        
                        # Add PDF URL if available
                        if 'pdf_url' in data['prescribing_info']:
                            metadata['pdf_url'] = data['prescribing_info']['pdf_url']
                        
                        # Create document
                        doc = Document(
                            page_content=f"{section_name}: {content}",
                            metadata=metadata
                        )
                        documents.append(doc)
        
        return documents
    
    def build(self) -> None:
        """Build the vector store from processed documents."""
        logging.info("Loading documents...")
        documents = self.load_documents()
        
        logging.info("Splitting documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        
        logging.info("Creating vector store...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # Save the vector store
        self.save()
        
        logging.info("Vector store built successfully")
    
    def save(self) -> None:
        """Save the vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(str(self.embeddings_dir))
            logging.info(f"Vector store saved to {self.embeddings_dir}")
    
    def load(self) -> bool:
        """Load the vector store from disk."""
        index_path = self.embeddings_dir / "index.faiss"
        if index_path.exists():
            self.vector_store = FAISS.load_local(
                str(self.embeddings_dir),
                self.embeddings,
                allow_dangerous_deserialization=True  # We trust our own files
            )
            logging.info("Vector store loaded successfully")
            return True
        return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            if not self.load():
                raise ValueError("Vector store not initialized. Run build() first.")
        
        return self.vector_store.similarity_search(query, k=k)
