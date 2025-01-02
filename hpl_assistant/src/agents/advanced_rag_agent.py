"""
Advanced RAG agent with enhanced retrieval, prompting, and reasoning capabilities.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.runnable import RunnableParallel
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import numpy as np

# Add src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from embeddings.vectorstore import HPLVectorStore

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries that can be handled."""
    FACTUAL = "factual"  # Simple factual queries
    COMPARISON = "comparison"  # Comparing multiple medications
    INTERACTION = "interaction"  # Drug interaction queries
    COMPLEX = "complex"  # Complex queries requiring multi-hop reasoning

class SourceDocument(BaseModel):
    """Model for source document with metadata."""
    content: str = Field(description="The content of the document")
    metadata: Dict[str, Any] = Field(description="Document metadata")
    relevance_score: float = Field(description="Relevance score of the document")

class QueryAnalysis(BaseModel):
    """Model for query analysis output."""
    query_type: QueryType = Field(description="Type of the query")
    sub_queries: List[str] = Field(description="List of sub-queries if query needs to be decomposed")
    requires_medical_context: bool = Field(description="Whether the query requires medical context")
    target_medications: List[str] = Field(description="List of medications mentioned in the query")
    key_aspects: List[str] = Field(description="Key aspects to focus on in the response")

class RetrievalResult(BaseModel):
    """Model for retrieval results."""
    documents: List[SourceDocument] = Field(description="Retrieved documents")
    strategy_used: str = Field(description="Retrieval strategy used")
    confidence_score: float = Field(description="Overall confidence in the retrieval results")

class AnswerMetadata(BaseModel):
    """Model for answer metadata."""
    confidence_score: float = Field(description="Confidence score for the answer")
    reasoning_path: List[str] = Field(description="Steps taken to arrive at the answer")
    sources_used: List[str] = Field(description="Sources used to generate the answer")
    medical_disclaimer: Optional[str] = Field(description="Medical disclaimer if applicable")

class Answer(BaseModel):
    """Model for the final answer."""
    answer_text: str = Field(description="The answer text")
    metadata: AnswerMetadata = Field(description="Metadata about the answer")

class AdvancedHPLRagAgent:
    """Advanced RAG agent with enhanced retrieval and reasoning capabilities."""
    
    def __init__(self, model_name: str = "llama2", num_documents: int = 4):
        """
        Initialize the advanced RAG agent.
        
        Args:
            model_name: Name of the Ollama model to use
            num_documents: Number of documents to retrieve and use for answering (default: 4)
        """
        self.model_name = model_name
        self.num_documents = num_documents
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components of the RAG pipeline."""
        try:
            # Initialize LLM
            logger.info("Initializing LLM...")
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.7,
                top_k=10,
                top_p=0.95,
                repeat_penalty=1.1
            )
            logger.info("LLM initialized successfully")
            
            # Initialize vector store
            logger.info("Initializing vector store...")
            base_dir = Path(__file__).parent.parent.parent
            processed_dir = base_dir / 'datasets' / 'processed'
            embeddings_dir = base_dir / 'embeddings'
            
            self.vector_store = HPLVectorStore(
                processed_data_dir=processed_dir,
                embeddings_dir=embeddings_dir
            )
            
            # Load or build vector store
            if not self.vector_store.load():
                logger.info("Building vector store...")
                self.vector_store.build()
            
            logger.info("Initializing retrievers...")
            self._initialize_retrievers()
            logger.info("Retrievers initialized successfully")
            
            logger.info("Initializing prompts...")
            self._initialize_prompts()
            logger.info("Prompts initialized successfully")
            
            logger.info("Initializing chains...")
            self._initialize_chains()
            logger.info("Chains initialized successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise
    
    def _initialize_retrievers(self):
        """Initialize various retrieval components."""
        # Dense retriever (FAISS)
        self.dense_retriever = self.vector_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.num_documents}
        )
        
        # BM25 retriever for keyword search
        all_docs = self.vector_store.load_documents()
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        
        # Ensemble retriever combining dense and sparse
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        # Multi-query retriever with reduced number of queries
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.ensemble_retriever,
            llm=self.llm,
            parser_key="queries",  # Only generate alternative queries
            num_queries=2  # Reduce number of query variations
        )
    
    def _rerank_documents(self, docs: List[Document], query: str) -> List[SourceDocument]:
        """Rerank documents based on relevance to query."""
        if not docs:
            return []
            
        # Batch documents for scoring to reduce API calls
        batch_size = 3  # Score 3 documents per LLM call
        batched_docs = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
        
        reranked_docs = []
        for batch in batched_docs:
            # Create batch prompt
            batch_prompt = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(batch)
            ])
            
            # Get relevance scores for batch
            score_response = self.llm.invoke(f"""
            Query: {query}
            
            Rate the relevance of each document to the query on a scale of 0-1:
            
            {batch_prompt}
            
            Output the scores as a comma-separated list of numbers between 0 and 1:
            """)
            
            try:
                # Parse scores
                scores = [float(s.strip()) for s in score_response.content.strip().split(",")]
                if len(scores) != len(batch):
                    scores = [0.5] * len(batch)  # Default if parsing fails
            except:
                scores = [0.5] * len(batch)  # Default if parsing fails
            
            # Create SourceDocuments with scores
            for doc, score in zip(batch, scores):
                reranked_docs.append(
                    SourceDocument(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        relevance_score=score
                    )
                )
        
        # Sort by relevance score
        reranked_docs.sort(key=lambda x: x.relevance_score, reverse=True)
        return reranked_docs
    
    def _format_context(self, docs: List[SourceDocument]) -> str:
        """Format documents for prompt context."""
        return "\n\n".join(
            f"Source ({doc.metadata['product_name']}, {doc.metadata['section']}, "
            f"Relevance: {doc.relevance_score:.2f}):\n{doc.content}"
            for doc in docs
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about pharmaceutical products using advanced RAG techniques.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and detailed metadata
        """
        try:
            # Retrieve documents using ensemble retrieval
            docs = self.multi_query_retriever.get_relevant_documents(question)
            
            # Rerank documents
            reranked_docs = self._rerank_documents(docs, question)
            
            # Format context
            context = self._format_context(reranked_docs[:self.num_documents])
            
            # Generate answer with a single LLM call
            response = self.llm.invoke(f"""
            Question: {question}
            
            Retrieved Information:
            {context}
            
            Provide a comprehensive answer with:
            1. Clear, direct response
            2. Supporting evidence from sources
            3. Any necessary medical disclaimers
            4. Confidence level in the answer (0-100%)
            5. Key reasoning steps taken
            
            Format your response as a JSON object with these fields:
            - answer_text: The main answer
            - confidence_score: Confidence as a decimal (0-1)
            - reasoning_path: List of reasoning steps
            - sources_used: List of sources used
            - medical_disclaimer: Any medical disclaimer (or null)
            """)
            
            # Parse the response
            try:
                result = eval(response.content)  # Basic parsing, could be improved
                return {
                    "answer": result["answer_text"],
                    "confidence": result["confidence_score"],
                    "reasoning": result["reasoning_path"],
                    "sources": result["sources_used"],
                    "disclaimer": result["medical_disclaimer"],
                    "query_type": "factual"  # Simplified
                }
            except:
                return {
                    "answer": response.content,
                    "confidence": 0.7,  # Default confidence
                    "reasoning": [],
                    "sources": [f"Source {i+1}" for i in range(len(reranked_docs))],
                    "disclaimer": "This information is for reference only. Please consult a healthcare professional.",
                    "query_type": "factual"
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "confidence": 0.0,
                "reasoning": [],
                "sources": [],
                "disclaimer": None,
                "query_type": None
            }
