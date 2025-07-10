"""
RAG Retrieval System with Intent-based Document Routing
Supports Chroma and FAISS vector databases
"""

import logging
import os
from typing import Dict, List, Optional, Any
import asyncio

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import yaml
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    content: str
    source: str
    score: float
    intent: str


class DocumentChunk(BaseModel):
    content: str
    source: str
    intent: str
    metadata: Dict[str, Any] = {}


class RAGRetriever:
    def __init__(self, config_path: str = "app/config.yaml"):
        """Initialize RAG retriever with configuration"""
        self.config = self._load_config(config_path)
        self.embedding_model = None
        self.chroma_client = None
        self.collections = {}
        
        self._setup_embedding_model()
        self._setup_vector_db()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'rag': {
                'vector_db': 'chroma',
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'top_k': 5,
                'similarity_threshold': 0.7
            },
            'data_paths': {
                'technical_support': 'data/tech_docs',
                'billing_account': 'data/billing_docs',
                'feature_request': 'data/roadmap',
                'vector_db_path': 'embeddings/vector_dbs'
            }
        }
    
    def _setup_embedding_model(self):
        """Setup sentence transformer model for embeddings"""
        try:
            model_name = self.config['rag']['embedding_model']
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to setup embedding model: {e}")
    
    def _setup_vector_db(self):
        """Setup vector database (Chroma)"""
        try:
            db_path = self.config['data_paths']['vector_db_path']
            os.makedirs(db_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(allow_reset=True)
            )
            
            # Create collections for each intent
            intents = ['technical_support', 'billing_account', 'feature_request']
            for intent in intents:
                try:
                    collection = self.chroma_client.get_or_create_collection(
                        name=f"{intent}_docs",
                        metadata={"intent": intent}
                    )
                    self.collections[intent] = collection
                    logger.info(f"Setup Chroma collection for {intent}")
                except Exception as e:
                    logger.error(f"Failed to create collection for {intent}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split text into chunks for embedding"""
        if chunk_size is None:
            chunk_size = self.config['rag']['chunk_size']
        if chunk_overlap is None:
            chunk_overlap = self.config['rag']['chunk_overlap']
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def load_documents_from_directory(self, directory: str, intent: str) -> List[DocumentChunk]:
        """Load documents from a directory and convert to chunks"""
        documents = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return documents
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.txt', '.md', '.rst')):
                        file_path = os.path.join(root, file)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            chunks = self.chunk_text(content)
                            
                            for i, chunk in enumerate(chunks):
                                doc_chunk = DocumentChunk(
                                    content=chunk,
                                    source=f"{file}#chunk_{i}",
                                    intent=intent,
                                    metadata={
                                        'file_path': file_path,
                                        'chunk_index': i,
                                        'total_chunks': len(chunks)
                                    }
                                )
                                documents.append(doc_chunk)
                        
                        except Exception as e:
                            logger.error(f"Failed to load file {file_path}: {e}")
            
            logger.info(f"Loaded {len(documents)} document chunks for {intent}")
            
        except Exception as e:
            logger.error(f"Failed to load documents from {directory}: {e}")
        
        return documents
    
    def add_documents_to_chroma(self, documents: List[DocumentChunk], intent: str):
        """Add documents to ChromaDB collection"""
        if intent not in self.collections:
            logger.error(f"No collection found for intent: {intent}")
            return
        
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return
        
        try:
            collection = self.collections[intent]
            
            texts = [doc.content for doc in documents]
            ids = [f"{doc.source}_{i}" for i, doc in enumerate(documents)]
            metadatas = []
            
            for doc in documents:
                metadata = {
                    'source': doc.source,
                    'intent': doc.intent,
                    **doc.metadata
                }
                metadatas.append(metadata)
            
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_model.encode(texts).tolist()
            
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to {intent} collection")
            
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma: {e}")
    
    def build_vector_database(self):
        """Build vector database from all data directories"""
        logger.info("Building vector database...")
        
        for intent, directory in self.config['data_paths'].items():
            if intent == 'vector_db_path':
                continue
            
            logger.info(f"Processing {intent} documents from {directory}")
            documents = self.load_documents_from_directory(directory, intent)
            
            if documents:
                self.add_documents_to_chroma(documents, intent)
        
        logger.info("Vector database build complete")
    
    async def retrieve_documents(
        self, 
        query: str, 
        intent: str, 
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        if top_k is None:
            top_k = self.config['rag']['top_k']
        if similarity_threshold is None:
            similarity_threshold = self.config['rag']['similarity_threshold']
        
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return []
        
        if intent not in self.collections:
            logger.warning(f"No collection found for intent: {intent}")
            return []
        
        try:
            collection = self.collections[intent]
            
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            retrieval_results = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    similarity_score = 1.0 - distance
                    
                    if similarity_score >= similarity_threshold:
                        result = RetrievalResult(
                            content=doc,
                            source=metadata.get('source', 'unknown'),
                            score=similarity_score,
                            intent=intent
                        )
                        retrieval_results.append(result)
            
            logger.info(f"Retrieved {len(retrieval_results)} documents for {intent}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []


# Sample data creation function
def create_sample_documents():
    """Create sample documents for testing"""
    
    tech_docs = {
        "api_authentication.txt": """# API Authentication Guide

## Overview
Our API uses API keys for authentication. Each request must include your API key in the Authorization header.

## Getting Your API Key
1. Log into your dashboard
2. Navigate to Settings > API Keys
3. Click "Generate New Key"

## Using Your API Key
Include the API key in the Authorization header:
Authorization: Bearer your-api-key-here

## Common Issues
- 403 Forbidden: Invalid or expired API key
- 401 Unauthorized: Missing Authorization header
- 429 Too Many Requests: Rate limit exceeded""",

        "webhook_setup.txt": """# Webhook Configuration

## What are Webhooks?
Webhooks allow real-time notifications when events occur in your account.

## Setting Up Webhooks
1. Go to Settings > Webhooks
2. Click "Add Webhook"
3. Enter your endpoint URL
4. Select events to subscribe to

## Troubleshooting
- Ensure your endpoint returns 200 status
- Check SSL certificate validity
- Verify webhook signature"""
    }
    
    billing_docs = {
        "pricing_plans.txt": """# Pricing Plans

## Free Tier
- 100 API calls/month
- Basic support
- $0/month

## Pro Plan
- 10,000 API calls/month
- Priority support
- $29/month

## Enterprise Plan
- Unlimited API calls
- Dedicated support
- Custom pricing""",

        "refund_policy.txt": """# Refund Policy

## Refund Eligibility
- Requests within 30 days of payment
- Service issues not resolved
- Duplicate charges

## How to Request Refund
1. Contact support@company.com
2. Provide transaction details
3. Allow 5-7 business days processing"""
    }
    
    roadmap_docs = {
        "product_roadmap.txt": """# Product Roadmap 2024

## Q1 2024
- Multi-language support
- Advanced analytics dashboard
- Mobile SDK release

## Q2 2024
- Real-time collaboration
- Enhanced security features
- Custom model training

## Feature Request Process
1. Submit via support portal
2. Community voting
3. Engineering evaluation""",

        "feature_voting.txt": """# Feature Voting System

## Top Requested Features
1. Dark mode interface (2,450 votes)
2. Mobile app (2,180 votes)
3. Advanced filtering (1,920 votes)

## Voting Guidelines
- One vote per user per feature
- Must be logged in to vote
- Feature descriptions should be clear"""
    }
    
    # Create directories and files
    base_dirs = {
        'data/tech_docs': tech_docs,
        'data/billing_docs': billing_docs,
        'data/roadmap': roadmap_docs
    }
    
    for directory, documents in base_dirs.items():
        os.makedirs(directory, exist_ok=True)
        
        for filename, content in documents.items():
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    logger.info("Sample documents created successfully")


if __name__ == "__main__":
    async def test_retriever():
        """Test the RAG retriever"""
        print("Creating sample documents...")
        create_sample_documents()
        
        print("Initializing retriever...")
        retriever = RAGRetriever()
        
        print("Building vector database...")
        retriever.build_vector_database()
        
        test_queries = [
            ("Why is my API key not working?", "technical_support"),
            ("How much does the pro plan cost?", "billing_account"),
            ("Can you add dark mode?", "feature_request")
        ]
        
        print("\nTesting retrieval:")
        for query, intent in test_queries:
            print(f"\nQuery: {query} (Intent: {intent})")
            results = await retriever.retrieve_documents(query, intent, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f} - {result.source}")
                print(f"     {result.content[:100]}...")
    
    asyncio.run(test_retriever()) 