"""
Retrieval module - handles vector search and document retrieval with Cross-Encoder reranking
"""
from typing import List, Dict, Optional
import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class KnowledgeBaseRetriever:
    """Handles retrieval from ChromaDB vector database with Cross-Encoder reranking"""

    def __init__(self, vector_db_path: str, collection_name: str = "banking_support",
                 use_reranking: bool = False, rerank_multiplier: int = 2):
        """
        Initialize retriever with existing knowledge base

        Args:
            vector_db_path: Path to ChromaDB storage
            collection_name: Name of the collection
            use_reranking: Whether to use Cross-Encoder reranking for better accuracy
            rerank_multiplier: How many extra candidates to fetch for reranking (default: 2)
        """
        self.client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-small"
        self.use_reranking = use_reranking
        self.rerank_multiplier = rerank_multiplier

        # Initialize Cross-Encoder for reranking
        # Using ms-marco-TinyBERT - faster and often better for short queries
        if use_reranking:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
        else:
            self.reranker = None
        
    def get_embedding(self, text: str) -> List[float]:
        """Create embedding for query text"""
        response = self.openai_client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str, n_results: int = 3) -> Dict:
        """
        Retrieve relevant documents for query with optional Cross-Encoder reranking

        Args:
            query: User's question
            n_results: Number of documents to retrieve

        Returns:
            Dictionary with documents, metadata, distances, and reranking scores
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Retrieve more candidates if reranking is enabled (over-fetch for better reranking)
        initial_n = n_results * self.rerank_multiplier if self.use_reranking else n_results

        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_n
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Apply Cross-Encoder reranking if enabled
        if self.use_reranking and self.reranker and len(documents) > 0:
            # Create query-document pairs for reranking
            pairs = [[query, doc] for doc in documents]

            # Get reranking scores (higher is better)
            rerank_scores = self.reranker.predict(pairs)

            # Sort by reranking scores (descending)
            ranked_indices = sorted(range(len(rerank_scores)),
                                   key=lambda i: rerank_scores[i],
                                   reverse=True)

            # Reorder results based on reranking scores
            documents = [documents[i] for i in ranked_indices[:n_results]]
            metadatas = [metadatas[i] for i in ranked_indices[:n_results]]
            distances = [distances[i] for i in ranked_indices[:n_results]]
            rerank_scores = [rerank_scores[i] for i in ranked_indices[:n_results]]

            return {
                'documents': documents,
                'metadatas': metadatas,
                'distances': distances,
                'rerank_scores': rerank_scores,
                'query': query,
                'reranked': True
            }
        else:
            return {
                'documents': documents[:n_results],
                'metadatas': metadatas[:n_results],
                'distances': distances[:n_results],
                'query': query,
                'reranked': False
            }
    
    def format_context(self, results: Dict) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        )):
            similarity = 1 - dist
            context_parts.append(
                f"[Document {i+1}] (Relevance: {similarity:.2f})\n"
                f"Category: {meta['category']}\n"
                f"{doc}\n"
            )
        
        return "\n".join(context_parts)