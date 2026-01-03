"""
Complete RAG pipeline - combines retrieval and generation with multi-layer intelligence
"""
from typing import Dict, List, Optional, Iterator
from pathlib import Path
import sys

# Support both relative and absolute imports
try:
    from .retrieval.retriever import KnowledgeBaseRetriever
    from .retrieval.contextual_retriever import ContextualRetriever
    from .retrieval.smart_retriever import SmartRetriever
    from .generation.generator import AnswerGenerator
except ImportError:
    from retrieval.retriever import KnowledgeBaseRetriever
    from retrieval.contextual_retriever import ContextualRetriever
    from retrieval.smart_retriever import SmartRetriever
    from generation.generator import AnswerGenerator

class RAGPipeline:
    """
    Complete RAG pipeline with three-layer retrieval architecture:
    1. ContextualRetriever (outermost): Query reformulation for multi-turn conversations
    2. SmartRetriever (middle): LLM disambiguation for overlapping categories
    3. KnowledgeBaseRetriever (innermost): Semantic search with ChromaDB
    """

    def __init__(
        self, 
        vector_db_path: str, 
        model: str = "gpt-4o-mini", 
        max_tokens: Optional[int] = None, 
        use_contextual_retriever: bool = True,
        use_smart_retriever: bool = True,
        confidence_threshold: float = 0.38,  
        similarity_gap_threshold: float = 0.10
    ):
        """
        Initialize RAG pipeline with configurable retrieval layers

        Args:
            vector_db_path: Path to ChromaDB vector database
            model: OpenAI model for generation and retrieval
            max_tokens: Maximum tokens for response (None = unlimited, 200 for demos, 300+ for production)
            use_contextual_retriever: Enable conversation context and query reformulation (default: True)
            use_smart_retriever: Enable LLM disambiguation for ambiguous queries (default: True)
            confidence_threshold: Trigger LLM if top similarity < this (default: 0.38)
            similarity_gap_threshold: Trigger LLM if gap between top 2 < this (default: 0.10)
        """
        print("\n[*] Building RAG Pipeline...")

        # Layer 3 (innermost): Base semantic retriever
        base_retriever = KnowledgeBaseRetriever(vector_db_path)
        print(f"  [+] Base retriever initialized")

        # Layer 2 (middle): Smart retrieval with LLM disambiguation
        if use_smart_retriever:
            base_retriever = SmartRetriever(
                base_retriever=base_retriever,
                confidence_threshold=confidence_threshold,
                similarity_gap_threshold=similarity_gap_threshold
            )
            print(f"  [+] Smart retrieval enabled")
            print(f"     - Confidence threshold: {confidence_threshold}")
            print(f"     - Gap threshold: {similarity_gap_threshold}")
            print(f"     - Handles: overlapping categories (declined_card_payment vs card_not_working)")
            print(f"     - Expected trigger rate: ~20-30% of queries")

        # Layer 1 (outermost): Contextual retrieval for conversations
        if use_contextual_retriever:
            self.retriever = ContextualRetriever(
                base_retriever=base_retriever,
                use_context=True,
                model=model
            )
            print(f"  [+] Contextual retrieval enabled")
            print(f"     - Query reformulation for follow-ups")
            print(f"     - Conversation history tracking")
        else:
            self.retriever = base_retriever

        # Generator
        self.generator = AnswerGenerator(model, max_tokens=max_tokens)
        print(f"  [+] Generator initialized (model: {model})")

        # Configuration flags
        self.use_contextual_retriever = use_contextual_retriever
        self.use_smart_retriever = use_smart_retriever
        self.confidence_threshold = confidence_threshold
        self.similarity_gap_threshold = similarity_gap_threshold

        # Conversation state
        self.conversation_history = []

        print("[+] Pipeline ready!\n")
    
    def query(
        self,
        question: str,
        n_results: int = 3,
        include_sources: bool = True
    ) -> Dict:
        """
        Process a user question through the RAG pipeline

        Args:
            question: User's question
            n_results: Number of documents to retrieve
            include_sources: Whether to include source metadata

        Returns:
            Dictionary with answer and optional metadata
        """
        # Step 1: Retrieve relevant documents (goes through all enabled layers)
        if self.use_contextual_retriever:
            retrieval_results = self.retriever.retrieve(
                question, 
                conversation_history=self.conversation_history, 
                n_results=n_results
            )
        else:
            retrieval_results = self.retriever.retrieve(question, n_results)

        # Step 2: Format context from retrieved documents
        # Navigate to base retriever regardless of wrapper layers
        base_retriever = self._get_base_retriever()
        context = base_retriever.format_context(retrieval_results)

        # Step 3: Generate answer
        answer = self.generator.generate(
            question,
            context,
            self.conversation_history
        )

        # Build response
        response = {
            'question': question,
            'answer': answer
        }

        if include_sources:
            response['sources'] = [
                {
                    'category': meta['category'],
                    'similarity': 1 - dist
                }
                for meta, dist in zip(
                    retrieval_results['metadatas'],
                    retrieval_results['distances']
                )
            ]
            response['metadata'] = retrieval_results['metadatas']

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Keep only last 6 messages (3 turns) for context window management
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]

        return response
    
    def query_stream(
        self,
        question: str,
        n_results: int = 3,
        include_sources: bool = True
    ) -> Iterator[Dict]:
        """
        Process a user question through the RAG pipeline with streaming

        Args:
            question: User's question
            n_results: Number of documents to retrieve
            include_sources: Whether to include source metadata

        Yields:
            Dictionaries with chunks of the answer and optional metadata
            First yield contains metadata, subsequent yields contain answer chunks
        """
        # Step 1: Retrieve relevant documents (goes through all enabled layers)
        if self.use_contextual_retriever:
            retrieval_results = self.retriever.retrieve(
                question,
                conversation_history=self.conversation_history,
                n_results=n_results
            )
        else:
            retrieval_results = self.retriever.retrieve(question, n_results)

        # Step 2: Format context
        base_retriever = self._get_base_retriever()
        context = base_retriever.format_context(retrieval_results)

        # Yield initial metadata
        initial_response = {'question': question, 'chunk': '', 'done': False}

        if include_sources:
            initial_response['sources'] = [
                {
                    'category': meta['category'],
                    'similarity': 1 - dist
                }
                for meta, dist in zip(
                    retrieval_results['metadatas'],
                    retrieval_results['distances']
                )
            ]
            initial_response['metadata'] = retrieval_results['metadatas']

        yield initial_response

        # Step 3: Stream answer generation
        full_answer = ""
        for chunk in self.generator.generate_stream(
            question,
            context,
            self.conversation_history
        ):
            full_answer += chunk
            yield {
                'question': question,
                'chunk': chunk,
                'done': False
            }

        # Final yield to signal completion
        yield {
            'question': question,
            'chunk': '',
            'done': True,
            'full_answer': full_answer
        }

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": full_answer})

        # Keep only last 6 messages (3 turns) for context window management
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    def _get_base_retriever(self) -> KnowledgeBaseRetriever:
        """
        Navigate through wrapper layers to get base retriever
        
        Returns:
            The innermost KnowledgeBaseRetriever instance
        """
        retriever = self.retriever
        
        # Unwrap layers until we reach the base
        while hasattr(retriever, 'base_retriever'):
            retriever = retriever.base_retriever
        
        return retriever
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the current pipeline configuration
        
        Returns:
            Dictionary with pipeline configuration details
        """
        return {
            'layers': {
                'contextual_retriever': self.use_contextual_retriever,
                'smart_retriever': self.use_smart_retriever,
                'base_retriever': True
            },
            'thresholds': {
                'confidence_threshold': self.confidence_threshold,
                'similarity_gap_threshold': self.similarity_gap_threshold
            },
            'model': self.generator.model,
            'max_tokens': self.generator.max_tokens,
            'conversation_turns': len(self.conversation_history) // 2
        }