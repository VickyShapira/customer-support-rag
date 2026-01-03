"""
Contextual Retriever
Reformulates queries using conversation history for better multi-turn performance
"""

import openai
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class ContextualRetriever:
    """
    Retriever that reformulates queries with conversation context.
    
    Wraps the base KnowledgeBaseRetriever and adds conversation awareness
    by reformulating follow-up queries into standalone queries.
    """
    
    def __init__(self, base_retriever, use_context: bool = True, model: str = "gpt-4o-mini"):
        """
        Initialize contextual retriever
        
        Args:
            base_retriever: KnowledgeBaseRetriever instance
            use_context: Whether to use conversation context (can disable for testing)
            model: OpenAI model to use for query reformulation
        """
        self.base_retriever = base_retriever
        self.use_context = use_context
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def retrieve(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        n_results: int = 5
    ):
        """
        Retrieve documents with conversation context
        
        Args:
            query: Current user query
            conversation_history: List of {"role": "user"/"assistant", "content": "..."}
            n_results: Number of results to return
            
        Returns:
            Same format as base_retriever.retrieve()
        """
        
        # If no context or context disabled, use base retrieval
        if not self.use_context or not conversation_history or len(conversation_history) == 0:
            return self.base_retriever.retrieve(query, n_results)
        
        # Check if query needs reformulation (is it a follow-up?)
        if self._needs_reformulation(query):
            # Reformulate query with context
            standalone_query = self._reformulate_with_context(query, conversation_history)
            print(f"[CONTEXT] Original: {query}")
            print(f"[CONTEXT] Reformulated: {standalone_query}")
        else:
            standalone_query = query
        
        # Retrieve with reformulated query
        return self.base_retriever.retrieve(standalone_query, n_results)
    
    def _needs_reformulation(self, query: str) -> bool:
        """
        Quick check if query looks like a follow-up that needs context
        
        Heuristics for follow-up queries:
        - Very short (< 5 words)
        - Starts with pronouns (it, that, this, they)
        - Starts with question words without clear subject (why, how come, what about)
        - Contains pronouns without antecedents (my card, the transfer - but no specific mention)
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Short queries likely need context
        if len(words) < 5:
            return True
        
        # Starts with follow-up indicators
        follow_up_starters = [
            'it ', 'that ', 'this ', 'they ', 'them ',
            'why would', 'how come', 'what about',
            'also', 'and ', 'but '
        ]
        if any(query_lower.startswith(starter) for starter in follow_up_starters):
            return True
        
        # Questions without clear nouns
        vague_questions = ['why?', 'how?', 'when?', 'really?', 'how so?']
        if query_lower in vague_questions:
            return True
        
        return False
    
    def _reformulate_with_context(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Use GPT-4o-mini to reformulate query into standalone form
        
        Args:
            query: Current follow-up query
            history: Conversation history
            
        Returns:
            Reformulated standalone query
        """
        
        # Build conversation context (last 6 messages = 3 turns)
        recent_history = history[-6:] if len(history) > 6 else history
        
        context_lines = []
        for msg in recent_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_lines.append(f"{role}: {msg['content']}")
        
        context = "\n".join(context_lines)
        
        prompt = f"""Given this conversation history and a follow-up query, rewrite the query to be self-contained and searchable.

Conversation History:
{context}

Follow-up Query: "{query}"

Your task: Rewrite this query to include all necessary context from the conversation. The rewritten query should be understandable WITHOUT reading the conversation history.

Rules:
1. Keep it concise (under 25 words)
2. Include the specific topic being discussed (e.g., "card payment", "bank transfer")
3. Preserve key details from context (amounts, locations, timeframes)
4. Don't add information not implied by the conversation
5. Make it a natural banking question

Examples:
- "Why would that happen?" → "Why would my card payment be declined?"
- "It was international" → "My declined card payment was an international transaction"
- "How long does it take?" → "How long does a bank transfer take to process?"

Standalone Query:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            
            reformulated = response.choices[0].message.content.strip()
            
            # Clean up formatting
            reformulated = reformulated.strip('"').strip("'").strip()
            
            # Fallback if reformulation is too similar or empty
            if not reformulated or reformulated.lower() == query.lower():
                return self._simple_context_concat(query, history)
            
            return reformulated
            
        except Exception as e:
            print(f"[WARNING] Query reformulation failed: {e}")
            # Fallback to simple concatenation
            return self._simple_context_concat(query, history)
    
    def _simple_context_concat(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Simple fallback: concatenate with last user query
        """
        last_user_query = next(
            (msg['content'] for msg in reversed(history) if msg['role'] == 'user'),
            ""
        )
        
        if last_user_query:
            return f"{last_user_query}. {query}"
        return query