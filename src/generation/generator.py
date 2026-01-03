"""
Generation module - handles LLM-based answer generation
"""
from typing import List, Dict, Optional, Iterator
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class AnswerGenerator:
    """Generates answers using LLM with retrieved context"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: Optional[int] = None):
        """
        Initialize generator

        Args:
            model: OpenAI model to use for generation
            max_tokens: Default maximum tokens for responses (None = use method default)
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model
        self.default_max_tokens = max_tokens
        
        self.system_prompt = """You are a helpful banking customer support assistant.

Your role:
- Answer user questions based on the provided context documents
- Be friendly, professional, and concise
- Use information from the context whenever possible
- If the context doesn't contain relevant information, politely say so
- Keep answers to 2-4 sentences unless more detail is needed
- Don't make up information or policies

Guidelines:
- Start directly with the answer (no "Based on the context..." preambles)
- Use natural, conversational language
- If multiple documents are relevant, synthesize information
- For urgent issues (card blocked, fraud), emphasize contacting support
"""
    
    def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate answer based on query and context
        
        Args:
            query: User's question
            context: Retrieved context from knowledge base
            conversation_history: Previous messages for multi-turn
            temperature: Sampling temperature
            max_tokens: Maximum response length
            
        Returns:
            Generated answer string
        """
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if exists (keep last 6 messages = 3 turns)
        if conversation_history:
            messages.extend(conversation_history[-6:])
        
        # Add current query with context
        user_message = f"""Context from knowledge base:
{context}

User question: {query}

Please provide a helpful answer based on the context above."""
        
        messages.append({"role": "user", "content": user_message})

        # Use default max_tokens if not specified, fall back to 300
        tokens = max_tokens if max_tokens is not None else (self.default_max_tokens if self.default_max_tokens is not None else 300)

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=tokens
        )
        
        return response.choices[0].message.content

    def generate_stream(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """
        Generate answer with streaming - yields chunks as they arrive

        Args:
            query: User's question
            context: Retrieved context from knowledge base
            conversation_history: Previous messages for multi-turn
            temperature: Sampling temperature
            max_tokens: Maximum response length

        Yields:
            Chunks of the generated answer as they arrive
        """
        # Build messages (same as generate)
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history if exists (keep last 6 messages = 3 turns)
        if conversation_history:
            messages.extend(conversation_history[-6:])

        # Add current query with context
        user_message = f"""Context from knowledge base:
{context}

User question: {query}

Please provide a helpful answer based on the context above."""

        messages.append({"role": "user", "content": user_message})

        # Use default max_tokens if not specified, fall back to 300
        tokens = max_tokens if max_tokens is not None else (self.default_max_tokens if self.default_max_tokens is not None else 300)

        # Generate streaming response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=tokens,
            stream=True
        )

        # Yield chunks as they arrive
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
