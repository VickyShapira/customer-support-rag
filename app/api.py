"""
FastAPI REST API for the chatbot
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
from typing import Optional, List, Dict
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asyncio
import threading

# Load environment
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from rag_pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Banking Support RAG API",
    description="RAG-powered customer support chatbot API with 3-layer retrieval (96% accuracy)",
    version="1.0.0"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared RAG pipeline (single instance for all sessions)
vector_db_path = str(Path(__file__).parent.parent / 'data' / 'vector_db')
rag = RAGPipeline(
    vector_db_path,
    use_contextual_retriever=True,
    use_smart_retriever=True
)

# Thread lock for RAG pipeline access (prevents race conditions)
rag_lock = threading.Lock()

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 3
    include_sources: Optional[bool] = True
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[Dict]] = None

# Session management (in-memory with TTL)
# Each session stores: {conversation_history: [...], last_accessed: datetime}
sessions = {}
sessions_lock = threading.Lock()  # Protect session dictionary access
SESSION_TTL_MINUTES = 30  # Sessions expire after 30 minutes of inactivity
MAX_SESSIONS = 1000  # Maximum concurrent sessions to prevent memory exhaustion

def cleanup_expired_sessions():
    """Remove sessions that haven't been accessed in SESSION_TTL_MINUTES"""
    with sessions_lock:
        now = datetime.now()
        expired = [
            sid for sid, session in sessions.items()
            if now - session['last_accessed'] > timedelta(minutes=SESSION_TTL_MINUTES)
        ]
        for sid in expired:
            del sessions[sid]
        return len(expired)

def enforce_session_limit():
    """Remove oldest sessions if MAX_SESSIONS exceeded"""
    with sessions_lock:
        if len(sessions) >= MAX_SESSIONS:
            # Sort by last_accessed and remove oldest 10%
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: x[1]['last_accessed']
            )
            to_remove = len(sessions) - int(MAX_SESSIONS * 0.9)
            for sid, _ in sorted_sessions[:to_remove]:
                del sessions[sid]
            return to_remove
        return 0

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Banking Support RAG API",
        "version": "1.0.0",
        "features": {
            "accuracy": "96%",
            "layers": ["contextual", "smart", "base"],
            "streaming": False
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a user question through the RAG pipeline

    Args:
        request: Query request with question and parameters

    Returns:
        QueryResponse with answer and metadata
    """
    try:
        # Cleanup expired sessions periodically (every ~100 requests)
        if len(sessions) > 10:
            cleanup_expired_sessions()
            enforce_session_limit()

        # Get or create session (thread-safe)
        session_id = request.session_id or "default"
        with sessions_lock:
            if session_id not in sessions:
                sessions[session_id] = {
                    'conversation_history': [],
                    'last_accessed': datetime.now()
                }

            # Update last accessed time
            sessions[session_id]['last_accessed'] = datetime.now()

            # Get conversation history copy
            conversation_history = sessions[session_id]['conversation_history'].copy()

        # Process query with thread-safe RAG access
        with rag_lock:
            # Set conversation history in shared RAG instance
            rag.conversation_history = conversation_history

            # Process query
            result = rag.query(
                question=request.question,
                n_results=request.n_results,
                include_sources=request.include_sources
            )

            # Get updated conversation history
            updated_history = rag.conversation_history.copy()

        # Save updated conversation history back to session (thread-safe)
        with sessions_lock:
            if session_id in sessions:  # Session might have been cleaned up
                sessions[session_id]['conversation_history'] = updated_history

        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_conversation(session_id: str = "default"):
    """Reset conversation history for a session"""
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id]['conversation_history'] = []
            sessions[session_id]['last_accessed'] = datetime.now()
            return {"status": "success", "message": "Conversation reset"}
        return {"status": "success", "message": "No active session"}

@app.get("/history")
async def get_history(session_id: str = "default"):
    """Get conversation history for a session"""
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id]['last_accessed'] = datetime.now()
            return {
                "session_id": session_id,
                "history": sessions[session_id]['conversation_history'].copy(),
                "message_count": len(sessions[session_id]['conversation_history'])
            }
        return {"session_id": session_id, "history": [], "message_count": 0}

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    # Cleanup before reporting stats
    cleaned = cleanup_expired_sessions()

    with sessions_lock:
        active_count = len(sessions)
        total_messages = sum(
            len(session['conversation_history'])
            for session in sessions.values()
        )

    # Get pipeline info
    pipeline_info = rag.get_pipeline_info()

    return {
        "active_sessions": active_count,
        "total_messages": total_messages,
        "expired_sessions_cleaned": cleaned,
        "session_ttl_minutes": SESSION_TTL_MINUTES,
        "max_sessions": MAX_SESSIONS,
        "vector_db_path": vector_db_path,
        "pipeline_config": {
            "contextual_retriever": pipeline_info['layers']['contextual_retriever'],
            "smart_retriever": pipeline_info['layers']['smart_retriever'],
            "model": pipeline_info['model'],
            "accuracy": "96%",
            "thresholds": pipeline_info['thresholds']
        }
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Manually delete a specific session"""
    with sessions_lock:
        if session_id in sessions:
            del sessions[session_id]
            return {"status": "success", "message": f"Session {session_id} deleted"}
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions():
    """List all active sessions with metadata"""
    with sessions_lock:
        session_list = [
            {
                "session_id": sid,
                "message_count": len(session['conversation_history']),
                "last_accessed": session['last_accessed'].isoformat(),
                "age_minutes": (datetime.now() - session['last_accessed']).total_seconds() / 60
            }
            for sid, session in sessions.items()
        ]
    return {
        "total_sessions": len(session_list),
        "sessions": sorted(session_list, key=lambda x: x['last_accessed'], reverse=True)
    }

# Run with: uvicorn app.api:app --reload
            