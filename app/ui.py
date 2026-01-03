"""
Streamlit web interface for the Banking Support RAG Chatbot
Optimized for portfolio demos with streaming responses and enhanced UX
"""
import streamlit as st
from pathlib import Path
import sys
import os
import hashlib
from dotenv import load_dotenv

# Load environment (override=True forces .env to take precedence over system env vars)
load_dotenv(override=True)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Banking Support AI Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact, professional design
st.markdown("""
<style>
    /* Reduce padding and margins */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .source-badge {
        background-color: #e1f5ff;
        color: #0277bd;
        padding: 0.15rem 0.5rem;
        border-radius: 0.8rem;
        font-size: 0.75rem;
        display: inline-block;
        margin: 0.15rem;
    }
    /* Compact metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    /* Reduce sidebar width slightly */
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 320px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG pipeline (cached with API key hash to detect changes)
@st.cache_resource
def load_rag_pipeline(_api_key_hash: str):
    """Load RAG pipeline with caching"""
    vector_db_path = str(Path(__file__).parent.parent / 'data' / 'vector_db')
    return RAGPipeline(
        vector_db_path,
        use_contextual_retriever=True,
        use_smart_retriever=True
    )

# Get API key hash to invalidate cache if key changes
api_key = os.getenv('OPENAI_API_KEY', '')
if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found in .env file. Please add it to continue.")
    st.stop()

api_key_hash = hashlib.md5(api_key.encode()).hexdigest()
rag = load_rag_pipeline(api_key_hash)

# Main header
st.markdown('<div class="main-header">🏦 Banking Support AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by 3-Layer RAG with Smart & Contextual Retrieval • 10,003 knowledge base entries • 96% accuracy</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Settings
    n_results = st.slider(
        "Retrieved Documents",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of knowledge base documents to retrieve for context"
    )

    show_sources = st.checkbox(
        "Show Source Documents",
        value=True,
        help="Display which knowledge base entries were used"
    )

    use_streaming = st.checkbox(
        "Enable Streaming",
        value=True,
        help="Stream responses in real-time for better UX"
    )

    st.divider()

    # Stats
    st.header("📊 System Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Knowledge Base", "10,003", help="Total training examples")
        st.metric("Categories", "77", help="Banking support topics")
    with col2:
        st.metric("Accuracy", "96%", help="Overall accuracy across all query types")
        st.metric("Streaming", "Enabled ⚡", help="Real-time response streaming")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.messages = []
        rag.reset_conversation()
        st.rerun()

    st.divider()

    # Example queries - more compact
    st.header("💡 Examples")
    example_queries = [
        "Card declined at store",
        "Reset my PIN",
        "Extra fee charged",
        "Transfer not arrived",
    ]

    for query in example_queries:
        if st.button(query, use_container_width=True, key=f"example_{query}"):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    st.divider()

    # About section
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        **3-Layer RAG Architecture:**
        1. Contextual Retrieval - Multi-turn conversation handling
        2. Smart Retrieval - LLM disambiguation for overlapping categories
        3. Base Retrieval - Semantic search + Cross-Encoder reranking

        **Tech Stack:**
        - OpenAI GPT-4o-mini (generation + disambiguation)
        - ChromaDB (vector database)
        - Cross-Encoder reranking (ms-marco-TinyBERT)
        - Sentence Transformers

        **Performance:** 96% accuracy • Real-time streaming • 77 categories
        """)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add compact welcome message
    welcome_msg = """👋 **Welcome!** I can help with card issues, transfers, payments, fees, and 73 other banking topics. Try the example questions in the sidebar!"""
    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome_msg,
        "metadata": {}
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if "metadata" in message and message["metadata"] and "sources" in message["metadata"]:
            with st.expander("📚 View Sources", expanded=False):
                sources = message["metadata"]["sources"]
                st.write(f"**Retrieved {len(sources)} relevant documents:**")
                for i, source in enumerate(sources, 1):
                    similarity_pct = source['similarity'] * 100
                    st.markdown(
                        f"<div class='source-badge'>{i}. {source['category']} ({similarity_pct:.1f}% match)</div>",
                        unsafe_allow_html=True
                    )

# Chat input
if prompt := st.chat_input("Type your banking question here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        if use_streaming:
            # Streaming response
            response_placeholder = st.empty()
            sources_placeholder = st.container()

            full_response = ""
            sources_data = None

            # Stream the response
            for chunk in rag.query_stream(prompt, n_results=n_results, include_sources=show_sources):
                if chunk['done']:
                    # Final chunk - get the complete answer
                    full_response = chunk['full_answer']
                elif 'sources' in chunk:
                    # First chunk with metadata
                    sources_data = chunk.get('sources', [])
                else:
                    # Content chunk
                    full_response += chunk['chunk']
                    response_placeholder.markdown(full_response + "▌")

            # Display final response
            response_placeholder.markdown(full_response)

            # Display sources
            metadata = {}
            if show_sources and sources_data:
                metadata["sources"] = sources_data
                with sources_placeholder.expander("📚 View Sources", expanded=False):
                    st.write(f"**Retrieved {len(sources_data)} relevant documents:**")
                    for i, source in enumerate(sources_data, 1):
                        similarity_pct = source['similarity'] * 100
                        st.markdown(
                            f"<div class='source-badge'>{i}. {source['category']} ({similarity_pct:.1f}% match)</div>",
                            unsafe_allow_html=True
                        )
        else:
            # Non-streaming response
            with st.spinner("🤔 Thinking..."):
                result = rag.query(
                    prompt,
                    n_results=n_results,
                    include_sources=show_sources
                )

            # Display response
            st.markdown(result["answer"])
            full_response = result["answer"]

            # Display sources
            metadata = {}
            if show_sources and "sources" in result:
                metadata["sources"] = result["sources"]
                with st.expander("📚 View Sources", expanded=False):
                    st.write(f"**Retrieved {len(result['sources'])} relevant documents:**")
                    for i, source in enumerate(result["sources"], 1):
                        similarity_pct = source['similarity'] * 100
                        st.markdown(
                            f"<div class='source-badge'>{i}. {source['category']} ({similarity_pct:.1f}% match)</div>",
                            unsafe_allow_html=True
                        )

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "metadata": metadata
    })

# Compact footer
st.caption("💡 Portfolio demo • 3-Layer RAG (Contextual + Smart + Base Retrieval) • 96% accuracy")

# Run instructions comment
# Run with: streamlit run app/ui.py
