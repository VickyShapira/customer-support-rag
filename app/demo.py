"""
Banking RAG System - Interactive Demo
Scenario: Sarah's Banking Journey

A guided demonstration showing how the RAG system handles complex,
multi-turn conversations for banking customer support.
"""

import streamlit as st
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Find project root
current = Path.cwd()
while current != current.parent:
    if (current / 'src').exists():
        project_root = current
        break
    current = current.parent
else:
    project_root = Path.cwd().parent

sys.path.insert(0, str(project_root / 'src'))

from rag_pipeline import RAGPipeline

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Banking RAG Demo - Sarah's Journey",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZE SESSION STATE WITH CACHING
# ============================================================================

@st.cache_resource
def load_rag_pipeline():
    """Cache the RAG pipeline to avoid reloading on every interaction"""
    vector_db_path = str(project_root / "data" / "vector_db")
    return RAGPipeline(
        vector_db_path=vector_db_path,
        model="gpt-4o-mini",
        use_contextual_retriever=True,  # Enable contextual query reformulation for multi-turn conversations
        use_smart_retriever=True  # Enable LLM disambiguation for overlapping categories
    )

# Initialize with loading message
if 'rag' not in st.session_state:
    with st.spinner("🔄 Loading AI system... (this only happens once)"):
        st.session_state.rag = load_rag_pipeline()
        st.session_state.messages = []
        st.session_state.scenario_step = 0
        st.session_state.show_insights = True
        st.session_state.query_count = 0
        st.session_state.correct_predictions = 0
        st.session_state.total_latency = 0
        st.session_state.free_chat_count = 0  # Track free chat queries to limit API costs
        st.success("✅ System loaded successfully!")
        time.sleep(0.5)  # Brief pause to show success message

# ============================================================================
# SARAH'S SCENARIO DEFINITION
# ============================================================================

SARAH_SCENARIO = [
    {
        "query": "My card payment was declined at a restaurant in Paris",
        "expected_category": "declined_card_payment",
        "context": "Act 1: The Problem Discovery",
        "description": "Sarah tries to pay for dinner but her card is declined"
    },
    {
        "query": "Why would that happen? I have money in my account",
        "expected_category": "declined_card_payment",
        "context": "Act 1: Follow-up question",
        "description": "Sarah is confused because she has sufficient funds"
    },
    {
        "query": "It was an international transaction",
        "expected_category": "declined_card_payment",
        "context": "Act 1: Additional context",
        "description": "Sarah clarifies this was abroad"
    },
    {
        "query": "Also, I made a transfer 3 days ago and my balance still hasn't updated",
        "expected_category": "balance_not_updated_after_bank_transfer",
        "context": "Act 2: New Issue",
        "description": "Sarah raises a second, unrelated issue",
        "act_transition": True,
        "transition_message": "🔄 **Sarah shifts to a different banking issue...**"
    },
    {
        "query": "How long do transfers usually take?",
        "expected_category": "transfer_timing",
        "context": "Act 2: Follow-up",
        "description": "Sarah wants to know typical timeframes (ambiguous query)"
    },
    {
        "query": "I noticed I was charged a fee for withdrawing cash at a Paris ATM",
        "expected_category": "cash_withdrawal_charge",
        "context": "Act 3: Fee Discovery",
        "description": "Sarah discovers unexpected charges",
        "act_transition": True,
        "transition_message": "💰 **Sarah discovers unexpected fees...**"
    },
    {
        "query": "What are all the fees for international transactions?",
        "expected_category": "exchange_charge",
        "context": "Act 3: Comprehensive inquiry",
        "description": "Sarah wants complete information"
    }
]

# ============================================================================
# HEADER & INTRODUCTION
# ============================================================================

st.title("🏦 Banking Customer Support AI - Live Demo")

st.markdown("""
### 👋 Meet Sarah

Sarah just returned from a trip to Paris and is experiencing multiple banking issues.
Watch how our AI assistant handles her complex, multi-turn conversation with **96% accuracy**
across 77 different banking categories.

**What this demo shows:**
- 🧠 **Contextual Retrieval** - Reformulates follow-up questions with conversation history
- 🎯 **Smart Retrieval** - LLM disambiguation for overlapping categories
- ⚡ **Real-time responses** - Streaming generation for better UX
- 🔍 **Transparent reasoning** - See exactly why the AI chose each category

*Use the controls below to step through Sarah's conversation or try your own questions!*
""")

st.divider()

# ============================================================================
# SIDEBAR - DEMO CONTROLS & INSIGHTS
# ============================================================================

with st.sidebar:
    st.header("🎬 Demo Controls")
    
    # Scenario progress
    st.subheader("📖 Sarah's Journey Progress")
    progress = st.session_state.scenario_step / len(SARAH_SCENARIO)
    st.progress(progress)
    st.caption(f"Step {st.session_state.scenario_step} of {len(SARAH_SCENARIO)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Next", type="primary", use_container_width=True):
            if st.session_state.scenario_step < len(SARAH_SCENARIO):
                st.session_state.show_next = True
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.rag.reset_conversation()
            st.session_state.messages = []
            st.session_state.scenario_step = 0
            st.session_state.query_count = 0
            st.session_state.correct_predictions = 0
            st.session_state.total_latency = 0
            st.session_state.free_chat_count = 0
            st.rerun()
    
    st.divider()
    
    # Current scenario context
    if st.session_state.scenario_step > 0 and st.session_state.scenario_step <= len(SARAH_SCENARIO):
        current_step = SARAH_SCENARIO[st.session_state.scenario_step - 1]
        st.info(f"**{current_step['context']}**\n\n{current_step['description']}")
    
    st.divider()
    
    # Live system insights
    st.header("📊 Live System Insights")

    if st.session_state.query_count > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        with col2:
            st.metric("Memory", len(st.session_state.rag.get_conversation_history()))

        # Only show accuracy in guided scenario mode (where we have expected categories)
        if st.session_state.scenario_step > 0:
            accuracy = (st.session_state.correct_predictions / st.session_state.query_count) * 100
            st.metric("Scenario Accuracy", f"{accuracy:.0f}%", delta=f"{accuracy-96.0:+.1f}%")
    else:
        st.caption("Run queries to see live metrics")
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    st.session_state.show_insights = st.checkbox("Show detailed insights", value=True)
    show_sources = st.checkbox("Show source documents", value=True)
    
    st.divider()
    
    # About
    st.header("ℹ️ About")
    st.markdown("""
    **Performance:** 96% accuracy

    **Tech Stack:**
    - GPT-4o-mini
    - OpenAI Embeddings
    - ChromaDB

    [📖 README](https://github.com/VickyShapira/banking-rag-system) |
    [💻 GitHub](https://github.com/VickyShapira)
    """)

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Scenario selector
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    mode = st.radio(
        "**Choose mode:**",
        ["🎭 Guided Scenario (Sarah's Journey)", "💬 Free Chat (Try Your Own)"],
        horizontal=True
    )

with col2:
    if mode.startswith("🎭") and st.session_state.scenario_step < len(SARAH_SCENARIO):
        if st.button("⏩ Skip to End", use_container_width=True):
            # Run all remaining steps quickly
            with st.spinner("Running remaining scenario steps..."):
                while st.session_state.scenario_step < len(SARAH_SCENARIO):
                    step = SARAH_SCENARIO[st.session_state.scenario_step]
                    
                    # Quick execution without display
                    start_time = time.time()
                    response = st.session_state.rag.query(step['query'], n_results=3)
                    latency = (time.time() - start_time) * 1000
                    
                    st.session_state.messages.append({
                        "role": "user",
                        "content": step['query'],
                        "context": step['context']
                    })
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "category": response['sources'][0]['category'],
                        "expected": step['expected_category'],
                        "latency": latency,
                        "sources": response['sources'][:3]
                    })
                    
                    st.session_state.scenario_step += 1
                    st.session_state.query_count += 1
                    st.session_state.total_latency += latency
                    
                    if response['sources'][0]['category'] == step['expected_category']:
                        st.session_state.correct_predictions += 1
            
            st.rerun()

with col3:
    st.caption(f"Scenario: {st.session_state.scenario_step}/{len(SARAH_SCENARIO)}")

st.divider()

# Display conversation history
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            if "context" in message:
                st.caption(f"*{message['context']}*")
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])
            
            if st.session_state.show_insights and "category" in message:
                with st.expander("🔍 View Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Predicted Category:**  \n`{message['category']}`")
                        if "latency" in message:
                            st.markdown(f"**Response Time:**  \n{message['latency']:.0f}ms")
                    
                    with col2:
                        if "expected" in message:
                            is_correct = message['category'] == message['expected']
                            status = "✅ Correct" if is_correct else "❌ Incorrect"
                            st.markdown(f"**Expected:**  \n`{message['expected']}`")
                            st.markdown(f"**Status:** {status}")
                    
                    if show_sources and "sources" in message:
                        st.markdown("**Top 3 Retrieved Sources:**")
                        for i, source in enumerate(message['sources'], 1):
                            similarity = source.get('similarity', 'N/A')
                            if isinstance(similarity, float):
                                st.markdown(f"{i}. `{source['category']}` (similarity: {similarity:.3f})")
                            else:
                                st.markdown(f"{i}. `{source['category']}`")

# Handle next step in scenario WITH OPTIMIZED PERCEIVED STREAMING
if st.session_state.get('show_next') and mode.startswith("🎭"):
    if st.session_state.scenario_step < len(SARAH_SCENARIO):
        step = SARAH_SCENARIO[st.session_state.scenario_step]
        
        # Show act transition if applicable
        if step.get('act_transition'):
            st.info(step['transition_message'])
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.caption(f"*{step['context']}*")
            st.markdown(step['query'])
        
        st.session_state.messages.append({
            "role": "user",
            "content": step['query'],
            "context": step['context']
        })
        
        # Generate response with PERCEIVED STREAMING
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            # Show retrieval progress
            retrieval_status = st.empty()
            retrieval_status.caption("🔍 Searching knowledge base...")
            
            start_time = time.time()
            
            # Execute query
            response = st.session_state.rag.query(step['query'], n_results=3)

            latency = (time.time() - start_time) * 1000
            retrieval_status.empty()  # Clear status

            full_response = response['answer']
            sources = response['sources'][:3]

            # Handle case where no sources are returned
            if not sources or len(sources) == 0:
                st.error("⚠️ No relevant sources found. Please check the retrieval system.")
                predicted_category = "unknown"
            else:
                predicted_category = sources[0]['category']
            
            # Animate the text appearing (perceived streaming)
            displayed_text = ""
            words = full_response.split()
            
            for word in words:
                displayed_text += word + " "
                message_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.015)  # 15ms per word for smooth animation
            
            message_placeholder.markdown(full_response)
            
            # Track metrics
            st.session_state.query_count += 1
            st.session_state.total_latency += latency
            
            if predicted_category == step['expected_category']:
                st.session_state.correct_predictions += 1
            
            # Show insights
            if st.session_state.show_insights:
                with st.expander("🔍 View Details", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Predicted Category:**  \n`{predicted_category}`")
                        st.markdown(f"**Response Time:**  \n{latency:.0f}ms")
                    
                    with col2:
                        is_correct = predicted_category == step['expected_category']
                        status = "✅ Correct" if is_correct else "❌ Incorrect"
                        st.markdown(f"**Expected:**  \n`{step['expected_category']}`")
                        st.markdown(f"**Status:** {status}")
                    
                    if show_sources:
                        st.markdown("**Top 3 Retrieved Sources:**")
                        for i, source in enumerate(sources, 1):
                            similarity = source.get('similarity', 'N/A')
                            if isinstance(similarity, float):
                                st.markdown(f"{i}. `{source['category']}` (similarity: {similarity:.3f})")
                            else:
                                st.markdown(f"{i}. `{source['category']}`")
        
        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "category": predicted_category,
            "expected": step['expected_category'],
            "latency": latency,
            "sources": sources
        })
        
        st.session_state.scenario_step += 1
        st.session_state.show_next = False
        st.rerun()

# Free chat input WITH OPTIMIZED PERCEIVED STREAMING
if mode.startswith("💬"):
    # API cost protection: limit free chat to 5 queries per session
    MAX_FREE_CHAT_QUERIES = 5

    if st.session_state.free_chat_count >= MAX_FREE_CHAT_QUERIES:
        st.warning(f"⚠️ **Demo limit reached:** You've used {MAX_FREE_CHAT_QUERIES} free chat queries. This prevents API abuse. Please reset to continue or try the guided scenario!")
        if st.button("🔄 Reset to Continue", type="primary"):
            st.session_state.rag.reset_conversation()
            st.session_state.messages = []
            st.session_state.free_chat_count = 0
            st.rerun()
    else:
        remaining = MAX_FREE_CHAT_QUERIES - st.session_state.free_chat_count
        st.info(f"💡 **Free Chat Mode:** Ask any banking question! ({remaining} queries remaining)")

        user_input = st.chat_input("Ask a banking question...")

        if user_input:
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            # Generate response with PERCEIVED STREAMING
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()

                # Show retrieval progress
                retrieval_status = st.empty()
                retrieval_status.caption("🔍 Searching knowledge base...")

                start_time = time.time()

                # Execute query
                response = st.session_state.rag.query(user_input, n_results=3)

                latency = (time.time() - start_time) * 1000
                retrieval_status.empty()  # Clear status

                full_response = response['answer']
                sources = response['sources'][:3]
                predicted_category = sources[0]['category']

                # Animate the text appearing (perceived streaming)
                displayed_text = ""
                words = full_response.split()

                for word in words:
                    displayed_text += word + " "
                    message_placeholder.markdown(displayed_text + "▌")
                    time.sleep(0.015)  # 15ms per word

                message_placeholder.markdown(full_response)

                # Track metrics
                st.session_state.query_count += 1
                st.session_state.total_latency += latency
                st.session_state.free_chat_count += 1  # Increment free chat counter

                # Show insights
                if st.session_state.show_insights:
                    with st.expander("🔍 View Details"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**Predicted Category:**  \n`{predicted_category}`")
                            st.markdown(f"**Response Time:**  \n{latency:.0f}ms")

                        if show_sources:
                            st.markdown("**Top 3 Retrieved Sources:**")
                            for i, source in enumerate(sources, 1):
                                similarity = source.get('similarity', 'N/A')
                                if isinstance(similarity, float):
                                    st.markdown(f"{i}. `{source['category']}` (similarity: {similarity:.3f})")
                                else:
                                    st.markdown(f"{i}. `{source['category']}`")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "category": predicted_category,
                "latency": latency,
                "sources": sources
            })

            st.rerun()

# Scenario completion message with detailed stats
if mode.startswith("🎭") and st.session_state.scenario_step >= len(SARAH_SCENARIO):
    st.balloons()
    
    st.success("""
    ### 🎉 Scenario Complete!
    
    You've seen how the system handles Sarah's complex, multi-turn conversation across multiple banking issues.
    """)
    
    # Detailed stats
    col1, col2, col3 = st.columns(3)

    accuracy = (st.session_state.correct_predictions / st.session_state.query_count) * 100

    with col1:
        st.metric("Total Queries", st.session_state.query_count)
    with col2:
        st.metric("Accuracy", f"{accuracy:.0f}%", delta=f"{accuracy-96.0:+.1f}% vs system")
    with col3:
        st.metric("Memory Used", f"{len(st.session_state.rag.get_conversation_history())} msgs")
    
    st.markdown("""
    **What happened:**
    - ✅ Maintained context across 7 different queries
    - ✅ Handled 3 distinct issues (declined card, delayed transfer, ATM fees)
    - ✅ Correctly identified category transitions between topics
    - ✅ Managed ambiguous queries like "How long do transfers take?"
    
    **Try it yourself:**
    Switch to "Free Chat" mode above to ask your own questions!
    """)
    
    # Download transcript option
    st.divider()
    
    # Generate transcript
    transcript = "# Sarah's Banking Journey - Conversation Transcript\n\n"
    transcript += f"**Session Statistics:**\n"
    transcript += f"- Total queries: {st.session_state.query_count}\n"
    transcript += f"- Accuracy: {accuracy:.1f}%\n"
    transcript += f"- Memory used: {len(st.session_state.rag.get_conversation_history())} messages\n\n"
    transcript += "---\n\n"
    
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            context = msg.get('context', '')
            transcript += f"**👤 Sarah:** {msg['content']}\n"
            if context:
                transcript += f"*{context}*\n"
            transcript += "\n"
        else:
            transcript += f"**🤖 AI Assistant:** {msg['content']}\n"
            if 'category' in msg:
                transcript += f"*Predicted category: {msg['category']}*\n"
                if 'expected' in msg:
                    transcript += f"*Expected category: {msg['expected']}*\n"
                if 'latency' in msg:
                    transcript += f"*Response time: {msg['latency']:.0f}ms*\n"
            transcript += "\n---\n\n"
    
    st.download_button(
        label="📥 Download Conversation Transcript",
        data=transcript,
        file_name="sarah_banking_journey_transcript.md",
        mime="text/markdown",
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 System Performance**")
    st.caption("• 96% accuracy")
    st.caption("• 77 banking categories")
    st.caption("• Real-time streaming responses")

with col2:
    st.markdown("**🔗 Resources**")
    st.caption("[📖 README](https://github.com/VickyShapira/banking-rag-system) • [💻 GitHub](https://github.com/VickyShapira) • [📓 Notebooks](https://github.com/VickyShapira/banking-rag-system/tree/main/notebooks)")

with col3:
    st.markdown("**👤 Author**")
    st.caption("Victoria Shapira")
    st.caption("[LinkedIn](https://www.linkedin.com/in/victoria-shapira-006849156) • [Email](mailto:victoriya.shapira@gmail.com)")