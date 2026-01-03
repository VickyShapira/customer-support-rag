# Banking Customer Support RAG System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-82%25-yellowgreen)

**[🎬 Try Live Demo](https://customer-support-rag.streamlit.app)** | **[📊 View Analysis Notebooks](notebooks/)**

A production-ready RAG system with **3-layer intelligent retrieval** for banking customer support, improving multi-turn conversation accuracy from **82% to 96%** through contextual reformulation and LLM disambiguation.

### 🎯 Key Insight

**The Problem:** Banking chatbots struggle with overlapping categories ("My card isn't working" - broken card? declined payment? lost card?) and multi-turn conversations where customers say "Why did that happen?" without context. Baseline retrieval achieves 90.5% on single queries but drops to 82% in realistic conversations.

**The Solution:** 3-layer retrieval architecture that reformulates vague follow-ups using conversation history and disambiguates similar categories with LLM when confidence is low. Achieves **96% accuracy on multi-turn conversations** (vs 82% baseline), with 97.1% on context-dependent queries.

**Key Learning:** Simple, well-executed solutions beat over-engineering. Cross-Encoder reranking provided minimal improvement (+0.8%) while adding complexity. The Contextual Retriever (Layer 1) was the breakthrough - improving context-dependent accuracy from 79.4% to 97.1%.

---

## 🎯 Project Overview

This project demonstrates advanced ML engineering by building an intelligent customer support chatbot with a **three-layer retrieval architecture**. The system combines contextual query reformulation, LLM-based disambiguation, and semantic search with large language models to handle multi-turn conversations and overlapping categories.

**Key Innovation:** Three-layer retrieval system that handles conversation context, disambiguates overlapping categories, and maintains high accuracy

### Problem Statement

Banking customer support faces unique challenges:
- **Overlapping categories** with similar language (e.g., "declined_card_payment" vs "card_not_working")
- **Multi-turn conversations** requiring context tracking ("Why did that happen?" → needs previous context)
- **Ambiguous queries** needing intelligent disambiguation
- **77 distinct categories** requiring precise classification

### Solution Architecture

The system implements a **multi-stage RAG pipeline** with nested retrieval layers:

**Retrieval Architecture (3 Layers):**
1. **Layer 1 - Contextual Retrieval**: Query reformulation for multi-turn conversations using GPT-4o-mini
2. **Layer 2 - Smart Retrieval**: LLM-based disambiguation for overlapping categories (triggers on low confidence/similarity gaps)
3. **Layer 3 - Base Retrieval**: Semantic search (OpenAI embeddings + ChromaDB)

**Generation:**
4. **Answer Generation**: GPT-4o-mini with conversation history and retrieved context

## 🏗️ Technical Architecture

```
User Query + Conversation History
    ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Contextual Retrieval                         │
│  - Detects follow-up queries (< 5 words, pronouns)    │
│  - GPT-4o-mini reformulation to standalone query      │
│  - Example: "Why?" → "Why was my card payment declined?"│
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Smart Retrieval (Disambiguation)             │
│  - Triggers on low confidence (< 0.38 similarity)      │
│  - Triggers on small gap (< 0.10 between top 2)       │
│  - Fetches 20 candidates, extracts 8 unique categories│
│  - GPT-4o-mini selects best category                  │
│  - Handles overlapping categories (declined vs broken) │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Base Retrieval                               │
│  - OpenAI text-embedding-3-small embeddings           │
│  - ChromaDB vector similarity search                  │
│  - Returns top-k most similar documents               │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Generation Layer                                       │
│  - GPT-4o-mini with retrieved context                 │
│  - Conversation history (last 6 messages)             │
│  - Streaming support for real-time responses          │
└─────────────────────────────────────────────────────────┘
    ↓
Final Response
```

### Key Components

- **[src/retrieval/contextual_retriever.py](src/retrieval/contextual_retriever.py)**: Query reformulation for multi-turn conversations
- **[src/retrieval/smart_retriever.py](src/retrieval/smart_retriever.py)**: LLM disambiguation for overlapping categories
- **[src/retrieval/retriever.py](src/retrieval/retriever.py)**: Vector search with OpenAI embeddings + ChromaDB
- **[src/generation/generator.py](src/generation/generator.py)**: LLM-based answer generation with streaming support
- **[src/rag_pipeline.py](src/rag_pipeline.py)**: End-to-end orchestration with three-layer retrieval architecture
- **[app/api.py](app/api.py)**: FastAPI endpoints for production deployment
- **[app/ui.py](app/ui.py)**: Streamlit interface for interactive testing

## 📊 Performance Metrics

### Multi-Turn Conversation Performance

| Query Type | Baseline (No Layers) | Combined (3 Layers) | Improvement |
|------------|---------------------|---------------------|-------------|
| **Overall Accuracy** | **82.0%** | **96.0%** | **+14.0%** ✨ |
| Context-independent | 87.5% | 93.8% | +6.3% |
| Context-dependent | 79.4% | 97.1% | **+17.7%** ✨ |
| Scenario success rate | 53.3% | 100.0% | +46.7% |

**Test Set:** 6 realistic conversation scenarios with multi-turn exchanges
**Single-Turn Baseline:** 90.5% accuracy on 3,080 test queries (see [Pipeline Evolution](#pipeline-evolution--impact) below)

**Key Insight:** The 3-layer architecture's value is in **conversational AI**, not single-turn retrieval. Context-dependent queries saw 17.7% improvement - these are the "Why?" and "How long?" follow-ups that require understanding conversation history.

### Pipeline Evolution & Impact

| Pipeline Configuration | Single-Turn Accuracy | Multi-Turn Overall | Notes |
|------------------------|---------------------|-------------------|-------|
| **Base Retrieval** | 90.5% | 82.0% | Semantic search only |
| **+ Cross-Encoder Reranking** | 91.3% (+0.8%) | - | Tested but not adopted - minimal benefit for added complexity |
| **+ Smart Retrieval** | - | 82.0% | LLM disambiguation for overlapping categories |
| **+ Contextual Retrieval (Final)** | - | **96.0%** | Breakthrough for multi-turn conversations |

### Smart Retrieval Impact

**Handles Challenging Category Pairs:**
- `balance_not_updated_after_bank_transfer` vs `pending_transfer` vs `transfer_not_received_by_recipient` (confusion rates >10%)
- `card_payment_not_recognised` vs `direct_debit_payment_not_recognised` (similar payment issues)
- `failed_transfer` vs `transfer_timing` (transfer problems vs status queries)

**Trigger Conditions:**
- Low confidence: Top similarity < 0.38
- Small gap: Difference between top 2 results < 0.10
- Known overlapping category groups (from confusion matrix analysis)

### Latency Analysis

**End-to-End Response Times:**

| Pipeline | Mean | Median | P95 | Primary Driver |
|----------|------|--------|-----|----------------|
| Original (Base) | 2,502ms | 2,450ms | 3,301ms | LLM generation (93%) |
| + Smart Retrieval | 3,318ms | 3,236ms | 4,337ms | +LLM disambiguation (~800ms) |
| **Combined (Full)** | **3,227ms** | **3,096ms** | **4,349ms** | Contextual + Smart |
| Multi-turn scenarios | 3,582ms | 3,487ms | 4,634ms | Additional context processing |

**Key Findings:**
- **LLM generation dominates latency** (93% of total time in baseline)
- Smart Retrieval adds ~800ms overhead when triggered
- Contextual reformulation is efficient (minimal added latency)
- **User experience**: Streaming enabled → responses appear in ~200ms (perceived 10x improvement)

**Optimization Priorities:**
1. ✅ **Streaming responses** - Highest UX impact, no accuracy trade-off
2. ⚠️ **Faster LLM** - Would sacrifice accuracy for speed
3. ⚠️ **Reduce Smart Retrieval** - Would hurt disambiguation quality

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
OpenAI API key
4GB+ RAM
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/VickyShapira/customer-support-rag.git
cd customer-support-rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

4. **Download the knowledge base**
```bash
# Place your ChromaDB vector database in ./vector_db/
# Or build from scratch using the BANKING77 dataset
```

### Quick Start

**Option 1: Interactive UI (Streamlit)**
```bash
streamlit run app/ui.py
```

**Option 2: API Server (FastAPI)**
```bash
uvicorn app.api:app --reload --port 8000
```

**Option 3: Python API**
```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline with all three layers enabled (default)
rag = RAGPipeline(
    vector_db_path="./vector_db",
    model="gpt-4o-mini",
    use_contextual_retriever=True,  # Layer 1: Multi-turn conversation support
    use_smart_retriever=True,        # Layer 2: Category disambiguation
    confidence_threshold=0.38,       # Smart retrieval trigger threshold
    similarity_gap_threshold=0.10    # Gap threshold for disambiguation
)

# Single query
response = rag.query(
    question="Why was my card payment declined?",
    n_results=3
)
print(response['answer'])

# Multi-turn conversation
response1 = rag.query("My card payment was declined")
print(response1['answer'])

# Follow-up question (contextual retriever reformulates this)
response2 = rag.query("Why did that happen?")
print(response2['answer'])

# Reset conversation history
rag.reset_conversation()
```

## 💡 Key Features

### 1. Three-Layer Retrieval Architecture

**Layer 1: Contextual Retrieval** ([contextual_retriever.py](src/retrieval/contextual_retriever.py))
- Detects follow-up queries using heuristics (short queries, pronouns, question words)
- Reformulates queries with conversation context using GPT-4o-mini
- Example: "Why?" → "Why was my card payment declined?"

```python
from src.retrieval.contextual_retriever import ContextualRetriever

contextual = ContextualRetriever(
    base_retriever=base_retriever,
    use_context=True,
    model="gpt-4o-mini"
)
```

**Layer 2: Smart Retrieval** ([smart_retriever.py](src/retrieval/smart_retriever.py))
- Triggers on low confidence (< 0.38) or small similarity gap (< 0.10)
- Fetches 20 candidates, extracts 8 unique categories
- Uses GPT-4o-mini to disambiguate and select best category
- Handles known overlapping category groups

```python
from src.retrieval.smart_retriever import SmartRetriever

smart = SmartRetriever(
    base_retriever=base_retriever,
    confidence_threshold=0.38,
    similarity_gap_threshold=0.10
)
```

**Layer 3: Base Retrieval** ([retriever.py](src/retrieval/retriever.py))
- OpenAI text-embedding-3-small for semantic search
- ChromaDB vector database for fast similarity search
- Returns top-k most relevant documents

```python
from src.retrieval.retriever import KnowledgeBaseRetriever

retriever = KnowledgeBaseRetriever(
    vector_db_path="./vector_db",
    use_reranking=False
)
```

### 2. Streaming Responses
- Real-time answer generation for better UX
- Immediate feedback while maintaining accuracy
- Conversation context management (last 6 messages = 3 turns)

```python
for chunk in rag.query_stream(question="How do I freeze my card?"):
    if not chunk['done']:
        print(chunk['chunk'], end='', flush=True)
```

### 3. Conversation Memory
- Maintains context across multiple exchanges
- Automatic history pruning (last 6 messages)
- Contextual follow-up question handling via Layer 1

### 4. Production-Ready API
- RESTful endpoints with FastAPI
- Health checks and monitoring
- Request validation and error handling
- Async support for concurrent requests

## 🔬 Technical Deep Dive

### Design Decisions

**Why Three Layers of Retrieval?**

Each layer solves a distinct problem:

1. **Contextual Retrieval (Layer 1)** - Solves multi-turn conversation ambiguity
   - Challenge: "Why?" or "How long?" depends entirely on prior context
   - Solution: LLM reformulation to standalone query before retrieval
   - Alternative considered: Manual context concatenation (too simplistic)
   - Trade-off: +200-300ms latency for significantly better conversational experience

2. **Smart Retrieval (Layer 2)** - Solves overlapping category confusion
   - Challenge: Embeddings alone can't distinguish semantically similar but functionally different categories
   - Solution: LLM classification with detailed category definitions when confidence is low
   - Alternative considered: Fine-tuned embeddings (requires extensive retraining)
   - Trade-off: +800ms latency when triggered for better accuracy on ambiguous cases

3. **Cross-Encoder Reranking (Tested, Not Adopted)** - Minimal improvement didn't justify complexity
   - Challenge: Bi-encoder embeddings might miss query-document interaction nuances
   - Solution tested: Cross-encoder with 2x over-fetch
   - Result: Only +0.8% accuracy improvement (90.5% → 91.3%)
   - Decision: **Not adopted** - OpenAI embeddings already excellent, reranking adds complexity for minimal gain

**Embedding Model Selection**
- OpenAI `text-embedding-3-small`: Fast, cost-effective, high quality
- Considered: sentence-transformers models (offline capability)
- Trade-off: API cost vs inference infrastructure complexity
- Decision: API approach for faster iteration and deployment

**Smart Retrieval Thresholds**
- **Confidence threshold (0.38)**: Tuned empirically by analyzing queries where top similarity was low
- **Gap threshold (0.10)**: Identified from confusion matrix analysis of close category pairs
- **Over-fetch (20 candidates, 8 categories)**: Balances category diversity with LLM context limits

### Data Pipeline

1. **Knowledge Base Construction**
   - Source: BANKING77 dataset (13,083 samples)
   - Train/test split: 70/30 (prevents data leakage)
   - Storage: ChromaDB with persistent storage
   - Embeddings: Pre-computed with text-embedding-3-small

2. **Query Processing**
   - Input validation and normalization
   - Embedding generation using OpenAI API
   - Vector similarity search in ChromaDB
   - Context formatting for LLM

3. **Answer Generation**
   - System prompt with role definition
   - Retrieved context injection
   - Conversation history inclusion
   - Temperature tuning for consistency (0.1 for deterministic responses)

## 📈 Evaluation Framework

### Metrics Tracked

- **Accuracy**: Category prediction correctness (top-1, top-3, top-5)
- **Latency**: End-to-end response time broken down by component (retrieval, generation, total)
- **Retrieval Quality**: Top-k accuracy, MRR (Mean Reciprocal Rank), NDCG
- **Conversation Context**: Turn-level accuracy for multi-turn scenarios
- **Cost**: API token consumption for generation

### Testing Strategy

#### Single-Turn Evaluation (BANKING77 Test Set)

- Proper train/test separation: No data leakage (70/30 split)
- Comprehensive evaluation: Full 3,080-sample test set from BANKING77
- Parallel processing: 10 workers for efficient evaluation (~2 minutes per run)
- Multiple query types: Short queries, negation queries, complex multi-clause questions
- Confusion matrix analysis: Identifying failure patterns and semantically overlapping categories

#### Multi-Turn Conversation Evaluation

- 15 realistic conversation scenarios testing context understanding across 50 total turns
- Context-dependent queries: Testing pronoun resolution, topic tracking, implicit references
- Three-way pipeline comparison: Original (basic) → Smart (LLM disambiguation) → Combined (Smart + Contextual)
- Scenario types: International transfers, card issues, ATM problems, top-up questions, pronoun-heavy follow-ups, implicit comparisons, topic switches
- Success threshold: 75% accuracy per scenario (≥3/4 turns correct)
- Metrics: Overall turn accuracy, context-dependent accuracy, context-independent accuracy, scenario pass rate

### Example Evaluation: Multi-Turn Context Understanding

```python
# Evaluate contextual retrieval (see notebooks/07_contextual_pipeline_evaluation.pdf)
from src.rag_pipeline import RAGPipeline

# Test three pipeline configurations
pipelines = {
    "Original": RAGPipeline(
        vector_db_path="./data/vector_db",
        model="gpt-4o-mini",
        use_contextual_retriever=False,
        use_smart_retriever=False
    ),
    "Smart": RAGPipeline(
        vector_db_path="./data/vector_db",
        model="gpt-4o-mini",
        use_contextual_retriever=False,
        use_smart_retriever=True  # LLM disambiguation
    ),
    "Combined": RAGPipeline(
        vector_db_path="./data/vector_db",
        model="gpt-4o-mini",
        use_contextual_retriever=True,  # Context tracking
        use_smart_retriever=True        # LLM disambiguation
    )
}

# Example scenario: Card Payment Issues
scenario = {
    "name": "Card Payment Issues",
    "turns": [
        {"query": "My card payment was declined at a store", 
         "expected": ["declined_card_payment"]},
        {"query": "Why did this happen?",  # Context-dependent
         "expected": ["declined_card_payment"]},
        {"query": "Can I fix it?",  # Context-dependent
         "expected": ["declined_card_payment", "contactless_not_working"]}
    ]
}

# Evaluate each pipeline
for name, pipeline in pipelines.items():
    pipeline.reset_conversation()
    correct = 0
    
    for turn in scenario["turns"]:
        response = pipeline.query(turn["query"], n_results=5)
        retrieved_cats = [s["category"] for s in response["sources"][:3]]
        if any(cat in retrieved_cats for cat in turn["expected"]):
            correct += 1
    
    print(f"{name}: {correct}/{len(scenario['turns'])} turns correct")

# Results across 15 scenarios (50 total turns):
# Original:  82.0% overall (41/50) | 79.4% context-dependent | 87.5% context-independent
# Smart:     82.0% overall (41/50) | 76.5% context-dependent | 93.8% context-independent
# Combined:  96.0% overall (48/50) | 97.1% context-dependent | 93.8% context-independent

# Key Finding: Context tracking (Combined pipeline) dramatically improves
# context-dependent turn accuracy from ~77-79% to 97.1%
```

## 🎓 Key Learnings

### What Worked
1. **Layered retrieval architecture**: Each layer addresses a specific failure mode (context, disambiguation, semantic search)
2. **Smart Retrieval with thresholds**: Triggering LLM only when needed (low confidence/gap) balances cost and accuracy
3. **Contextual query reformulation**: Critical for natural multi-turn conversations - improved context-dependent accuracy from 79.4% to 97.1%
4. **Confusion matrix-driven optimization**: Analyzing category confusions identified overlapping groups (e.g., balance_not_updated vs pending_transfer vs transfer_not_received)
5. **Proper data separation**: Rigorous train/test split prevents overfitting and ensures realistic performance metrics

### What Didn't Work
1. **Cross-Encoder reranking**: Provided minimal improvement (+0.8%, from 90.5% to 91.3%) while adding complexity - OpenAI's embeddings are already excellent for this dataset
2. **Higher rerank multipliers (5x)**: During reranking tests, 5x over-fetch (90.9%) actually decreased accuracy vs 2x (91.3%) - too much noise in candidates
3. **Simple context concatenation**: Just appending previous query didn't handle pronouns and implicit references adequately

### Challenges Solved
1. **Overlapping categories**: Smart Retrieval with detailed category definitions + examples handles semantic similarity
2. **Follow-up questions**: Contextual Retrieval with GPT-4o-mini reformulation achieved 97.1% on context-dependent queries
3. **Low-confidence queries**: Smart Retrieval triggers on similarity < 0.38 to request LLM disambiguation
4. **Close category pairs**: Gap threshold < 0.10 catches cases where top 2 results are too similar

### Future Improvements
- **Fine-tuned embeddings**: Domain-specific embedding models could reduce Smart Retrieval trigger rate and improve baseline performance
- **Model distillation**: Deploy smaller, faster models for production cost optimization
- **Adaptive thresholds**: Learn confidence/gap thresholds per category group based on confusion patterns
- **Caching**: Cache LLM reformulations and classifications for repeated queries

## ⚠️ Known Limitations

- **API dependency**: Requires OpenAI API (cost scales with usage; varies based on query complexity and whether Smart/Contextual retrieval is triggered)
- **English only**: Models trained primarily on English language
- **Conversation history management**: Requires manual reset between different users
- **Latency**: 3-4 seconds end-to-end (mitigated with streaming for perceived <500ms response time)
- **Domain-specific**: Optimized for banking queries, may need retraining for other domains

## 🧪 Testing

Comprehensive test suite with **82% code coverage**:
- **29 test cases** covering all pipeline layers
- **95% coverage** on core retrieval logic (retriever.py)
- **92% coverage** on Smart Retriever disambiguation
- Proper train/test separation prevents data leakage

Run tests:
```bash
# All tests
pytest tests/ -v

# Specific test module
pytest tests/test_retriever.py -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

## 📁 Project Structure

```
customer-support-rag/
│
├── src/
│   ├── retrieval/
│   │   ├── contextual_retriever.py   # Layer 1: Query reformulation for multi-turn conversations
│   │   ├── smart_retriever.py        # Layer 2: LLM disambiguation for overlapping categories
│   │   └── retriever.py              # Layer 3: Vector search with OpenAI embeddings
│   ├── generation/
│   │   └── generator.py              # LLM-based answer generation with streaming
│   └── rag_pipeline.py               # End-to-end orchestration with 3-layer retrieval
│
├── app/
│   ├── api.py                        # FastAPI server
│   ├── ui.py                         # Streamlit interface
│   └── demo.py                       # Demo application
│
├── notebooks/
│   ├── 01_eda.ipynb                             # Data exploration
│   ├── 02_build_knowledge_base.ipynb            # Vector DB construction
│   ├── 03_knowledge_base_analysis.ipynb         # Retrieval quality analysis (90.5% baseline accuracy)
│   ├── 04_cross_encoder_reranking_comparison.ipynb # Reranking experiments (+0.8% improvement)
│   ├── 05_confusion_matrix_analysis.ipynb       # Error analysis & overlapping categories
│   ├── 06_contextual_pipeline_evaluation.ipynb  # Multi-turn evaluation (96% accuracy)
│   └── 07_latency_benchmarking.ipynb            # Performance profiling
│
├── tests/
│   ├── test_retriever.py
│   ├── test_contextual_retriever.py
│   ├── test_smart_retriever.py
│   ├── test_generator.py
│   ├── test_rag_pipeline.py
│   ├── test_e2e.py
│   └── conftest.py
│
├── analysis/                         # Evaluation results and metrics
├── data/                             # Dataset and generated visualizations
├── docs/                             # Additional documentation
├── .github/                          # CI/CD workflows
│
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
CHROMADB_PATH=./vector_db
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o-mini
```

### Pipeline Configuration

**Full Three-Layer Configuration (Recommended for Production)**
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline(
    vector_db_path="./vector_db",
    model="gpt-4o-mini",
    use_contextual_retriever=True,      # Enable Layer 1: Conversation context
    use_smart_retriever=True,            # Enable Layer 2: Disambiguation
    confidence_threshold=0.38,           # Smart retrieval confidence trigger
    similarity_gap_threshold=0.10        # Smart retrieval gap trigger
)
```

**Single-Layer Configuration (Fast, No Context)**
```python
# Only base retrieval (fastest, for single-turn queries)
rag = RAGPipeline(
    vector_db_path="./vector_db",
    model="gpt-4o-mini",
    use_contextual_retriever=False,
    use_smart_retriever=False
)
```

**Two-Layer Configuration (Smart Retrieval Only)**
```python
# Base + Smart retrieval (handles ambiguity, no conversation context)
rag = RAGPipeline(
    vector_db_path="./vector_db",
    model="gpt-4o-mini",
    use_contextual_retriever=False,
    use_smart_retriever=True,
    confidence_threshold=0.38,
    similarity_gap_threshold=0.10
)
```

## 📊 Benchmarking & Evaluation

The project includes comprehensive evaluation across all pipeline stages:

### Notebooks
- **[01_eda.ipynb](notebooks/01_eda.ipynb)**: Exploratory data analysis of BANKING77 dataset
- **[02_build_knowledge_base.ipynb](notebooks/02_build_knowledge_base.ipynb)**: Vector database construction with OpenAI embeddings
- **[03_knowledge_base_analysis.ipynb](notebooks/03_knowledge_base_analysis.ipynb)**: Baseline accuracy evaluation (90.5%)
- **[04_cross_encoder_reranking_comparison.ipynb](notebooks/04_cross_encoder_reranking_comparison.ipynb)**: Reranking impact analysis (+0.8% improvement, 90.5% → 91.3%)
- **[05_confusion_matrix_analysis.ipynb](notebooks/05_confusion_matrix_analysis.ipynb)**: Category confusion identification → drives Smart Retrieval design
- **[06_contextual_pipeline_evaluation.ipynb](notebooks/06_contextual_pipeline_evaluation.ipynb)**: Multi-turn conversation testing (96% accuracy)
- **[07_latency_benchmarking.ipynb](notebooks/07_latency_benchmarking.ipynb)**: Latency breakdown by layer

### Evaluation Methodology
- **Ablation studies**: Test each layer independently and in combination
- **Multi-turn scenarios**: 6 realistic conversation scenarios (e.g., International Transfer Follow-ups, Lost Card Scenario)
- **Latency profiling**: Breakdown by pipeline stage showing LLM generation dominates (93%)
- **Category-specific analysis**: Performance on overlapping vs distinct categories using confusion matrix

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Dataset**: BANKING77 by PolyAI ([HuggingFace](https://huggingface.co/datasets/PolyAI/banking77))
- **Models**: OpenAI (embeddings & generation)
- **Infrastructure**: ChromaDB (vector storage), FastAPI (serving)

## 📧 Contact

**Victoria Shapira** - victoriya.shapira@gmail.com - [LinkedIn](https://www.linkedin.com/in/victoria-shapira-006849156)

**Portfolio**: https://github.com/VickyShapira

---

*Built as a demonstration of ML engineering practices: systematic evaluation methodology, iterative optimization with ablation studies, and honest assessment of what works vs what doesn't.*