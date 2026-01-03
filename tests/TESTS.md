# Testing Guide

Complete guide for testing your RAG pipeline with smart and contextual retrievers.

---

## 🚀 Quick Start

### 1. Activate Virtual Environment (Required!)

```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Tests

**Easiest (Windows):**
```bash
run_tests_venv.bat smart        # Smart retriever tests
run_tests_venv.bat contextual   # Contextual retriever tests
run_tests_venv.bat coverage     # All tests with coverage report
```

**Manual (All platforms):**
```bash
# Activate venv first!
.\venv\Scripts\activate

# Then run tests
python run_comprehensive_tests.py smart
python run_comprehensive_tests.py coverage
```

---

## 📋 Available Test Commands

| Command | What it Tests | Speed |
|---------|---------------|-------|
| `database` | Database creation & validation (6 tests) | Fast |
| `smart` | Smart retriever (LLM disambiguation) (5 tests) | Fast |
| `contextual` | Contextual retriever (query reformulation) (7 tests) | Fast |
| `combined` | Smart + Contextual layering (2 tests) | Fast |
| `retrievers` | All retriever tests (14 tests) | Fast |
| `pipeline` | Pipeline configuration (5 tests) | Fast |
| `fast` | All tests except E2E (25 tests) | Medium |
| `coverage` | All tests with text coverage report | Medium |
| `coverage-html` | All tests + HTML coverage (open htmlcov/index.html) | Medium |
| `e2e` | End-to-end with real API/DB (6 tests) ⚠️ Expensive | Slow |
| `all` | Everything (31 tests) | Medium |

---

## 📊 What Gets Tested

### ✅ Database Tests (6 tests)
- Database directory exists
- ChromaDB connection works
- Collection "banking_support" exists
- 10,003 documents present
- Metadata structure correct
- Embeddings present

### ✅ Smart Retriever Tests (5 tests)
- Initialization and configuration
- Overlapping category groups defined
- LLM triggers on low confidence (< 0.38)
- LLM triggers on small similarity gap (< 0.10)
- High confidence passthrough (no LLM needed)

**What it does:**
- Detects ambiguous queries (e.g., "card declined" vs "card not working")
- Uses GPT-4o-mini to pick the best category
- Handles overlapping categories intelligently

### ✅ Contextual Retriever Tests (7 tests)
- Initialization and configuration
- Short query detection (< 5 words)
- Pronoun detection ("it", "that", "why?")
- Standalone query passthrough
- Query reformulation with context
- Behavior without context
- Disabled mode

**What it does:**
- Reformulates follow-up queries using conversation history
- "Why?" → "Why was my card declined?"
- "How long?" → "How long does a bank transfer take?"

### ✅ Combined Retriever Tests (2 tests)
- Proper layer structure: Contextual → Smart → Base
- Both layers working together

### ✅ Pipeline Integration Tests (5 tests)
- All layer combinations (both, smart only, contextual only, base only)
- Configuration reporting

### ✅ E2E Tests (4 tests) ⚠️ Requires API key
- Real database queries
- Smart retrieval with real API
- Contextual retrieval with real API
- Full pipeline test

---

## 💻 Usage Examples

### Before Committing Code
```bash
run_tests_venv.bat fast
```

### Check Code Coverage
```bash
run_tests_venv.bat coverage-html
# Then open: htmlcov/index.html
```

### Test Specific Component
```bash
run_tests_venv.bat smart        # Smart retriever only
run_tests_venv.bat contextual   # Contextual retriever only
run_tests_venv.bat database     # Database validation
```

### Full Test Suite
```bash
run_tests_venv.bat all
```

### Using pytest Directly
```bash
# Activate venv first!
.\venv\Scripts\activate

# Run specific test class
pytest tests/test_pipeline_comprehensive.py::TestSmartRetriever -v

# Run with coverage
pytest tests/test_pipeline_comprehensive.py --cov=src --cov-report=html -v

# Run all fast tests
pytest -m "not e2e" -v
```

---

## 📁 Test Files

### Main Test File
**test_pipeline_comprehensive.py** (29 tests)
- `TestDatabaseCreation` - Database validation
- `TestSmartRetriever` - Smart retriever logic
- `TestContextualRetriever` - Contextual retriever logic
- `TestCombinedRetrievers` - Layer integration
- `TestRAGPipelineIntegration` - Pipeline configuration
- `TestEndToEnd` - E2E tests (requires API/DB)

### Other Test Files
- **test_pipeline.py** - Quick E2E integration tests
- **test_dual_retriever.py** - Dual retriever layer tests
- **test_retriever.py** - Base retriever tests
- **test_generator.py** - Answer generator tests
- **test_e2e.py** - Comprehensive E2E tests
- **test_performance.py** - Performance benchmarks

---

## 🔧 Test Architecture

```
Query
  ↓
ContextualRetriever (Layer 1) ✅ Tested
  ├─ Reformulates follow-ups
  └─ Uses conversation history
  ↓
SmartRetriever (Layer 2) ✅ Tested
  ├─ LLM disambiguation
  └─ Handles overlapping categories
  ↓
KnowledgeBaseRetriever (Layer 3) ✅ Tested
  ├─ Semantic search
  └─ Cross-encoder reranking
  ↓
ChromaDB (10,003 documents) ✅ Tested
```

---

## 🎯 Test Results Summary

**Current Status:**
- ✅ 21/29 tests pass (unit + integration tests)
- ⚠️ 8/29 tests skipped (database tests - ChromaDB version mismatch outside venv)

**When run in venv:** All tests should pass!

**What's Working:**
- ✅ Smart Retriever (5/5 tests pass)
- ✅ Contextual Retriever (7/7 tests pass)
- ✅ Combined Retrievers (2/2 tests pass)
- ✅ Pipeline Configuration (5/5 tests pass)
- ✅ Database validation (2/2 basic tests pass)

---

## 🐛 Troubleshooting

### Tests fail with "No module named pytest"
```bash
# Activate venv first!
.\venv\Scripts\activate
pip install pytest pytest-cov
```

### Tests fail with ChromaDB errors
**Solution:** Always activate venv before running tests!
```bash
.\venv\Scripts\activate
python run_comprehensive_tests.py fast
```

Your venv has the correct ChromaDB version that works with your database.

### "Vector database not found" (for E2E tests)
The database exists, this error means it couldn't be accessed. Make sure:
1. Venv is activated
2. Running from project root directory

### "OPENAI_API_KEY not set" (for E2E tests)
```bash
# Add to .env file
echo "OPENAI_API_KEY=sk-your-key" >> .env
```

### Emojis causing errors on Windows
The test runner handles this automatically. If you see encoding errors, it means you're running tests incorrectly (not using the provided scripts).

---

## 📈 Coverage Reports

### Generate HTML Coverage Report
```bash
run_tests_venv.bat coverage-html
```

Then open `htmlcov/index.html` in your browser to see:
- Which lines are tested (green)
- Which lines need testing (red)
- Overall coverage percentage

**Coverage Goal:** >90%

**Current Coverage:**
- Smart Retriever: 74%
- Contextual Retriever: 80%
- Combined Pipeline: 48%
- Overall: ~60% (improving)

---

## 🎓 Understanding the Tests

### Unit Tests
**Fast, isolated, mocked dependencies**
- Smart retriever logic
- Contextual retriever logic
- No external API calls
- No database needed

**Run with:**
```bash
pytest -m unit -v
```

### Integration Tests
**Medium speed, multiple components working together**
- Layer combinations
- Pipeline configuration
- Component interactions
- Mocked external dependencies

**Run with:**
```bash
pytest -m integration -v
```

### E2E Tests
**Slow, expensive, uses real API and database**
- Full pipeline with real components
- Real OpenAI API calls (costs money!)
- Real ChromaDB queries
- Complete user scenarios

**Run with:**
```bash
pytest -m e2e -v
```

⚠️ **Warning:** E2E tests use your OpenAI API key and cost money!

---

## 🚦 CI/CD Integration

### Example GitHub Actions Workflow

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt

      - name: Run tests
        run: |
          source venv/bin/activate
          pytest -m "not e2e" --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 📚 Additional Information

### Test Markers
Tests are organized with pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Long-running tests

### Fixtures
Common test fixtures available in `conftest.py`:

- `project_root` - Project root directory
- `mock_env_vars` - Mocked environment variables
- `sample_documents` - Sample banking documents
- `sample_queries` - Test queries
- `mock_chroma_collection` - Mocked ChromaDB
- `skip_if_no_api_key` - Skip if no API key
- `skip_if_no_vector_db` - Skip if no database

### Writing New Tests

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
def test_my_feature(mock_env_vars):
    """Test description"""
    # Arrange
    # Act
    # Assert
    pass
```

---

## ✨ Quick Reference Card

**Most Common Commands:**

```bash
# 1. Activate venv
.\venv\Scripts\activate

# 2. Run all tests with coverage
run_tests_venv.bat coverage

# 3. Run fast tests (no E2E)
run_tests_venv.bat fast

# 4. Test specific component
run_tests_venv.bat smart
run_tests_venv.bat contextual

# 5. Generate HTML coverage
run_tests_venv.bat coverage-html
```

**Remember:** ALWAYS activate venv first!

---

## 🎉 Summary

You have a comprehensive test suite covering:
- ✅ Database validation (10,003 documents)
- ✅ Smart retriever (LLM disambiguation)
- ✅ Contextual retriever (query reformulation)
- ✅ Combined retriever layers
- ✅ Pipeline configuration
- ✅ End-to-end functionality

**Total:** 29 tests across 6 test classes

**Run tests:** `run_tests_venv.bat coverage`

**Check coverage:** Open `htmlcov/index.html` after running coverage tests

---

**Last Updated:** 2026-01-02
**Python Version:** 3.11
**Test Framework:** pytest 9.0.2
**ChromaDB Version:** 0.4.x (in venv) / 0.5.5 (system)
