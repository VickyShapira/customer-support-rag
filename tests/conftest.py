"""
Pytest configuration and shared fixtures
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock
from dotenv import load_dotenv

# Load environment variables from .env file (override=True to use .env over system env vars)
load_dotenv(override=True)

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end (requiring real API/DB access)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Get data directory path"""
    return project_root / "data"


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get test data directory"""
    test_dir = project_root / "tests" / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")
    return {
        "OPENAI_API_KEY": "test-api-key-12345"
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "text": "To reset your password, go to Settings and click 'Forgot Password'.",
            "category": "passcode_forgotten",
            "metadata": {"source": "FAQ", "id": 1}
        },
        {
            "text": "Card payments can be declined due to insufficient funds or security holds.",
            "category": "declined_card_payment",
            "metadata": {"source": "FAQ", "id": 2}
        },
        {
            "text": "International transfers typically take 3-5 business days to process.",
            "category": "transfer_timing",
            "metadata": {"source": "FAQ", "id": 3}
        },
        {
            "text": "You can top up your account using a debit card or bank transfer.",
            "category": "top_up_by_card_charge",
            "metadata": {"source": "FAQ", "id": 4}
        },
        {
            "text": "Virtual cards can be created instantly through the mobile app.",
            "category": "getting_virtual_card",
            "metadata": {"source": "FAQ", "id": 5}
        }
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return [
        {
            "query": "How do I reset my password?",
            "expected_category": "passcode_forgotten",
            "has_negation": False
        },
        {
            "query": "My card was declined",
            "expected_category": "declined_card_payment",
            "has_negation": False
        },
        {
            "query": "How long does a transfer take?",
            "expected_category": "transfer_timing",
            "has_negation": False
        },
        {
            "query": "I can't use my card",
            "expected_category": "card_not_working",
            "has_negation": True
        }
    ]


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for testing"""
    return {
        'documents': [
            'To reset your password, go to Settings.',
            'Card payments may be declined for various reasons.',
            'Transfers take 3-5 business days.'
        ],
        'metadatas': [
            {'category': 'passcode_forgotten', 'id': 1},
            {'category': 'declined_card_payment', 'id': 2},
            {'category': 'transfer_timing', 'id': 3}
        ],
        'distances': [0.15, 0.25, 0.35],
        'query': 'How do I reset my password?',
        'reranked': False
    }


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing"""
    return [
        {"role": "user", "content": "What are transfer fees?"},
        {"role": "assistant", "content": "Transfer fees vary by type. Domestic transfers are typically $2-5."},
        {"role": "user", "content": "How long do they take?"},
        {"role": "assistant", "content": "Domestic transfers take 1-3 business days."}
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "To reset your password, please go to the login page and click on 'Forgot Password'."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }


@pytest.fixture
def mock_embedding_response():
    """Mock OpenAI embedding response"""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * 1536,
                "index": 0
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection"""
    mock = Mock()
    mock.query.return_value = {
        'documents': [['doc1', 'doc2', 'doc3']],
        'metadatas': [[
            {'category': 'cat1'},
            {'category': 'cat2'},
            {'category': 'cat3'}
        ]],
        'distances': [[0.2, 0.3, 0.4]],
        'ids': [['id1', 'id2', 'id3']]
    }
    mock.count.return_value = 10003
    mock.name = "banking_support"
    return mock


@pytest.fixture
def performance_threshold():
    """Performance thresholds for testing"""
    return {
        'retrieval_time': 2.0,  # seconds
        'generation_time': 5.0,  # seconds
        'total_query_time': 7.0,  # seconds
        'streaming_first_chunk': 1.0,  # seconds
        'min_accuracy_top1': 0.70,
        'min_accuracy_top3': 0.85,
        'min_accuracy_top5': 0.90
    }


@pytest.fixture(scope="session")
def skip_if_no_api_key():
    """Skip test if OpenAI API key is not available"""
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture(scope="session")
def skip_if_no_vector_db(project_root):
    """Skip test if vector database doesn't exist"""
    db_path = project_root / "data" / "vector_db"
    if not db_path.exists():
        pytest.skip("Vector database not found")


# Pytest hooks for better test output
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Auto-mark tests based on file name
        if "test_e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
        elif "test_rag_pipeline" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_retriever" in item.nodeid or "test_generator" in item.nodeid:
            item.add_marker(pytest.mark.unit)


def pytest_report_header(config):
    """Add custom header to pytest output"""
    return [
        "Customer Support RAG System - Test Suite",
        "=" * 60,
        f"Project: Banking Customer Support Assistant",
        f"Python path includes: {sys.path[0]}",
    ]


def pytest_runtest_setup(item):
    """Setup for each test"""
    # Add any per-test setup here
    pass


def pytest_runtest_teardown(item):
    """Teardown for each test"""
    # Add any per-test cleanup here
    pass
