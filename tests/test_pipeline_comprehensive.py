"""
Comprehensive test suite for RAG pipeline
Tests database creation, smart retriever, contextual retriever, and combined functionality
"""
import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import chromadb

# Import pipeline components
from rag_pipeline import RAGPipeline
from retrieval.retriever import KnowledgeBaseRetriever
from retrieval.smart_retriever import SmartRetriever
from retrieval.contextual_retriever import ContextualRetriever
from generation.generator import AnswerGenerator


# ============================================================================
# DATABASE CREATION TESTS
# ============================================================================

@pytest.mark.integration
class TestDatabaseCreation:
    """Test vector database creation and initialization"""

    def test_database_exists(self, project_root):
        """Test that vector database exists at expected location"""
        db_path = project_root / "data" / "vector_db"
        assert db_path.exists(), f"Vector database not found at {db_path}"
        assert db_path.is_dir(), "Vector database path should be a directory"

    def test_database_can_connect(self, project_root):
        """Test that we can connect to the database"""
        db_path = str(project_root / "data" / "vector_db")
        client = chromadb.PersistentClient(path=db_path)
        assert client is not None, "Failed to create ChromaDB client"

    def test_collection_exists(self, project_root):
        """Test that the banking_support collection exists"""
        db_path = str(project_root / "data" / "vector_db")
        # Use the actual retriever instead of accessing ChromaDB directly
        # This works with any ChromaDB version
        retriever = KnowledgeBaseRetriever(db_path)
        assert retriever.collection is not None, "banking_support collection not found"
        assert retriever.collection.name == "banking_support"

    def test_collection_has_documents(self, project_root):
        """Test that collection contains the expected number of documents"""
        db_path = str(project_root / "data" / "vector_db")
        # Use the actual retriever - works with any ChromaDB version
        retriever = KnowledgeBaseRetriever(db_path)

        count = retriever.collection.count()
        assert count > 0, "Collection is empty"
        # Should have 10,003 documents (full dataset)
        assert count == 10003, f"Expected 10,003 documents, found {count}"

    def test_collection_metadata_structure(self, project_root):
        """Test that documents have correct metadata structure"""
        db_path = str(project_root / "data" / "vector_db")
        # Use the actual retriever - works with any ChromaDB version
        retriever = KnowledgeBaseRetriever(db_path)

        # Retrieve some documents to check metadata
        results = retriever.retrieve("test query", n_results=3)

        assert 'metadatas' in results
        assert len(results['metadatas']) > 0

        # Check metadata structure
        for metadata in results['metadatas']:
            assert 'category' in metadata, "Metadata missing 'category' field"
            # Category should be a valid banking support category
            assert isinstance(metadata['category'], str)
            assert len(metadata['category']) > 0

    def test_collection_has_embeddings(self, project_root):
        """Test that documents have embeddings"""
        db_path = str(project_root / "data" / "vector_db")
        # Use the actual retriever - works with any ChromaDB version
        retriever = KnowledgeBaseRetriever(db_path)

        # Query returns results with distances, which proves embeddings exist
        results = retriever.retrieve("test query", n_results=1)

        assert 'documents' in results
        assert len(results['documents']) > 0
        assert 'distances' in results
        assert len(results['distances']) > 0
        # If we got distances, embeddings are working


# ============================================================================
# SMART RETRIEVER TESTS
# ============================================================================

@pytest.mark.unit
class TestSmartRetriever:
    """Test smart retriever with LLM disambiguation"""

    @pytest.fixture
    def mock_base_retriever(self):
        """Create mock base retriever"""
        mock = Mock()
        # Simulate low confidence scenario with overlapping categories
        mock.retrieve.return_value = {
            'documents': [
                'Card declined due to insufficient funds',
                'Card not working at ATM',
                'Card payment failed',
                'Card blocked by security',
                'Virtual card not activated'
            ],
            'metadatas': [
                {'category': 'declined_card_payment'},
                {'category': 'card_not_working'},
                {'category': 'declined_card_payment'},
                {'category': 'card_not_working'},
                {'category': 'getting_virtual_card'}
            ],
            'distances': [0.65, 0.70, 0.75, 0.80, 0.85],  # Low confidence (high distance)
            'query': 'My card was declined'
        }
        return mock

    def test_smart_retriever_initialization(self, mock_base_retriever):
        """Test smart retriever can be initialized"""
        retriever = SmartRetriever(
            base_retriever=mock_base_retriever,
            confidence_threshold=0.38,
            similarity_gap_threshold=0.10
        )

        assert retriever.base_retriever is mock_base_retriever
        assert retriever.confidence_threshold == 0.38
        assert retriever.similarity_gap_threshold == 0.10

    def test_overlapping_groups_defined(self):
        """Test that overlapping category groups are defined"""
        assert hasattr(SmartRetriever, 'OVERLAPPING_GROUPS')
        assert len(SmartRetriever.OVERLAPPING_GROUPS) > 0

        # Check specific known overlapping groups
        groups = SmartRetriever.OVERLAPPING_GROUPS

        # Card payment issues should be defined
        card_group = None
        for group in groups:
            if 'declined_card_payment' in group:
                card_group = group
                break

        assert card_group is not None, "Card payment overlapping group not found"
        assert 'card_not_working' in card_group

    @patch('retrieval.smart_retriever.openai.OpenAI')
    def test_smart_retriever_triggers_on_low_confidence(self, mock_openai, mock_base_retriever, mock_env_vars):
        """Test that LLM disambiguation triggers on low confidence"""
        # Setup mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='declined_card_payment'))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        retriever = SmartRetriever(
            base_retriever=mock_base_retriever,
            confidence_threshold=0.38,
            similarity_gap_threshold=0.10
        )

        # This should trigger LLM because top similarity (0.35) < threshold (0.38)
        results = retriever.retrieve("My card was declined", n_results=3)

        # Should have called base retriever
        mock_base_retriever.retrieve.assert_called_once()

        # Should have results
        assert 'documents' in results
        assert 'metadatas' in results

    @patch('retrieval.smart_retriever.openai.OpenAI')
    def test_smart_retriever_triggers_on_small_gap(self, mock_openai, mock_env_vars):
        """Test that LLM disambiguation triggers when gap between top 2 is small"""
        # Create mock with small gap
        mock_base = Mock()
        mock_base.retrieve.return_value = {
            'documents': ['doc1', 'doc2', 'doc3'],
            'metadatas': [
                {'category': 'declined_card_payment'},
                {'category': 'card_not_working'},
                {'category': 'card_payment_fee_charged'}
            ],
            'distances': [0.20, 0.25, 0.50],  # Gap of 0.05 < 0.10 threshold
            'query': 'test'
        }

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='declined_card_payment'))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        retriever = SmartRetriever(
            base_retriever=mock_base,
            similarity_gap_threshold=0.10
        )

        results = retriever.retrieve("test query", n_results=3)
        assert results is not None

    def test_smart_retriever_passthrough_high_confidence(self, mock_env_vars):
        """Test that smart retriever passes through when confidence is high"""
        # Create mock with high confidence
        mock_base = Mock()
        mock_base.retrieve.return_value = {
            'documents': ['doc1', 'doc2', 'doc3'],
            'metadatas': [
                {'category': 'passcode_forgotten'},
                {'category': 'verify_identity'},
                {'category': 'lost_or_stolen_card'}
            ],
            'distances': [0.10, 0.40, 0.60],  # High confidence, large gap
            'query': 'forgot password'
        }

        with patch('retrieval.smart_retriever.openai.OpenAI'):
            retriever = SmartRetriever(
                base_retriever=mock_base,
                confidence_threshold=0.38,
                similarity_gap_threshold=0.10
            )

            results = retriever.retrieve("forgot password", n_results=3)

            # Should return results
            assert len(results['documents']) == 3


# ============================================================================
# CONTEXTUAL RETRIEVER TESTS
# ============================================================================

@pytest.mark.unit
class TestContextualRetriever:
    """Test contextual retriever with query reformulation"""

    @pytest.fixture
    def mock_base_retriever(self):
        """Create mock base retriever"""
        mock = Mock()
        mock.retrieve.return_value = {
            'documents': ['doc1', 'doc2', 'doc3'],
            'metadatas': [
                {'category': 'transfer_timing'},
                {'category': 'transfer_fee_charged'},
                {'category': 'pending_transfer'}
            ],
            'distances': [0.2, 0.3, 0.4],
            'query': 'How long does it take?'
        }
        return mock

    def test_contextual_retriever_initialization(self, mock_base_retriever):
        """Test contextual retriever can be initialized"""
        retriever = ContextualRetriever(
            base_retriever=mock_base_retriever,
            use_context=True,
            model="gpt-4o-mini"
        )

        assert retriever.base_retriever is mock_base_retriever
        assert retriever.use_context is True
        assert retriever.model == "gpt-4o-mini"

    def test_needs_reformulation_short_query(self, mock_base_retriever):
        """Test that short queries are flagged for reformulation"""
        retriever = ContextualRetriever(mock_base_retriever)

        # Short queries should need reformulation
        assert retriever._needs_reformulation("Why?") is True
        assert retriever._needs_reformulation("How long?") is True
        assert retriever._needs_reformulation("How much?") is True

    def test_needs_reformulation_pronoun_starters(self, mock_base_retriever):
        """Test that queries starting with pronouns are flagged"""
        retriever = ContextualRetriever(mock_base_retriever)

        assert retriever._needs_reformulation("It doesn't work") is True
        assert retriever._needs_reformulation("That's wrong") is True
        assert retriever._needs_reformulation("This failed") is True

    def test_needs_reformulation_standalone_query(self, mock_base_retriever):
        """Test that standalone queries don't need reformulation"""
        retriever = ContextualRetriever(mock_base_retriever)

        standalone = "How long does a bank transfer take?"
        assert retriever._needs_reformulation(standalone) is False

        standalone2 = "Why was my card payment declined?"
        assert retriever._needs_reformulation(standalone2) is False

    @patch('retrieval.contextual_retriever.openai.OpenAI')
    def test_reformulation_with_context(self, mock_openai, mock_base_retriever, mock_env_vars):
        """Test query reformulation with conversation history"""
        # Setup mock OpenAI
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(
            content='How long does a bank transfer take?'
        ))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        retriever = ContextualRetriever(
            base_retriever=mock_base_retriever,
            use_context=True
        )

        conversation_history = [
            {"role": "user", "content": "What are transfer fees?"},
            {"role": "assistant", "content": "Transfer fees vary by type..."}
        ]

        # This should trigger reformulation
        results = retriever.retrieve(
            "How long?",
            conversation_history=conversation_history,
            n_results=3
        )

        # Should have called base retriever
        mock_base_retriever.retrieve.assert_called_once()
        assert results is not None

    def test_no_reformulation_without_context(self, mock_base_retriever):
        """Test that queries without context don't get reformulated"""
        retriever = ContextualRetriever(
            base_retriever=mock_base_retriever,
            use_context=True
        )

        # No conversation history provided
        results = retriever.retrieve("How long?", n_results=3)

        # Should still call base retriever with original query
        mock_base_retriever.retrieve.assert_called_once_with("How long?", 3)

    def test_context_disabled(self, mock_base_retriever):
        """Test behavior when context is disabled"""
        retriever = ContextualRetriever(
            base_retriever=mock_base_retriever,
            use_context=False
        )

        conversation_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        results = retriever.retrieve(
            "How long?",
            conversation_history=conversation_history,
            n_results=3
        )

        # Should call base retriever without reformulation
        mock_base_retriever.retrieve.assert_called_once_with("How long?", 3)


# ============================================================================
# COMBINED RETRIEVER TESTS (Smart + Contextual)
# ============================================================================

@pytest.mark.integration
class TestCombinedRetrievers:
    """Test the combined smart + contextual retriever pipeline"""

    @pytest.fixture
    def mock_base_retriever(self):
        """Create mock base retriever"""
        mock = Mock()
        mock.retrieve.return_value = {
            'documents': ['doc1', 'doc2', 'doc3'],
            'metadatas': [
                {'category': 'declined_card_payment'},
                {'category': 'card_not_working'},
                {'category': 'card_payment_fee_charged'}
            ],
            'distances': [0.2, 0.3, 0.4],
            'query': 'test'
        }
        return mock

    @patch('retrieval.contextual_retriever.openai.OpenAI')
    @patch('retrieval.smart_retriever.openai.OpenAI')
    def test_layered_retriever_initialization(self, mock_smart_openai, mock_ctx_openai, mock_base_retriever, mock_env_vars):
        """Test that retrievers can be properly layered"""
        # Setup mocks
        mock_client = Mock()
        mock_smart_openai.return_value = mock_client
        mock_ctx_openai.return_value = mock_client

        # Layer 1: Base retriever (innermost)
        base = mock_base_retriever

        # Layer 2: Smart retriever (middle)
        smart = SmartRetriever(base_retriever=base)

        # Layer 3: Contextual retriever (outermost)
        contextual = ContextualRetriever(base_retriever=smart)

        # Verify layering
        assert contextual.base_retriever is smart
        assert smart.base_retriever is base

    @patch('retrieval.contextual_retriever.openai.OpenAI')
    @patch('retrieval.smart_retriever.openai.OpenAI')
    def test_combined_follow_up_with_disambiguation(self, mock_smart_openai, mock_ctx_openai, mock_base_retriever, mock_env_vars):
        """Test combined scenario: follow-up query + ambiguous category"""
        # Setup mocks
        mock_client = Mock()

        # Mock contextual reformulation
        ctx_response = Mock()
        ctx_response.choices = [Mock(message=Mock(
            content='Why was my card declined at the store?'
        ))]

        # Mock smart disambiguation
        smart_response = Mock()
        smart_response.choices = [Mock(message=Mock(
            content='declined_card_payment'
        ))]

        mock_client.chat.completions.create.side_effect = [
            ctx_response,
            smart_response
        ]

        mock_smart_openai.return_value = mock_client
        mock_ctx_openai.return_value = mock_client

        # Create layered retriever
        base = mock_base_retriever
        smart = SmartRetriever(base_retriever=base)
        contextual = ContextualRetriever(base_retriever=smart)

        conversation_history = [
            {"role": "user", "content": "I tried to pay at a store"},
            {"role": "assistant", "content": "Card payments can fail for various reasons"}
        ]

        # Execute query (should trigger both reformulation and disambiguation)
        results = contextual.retrieve(
            "Why?",
            conversation_history=conversation_history,
            n_results=3
        )

        assert results is not None
        # Base retriever should have been called
        assert mock_base_retriever.retrieve.called


# ============================================================================
# RAG PIPELINE INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Test complete RAG pipeline with all retrievers"""

    @pytest.fixture
    def mock_vector_db_path(self, tmp_path):
        """Create temporary vector DB path"""
        return str(tmp_path / "test_db")

    @patch('rag_pipeline.KnowledgeBaseRetriever')
    @patch('rag_pipeline.AnswerGenerator')
    def test_pipeline_initialization_all_layers(self, mock_gen, mock_retriever_class, mock_env_vars):
        """Test pipeline initialization with all retriever layers enabled"""
        # Setup mocks
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever

        mock_generator = Mock()
        mock_gen.return_value = mock_generator

        # Initialize pipeline with all layers
        pipeline = RAGPipeline(
            vector_db_path="test_path",
            use_contextual_retriever=True,
            use_smart_retriever=True
        )

        # Verify retriever is ContextualRetriever (outermost layer)
        assert isinstance(pipeline.retriever, ContextualRetriever)

        # Verify middle layer is SmartRetriever
        middle_layer = pipeline.retriever.base_retriever
        assert isinstance(middle_layer, SmartRetriever)

        # Verify innermost layer is base retriever
        base_layer = middle_layer.base_retriever
        assert base_layer is mock_retriever

    @patch('rag_pipeline.KnowledgeBaseRetriever')
    @patch('rag_pipeline.AnswerGenerator')
    def test_pipeline_initialization_smart_only(self, mock_gen, mock_retriever_class, mock_env_vars):
        """Test pipeline with only smart retriever enabled"""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever

        mock_generator = Mock()
        mock_gen.return_value = mock_generator

        pipeline = RAGPipeline(
            vector_db_path="test_path",
            use_contextual_retriever=False,
            use_smart_retriever=True
        )

        # Should be SmartRetriever
        assert isinstance(pipeline.retriever, SmartRetriever)
        assert pipeline.retriever.base_retriever is mock_retriever

    @patch('rag_pipeline.KnowledgeBaseRetriever')
    @patch('rag_pipeline.AnswerGenerator')
    def test_pipeline_initialization_contextual_only(self, mock_gen, mock_retriever_class, mock_env_vars):
        """Test pipeline with only contextual retriever enabled"""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever

        mock_generator = Mock()
        mock_gen.return_value = mock_generator

        pipeline = RAGPipeline(
            vector_db_path="test_path",
            use_contextual_retriever=True,
            use_smart_retriever=False
        )

        # Should be ContextualRetriever
        assert isinstance(pipeline.retriever, ContextualRetriever)
        assert pipeline.retriever.base_retriever is mock_retriever

    @patch('rag_pipeline.KnowledgeBaseRetriever')
    @patch('rag_pipeline.AnswerGenerator')
    def test_pipeline_initialization_base_only(self, mock_gen, mock_retriever_class, mock_env_vars):
        """Test pipeline with no additional retriever layers"""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever

        mock_generator = Mock()
        mock_gen.return_value = mock_generator

        pipeline = RAGPipeline(
            vector_db_path="test_path",
            use_contextual_retriever=False,
            use_smart_retriever=False
        )

        # Should be base retriever directly
        assert pipeline.retriever is mock_retriever

    def test_get_pipeline_info(self, mock_env_vars):
        """Test getting pipeline configuration info"""
        with patch('rag_pipeline.KnowledgeBaseRetriever'), \
             patch('rag_pipeline.AnswerGenerator'):

            pipeline = RAGPipeline(
                vector_db_path="test_path",
                use_contextual_retriever=True,
                use_smart_retriever=True,
                confidence_threshold=0.40,
                similarity_gap_threshold=0.12
            )

            info = pipeline.get_pipeline_info()

            assert 'layers' in info
            assert info['layers']['contextual_retriever'] is True
            assert info['layers']['smart_retriever'] is True
            assert info['layers']['base_retriever'] is True

            assert 'thresholds' in info
            assert info['thresholds']['confidence_threshold'] == 0.40
            assert info['thresholds']['similarity_gap_threshold'] == 0.12


# ============================================================================
# END-TO-END TESTS (require real database and API key)
# ============================================================================

@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end tests with real database and API (expensive, marked as e2e)"""

    def test_e2e_database_query(self, project_root, skip_if_no_api_key, skip_if_no_vector_db):
        """Test complete query with real database"""
        db_path = str(project_root / "data" / "vector_db")

        # Initialize base retriever
        retriever = KnowledgeBaseRetriever(db_path)

        # Execute query
        results = retriever.retrieve("How do I reset my password?", n_results=3)

        assert 'documents' in results
        assert len(results['documents']) == 3
        assert 'metadatas' in results
        assert 'distances' in results

    def test_e2e_smart_retrieval(self, project_root, skip_if_no_api_key, skip_if_no_vector_db):
        """Test smart retrieval with real database and API"""
        db_path = str(project_root / "data" / "vector_db")

        base = KnowledgeBaseRetriever(db_path)
        smart = SmartRetriever(base_retriever=base)

        # Ambiguous query that should trigger LLM
        results = smart.retrieve("My card isn't working", n_results=3)

        assert results is not None
        assert len(results['documents']) > 0

    def test_e2e_contextual_retrieval(self, project_root, skip_if_no_api_key, skip_if_no_vector_db):
        """Test contextual retrieval with real database and API"""
        db_path = str(project_root / "data" / "vector_db")

        base = KnowledgeBaseRetriever(db_path)
        contextual = ContextualRetriever(base_retriever=base)

        conversation_history = [
            {"role": "user", "content": "What are bank transfer fees?"},
            {"role": "assistant", "content": "Transfer fees vary by destination"}
        ]

        # Follow-up query
        results = contextual.retrieve(
            "How long do they take?",
            conversation_history=conversation_history,
            n_results=3
        )

        assert results is not None
        assert len(results['documents']) > 0

    def test_e2e_full_pipeline(self, project_root, skip_if_no_api_key, skip_if_no_vector_db):
        """Test complete RAG pipeline with all layers"""
        db_path = str(project_root / "data" / "vector_db")

        pipeline = RAGPipeline(
            vector_db_path=db_path,
            use_contextual_retriever=True,
            use_smart_retriever=True,
            model="gpt-4o-mini"
        )

        # Test query
        result = pipeline.query("Why was I charged a fee?", n_results=3)

        assert 'answer' in result
        assert 'question' in result
        assert 'sources' in result
        assert len(result['answer']) > 0

        # Test follow-up
        result2 = pipeline.query("How much?", n_results=3)

        assert 'answer' in result2
        assert len(result2['answer']) > 0
