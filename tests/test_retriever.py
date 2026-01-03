"""
Unit tests for the KnowledgeBaseRetriever module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.retrieval.retriever import KnowledgeBaseRetriever


class TestKnowledgeBaseRetriever:
    """Test suite for KnowledgeBaseRetriever"""

    @pytest.fixture
    def mock_vector_db(self, tmp_path):
        """Create a mock vector database path"""
        db_path = tmp_path / "test_vector_db"
        db_path.mkdir()
        return str(db_path)

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        mock = Mock()
        mock.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        return mock

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client"""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['Test document 1', 'Test document 2', 'Test document 3']],
            'metadatas': [[
                {'category': 'card_payment'},
                {'category': 'transfer_timing'},
                {'category': 'refund'}
            ]],
            'distances': [[0.2, 0.3, 0.4]]
        }

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        return mock_client, mock_collection

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    @patch('src.retrieval.retriever.CrossEncoder')
    def test_init_with_reranking(self, mock_cross_encoder, mock_openai,
                                  mock_chroma, mock_vector_db):
        """Test initialization with reranking enabled"""
        retriever = KnowledgeBaseRetriever(
            mock_vector_db,
            use_reranking=True
        )

        assert retriever.use_reranking is True
        assert retriever.rerank_multiplier == 2
        assert retriever.embedding_model == "text-embedding-3-small"
        mock_cross_encoder.assert_called_once_with('cross-encoder/ms-marco-TinyBERT-L-2-v2')

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    def test_init_without_reranking(self, mock_openai, mock_chroma, mock_vector_db):
        """Test initialization without reranking"""
        retriever = KnowledgeBaseRetriever(
            mock_vector_db,
            use_reranking=False
        )

        assert retriever.use_reranking is False
        assert retriever.reranker is None

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    def test_get_embedding(self, mock_openai_class, mock_chroma,
                          mock_vector_db, mock_openai_client):
        """Test embedding generation"""
        mock_openai_class.return_value = mock_openai_client

        retriever = KnowledgeBaseRetriever(mock_vector_db, use_reranking=False)
        embedding = retriever.get_embedding("test query")

        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        mock_openai_client.embeddings.create.assert_called_once()

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    def test_retrieve_without_reranking(self, mock_openai_class, mock_chroma_class,
                                        mock_vector_db, mock_openai_client,
                                        mock_chroma_client):
        """Test retrieval without reranking"""
        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value, mock_collection = mock_chroma_client

        retriever = KnowledgeBaseRetriever(mock_vector_db, use_reranking=False)
        results = retriever.retrieve("test query", n_results=3)

        assert 'documents' in results
        assert 'metadatas' in results
        assert 'distances' in results
        assert 'query' in results
        assert results['reranked'] is False
        assert len(results['documents']) == 3
        assert results['query'] == "test query"

        # Verify ChromaDB was queried with correct n_results
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        assert call_args[1]['n_results'] == 3

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    @patch('src.retrieval.retriever.CrossEncoder')
    def test_retrieve_with_reranking(self, mock_cross_encoder_class,
                                     mock_openai_class, mock_chroma_class,
                                     mock_vector_db, mock_openai_client,
                                     mock_chroma_client):
        """Test retrieval with reranking"""
        # Setup mocks
        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value, mock_collection = mock_chroma_client

        # Mock reranker to return scores
        mock_reranker = Mock()
        mock_reranker.predict.return_value = [0.9, 0.5, 0.7, 0.4, 0.6, 0.3]
        mock_cross_encoder_class.return_value = mock_reranker

        # Mock ChromaDB to return 6 results (2x multiplier)
        mock_collection.query.return_value = {
            'documents': [['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6']],
            'metadatas': [[
                {'category': 'cat1'}, {'category': 'cat2'},
                {'category': 'cat3'}, {'category': 'cat4'},
                {'category': 'cat5'}, {'category': 'cat6'}
            ]],
            'distances': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        }

        retriever = KnowledgeBaseRetriever(mock_vector_db, use_reranking=True)
        results = retriever.retrieve("test query", n_results=3)

        assert results['reranked'] is True
        assert 'rerank_scores' in results
        assert len(results['documents']) == 3

        # Verify documents are reordered by rerank scores (highest first)
        # Scores are [0.9, 0.5, 0.7, 0.4, 0.6, 0.3]
        # Top 3 should be indices 0, 2, 4 (scores 0.9, 0.7, 0.6)
        assert results['documents'][0] == 'doc1'  # highest score (0.9)
        assert results['documents'][1] == 'doc3'  # second highest (0.7)
        assert results['documents'][2] == 'doc5'  # third highest (0.6)

        # Verify ChromaDB was queried with 2x n_results
        call_args = mock_collection.query.call_args
        assert call_args[1]['n_results'] == 6

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    def test_format_context(self, mock_openai_class, mock_chroma_class,
                           mock_vector_db, mock_openai_client, mock_chroma_client):
        """Test context formatting"""
        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value, _ = mock_chroma_client

        retriever = KnowledgeBaseRetriever(mock_vector_db, use_reranking=False)

        test_results = {
            'documents': ['Document one', 'Document two'],
            'metadatas': [
                {'category': 'card_payment'},
                {'category': 'transfer_timing'}
            ],
            'distances': [0.2, 0.3]
        }

        context = retriever.format_context(test_results)

        # Verify format includes all expected elements
        assert '[Document 1]' in context
        assert '[Document 2]' in context
        assert 'Category: card_payment' in context
        assert 'Category: transfer_timing' in context
        assert 'Relevance: 0.80' in context  # 1 - 0.2
        assert 'Relevance: 0.70' in context  # 1 - 0.3
        assert 'Document one' in context
        assert 'Document two' in context

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    def test_retrieve_empty_query(self, mock_openai_class, mock_chroma_class,
                                  mock_vector_db, mock_openai_client, mock_chroma_client):
        """Test retrieval with empty query"""
        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value, mock_collection = mock_chroma_client

        retriever = KnowledgeBaseRetriever(mock_vector_db, use_reranking=False)

        # Should still work, just retrieve documents
        results = retriever.retrieve("", n_results=3)
        assert 'documents' in results
        assert results['query'] == ""

    @patch('src.retrieval.retriever.chromadb.PersistentClient')
    @patch('src.retrieval.retriever.OpenAI')
    @patch('src.retrieval.retriever.CrossEncoder')
    def test_retrieve_with_custom_multiplier(self, mock_cross_encoder_class,
                                             mock_openai_class, mock_chroma_class,
                                             mock_vector_db, mock_openai_client,
                                             mock_chroma_client):
        """Test retrieval with custom rerank multiplier"""
        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value, mock_collection = mock_chroma_client

        # Create retriever with 3x multiplier
        retriever = KnowledgeBaseRetriever(
            mock_vector_db,
            use_reranking=True,
            rerank_multiplier=3
        )

        results = retriever.retrieve("test query", n_results=2)

        # Verify ChromaDB was queried with 3x n_results
        call_args = mock_collection.query.call_args
        assert call_args[1]['n_results'] == 6  # 2 * 3
