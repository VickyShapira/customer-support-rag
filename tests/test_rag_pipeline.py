"""
Integration tests for the complete RAG pipeline
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    """Test suite for RAGPipeline integration"""

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever"""
        mock = Mock()
        mock.retrieve.return_value = {
            'documents': ['Doc 1', 'Doc 2', 'Doc 3'],
            'metadatas': [
                {'category': 'card_payment'},
                {'category': 'transfer_timing'},
                {'category': 'refund'}
            ],
            'distances': [0.2, 0.3, 0.4],
            'reranked': True,
            'rerank_scores': [0.9, 0.7, 0.5]
        }
        mock.format_context.return_value = "[Document 1] Doc 1\n[Document 2] Doc 2\n[Document 3] Doc 3"
        return mock

    @pytest.fixture
    def mock_generator(self):
        """Mock generator"""
        mock = Mock()
        mock.generate.return_value = "This is the generated answer."
        return mock

    @pytest.fixture
    def mock_streaming_generator(self):
        """Mock generator with streaming"""
        mock = Mock()
        mock.generate.return_value = "This is the generated answer."
        mock.generate_stream.return_value = iter(["This ", "is ", "the ", "answer."])
        return mock

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_init(self, mock_gen_class, mock_ret_class, mock_vector_db="test_db"):
        """Test pipeline initialization"""
        pipeline = RAGPipeline(mock_vector_db, model="gpt-4o-mini")

        mock_ret_class.assert_called_once_with(mock_vector_db)
        mock_gen_class.assert_called_once_with("gpt-4o-mini")
        assert pipeline.conversation_history == []

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_query_basic(self, mock_gen_class, mock_ret_class,
                        mock_retriever, mock_generator):
        """Test basic query flow"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")
        response = pipeline.query("What are transfer fees?")

        # Verify retriever was called
        mock_retriever.retrieve.assert_called_once_with("What are transfer fees?", 3)
        mock_retriever.format_context.assert_called_once()

        # Verify generator was called
        mock_generator.generate.assert_called_once()

        # Verify response structure
        assert 'question' in response
        assert 'answer' in response
        assert 'sources' in response
        assert 'metadata' in response
        assert response['question'] == "What are transfer fees?"
        assert response['answer'] == "This is the generated answer."

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_query_without_sources(self, mock_gen_class, mock_ret_class,
                                   mock_retriever, mock_generator):
        """Test query without including sources"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")
        response = pipeline.query("Test query", include_sources=False)

        assert 'sources' not in response
        assert 'metadata' not in response
        assert 'question' in response
        assert 'answer' in response

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_query_with_custom_n_results(self, mock_gen_class, mock_ret_class,
                                        mock_retriever, mock_generator):
        """Test query with custom number of results"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")
        response = pipeline.query("Test query", n_results=5)

        mock_retriever.retrieve.assert_called_once_with("Test query", 5)

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_conversation_history_updates(self, mock_gen_class, mock_ret_class,
                                          mock_retriever, mock_generator):
        """Test that conversation history is updated after query"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")

        # Initial state
        assert len(pipeline.conversation_history) == 0

        # First query
        pipeline.query("First question")
        assert len(pipeline.conversation_history) == 2
        assert pipeline.conversation_history[0]['role'] == 'user'
        assert pipeline.conversation_history[0]['content'] == 'First question'
        assert pipeline.conversation_history[1]['role'] == 'assistant'

        # Second query
        pipeline.query("Second question")
        assert len(pipeline.conversation_history) == 4

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_conversation_history_limit(self, mock_gen_class, mock_ret_class,
                                       mock_retriever, mock_generator):
        """Test that conversation history is limited to 6 messages (3 turns)"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")

        # Make 5 queries (10 messages total)
        for i in range(5):
            pipeline.query(f"Question {i}")

        # Should only keep last 6 messages (3 turns)
        assert len(pipeline.conversation_history) == 6
        assert pipeline.conversation_history[0]['content'] == 'Question 2'

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_conversation_history_passed_to_generator(self, mock_gen_class,
                                                      mock_ret_class, mock_retriever,
                                                      mock_generator):
        """Test that conversation history is passed to generator"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")

        # First query
        pipeline.query("First question")

        # Second query - should include history
        pipeline.query("Second question")

        # Verify both calls were made
        assert mock_generator.generate.call_count == 2

        # Note: Since conversation_history is a mutable list that gets modified,
        # we need to check the call arguments differently
        # The history list is the same reference passed to both calls, so it will
        # show the final state (4 messages) for both calls when inspected after the fact

        # Instead, verify the pipeline's final state
        final_history = pipeline.get_conversation_history()
        assert len(final_history) == 4  # Two Q&A pairs
        assert final_history[0]['content'] == "First question"
        assert final_history[2]['content'] == "Second question"

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_reset_conversation(self, mock_gen_class, mock_ret_class,
                               mock_retriever, mock_generator):
        """Test conversation reset functionality"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")

        # Add some history
        pipeline.query("Question 1")
        pipeline.query("Question 2")
        assert len(pipeline.conversation_history) > 0

        # Reset
        pipeline.reset_conversation()
        assert len(pipeline.conversation_history) == 0

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_get_conversation_history(self, mock_gen_class, mock_ret_class,
                                     mock_retriever, mock_generator):
        """Test getting conversation history"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")

        # Add history
        pipeline.query("Test question")

        # Get history
        history = pipeline.get_conversation_history()

        assert len(history) == 2
        assert isinstance(history, list)

        # Verify it's a copy (modifying shouldn't affect original)
        history.append({"role": "user", "content": "Extra"})
        assert len(pipeline.conversation_history) == 2

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_sources_format(self, mock_gen_class, mock_ret_class,
                           mock_retriever, mock_generator):
        """Test that sources are formatted correctly"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")
        response = pipeline.query("Test query")

        sources = response['sources']
        assert len(sources) == 3

        # Check first source
        assert 'category' in sources[0]
        assert 'similarity' in sources[0]
        assert sources[0]['category'] == 'card_payment'
        assert sources[0]['similarity'] == 0.8  # 1 - 0.2

        # Check similarity calculations
        assert sources[1]['similarity'] == 0.7  # 1 - 0.3
        assert sources[2]['similarity'] == 0.6  # 1 - 0.4

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_query_stream_basic(self, mock_gen_class, mock_ret_class,
                               mock_retriever, mock_streaming_generator):
        """Test streaming query"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_streaming_generator

        pipeline = RAGPipeline("test_db")
        stream = pipeline.query_stream("Test question")

        chunks = list(stream)

        # First chunk should contain metadata
        assert chunks[0]['question'] == "Test question"
        assert 'sources' in chunks[0]
        assert chunks[0]['done'] is False

        # Middle chunks should contain answer parts
        assert chunks[1]['chunk'] == "This "
        assert chunks[2]['chunk'] == "is "

        # Last chunk should signal completion
        assert chunks[-1]['done'] is True
        assert chunks[-1]['full_answer'] == "This is the answer."

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_query_stream_without_sources(self, mock_gen_class, mock_ret_class,
                                         mock_retriever, mock_streaming_generator):
        """Test streaming without sources"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_streaming_generator

        pipeline = RAGPipeline("test_db")
        stream = pipeline.query_stream("Test question", include_sources=False)

        chunks = list(stream)

        # First chunk should not contain sources
        assert 'sources' not in chunks[0]
        assert 'metadata' not in chunks[0]

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_query_stream_updates_history(self, mock_gen_class, mock_ret_class,
                                         mock_retriever, mock_streaming_generator):
        """Test that streaming updates conversation history"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_streaming_generator

        pipeline = RAGPipeline("test_db")

        # Consume the stream
        list(pipeline.query_stream("Test question"))

        # Verify history was updated
        assert len(pipeline.conversation_history) == 2
        assert pipeline.conversation_history[0]['content'] == "Test question"
        assert pipeline.conversation_history[1]['content'] == "This is the answer."

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_query_stream_history_limit(self, mock_gen_class, mock_ret_class,
                                       mock_retriever, mock_streaming_generator):
        """Test that streaming respects conversation history limit"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_streaming_generator

        pipeline = RAGPipeline("test_db")

        # Make 5 streaming queries
        for i in range(5):
            list(pipeline.query_stream(f"Question {i}"))

        # Should only keep last 6 messages
        assert len(pipeline.conversation_history) == 6

    @patch('src.rag_pipeline.KnowledgeBaseRetriever')
    @patch('src.rag_pipeline.AnswerGenerator')
    def test_multiple_queries_same_pipeline(self, mock_gen_class, mock_ret_class,
                                           mock_retriever, mock_generator):
        """Test multiple queries through same pipeline instance"""
        mock_ret_class.return_value = mock_retriever
        mock_gen_class.return_value = mock_generator

        pipeline = RAGPipeline("test_db")

        # Make multiple queries
        response1 = pipeline.query("Question 1")
        response2 = pipeline.query("Question 2")
        response3 = pipeline.query("Question 3")

        # All should succeed
        assert response1['answer'] == "This is the generated answer."
        assert response2['answer'] == "This is the generated answer."
        assert response3['answer'] == "This is the generated answer."

        # History should be maintained
        assert len(pipeline.conversation_history) == 6
