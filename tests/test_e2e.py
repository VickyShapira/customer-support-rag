"""
End-to-end tests for the complete RAG system
Tests against real evaluation dataset (requires actual vector DB and API access)
"""
import pytest
import json
import os
from pathlib import Path
from src.rag_pipeline import RAGPipeline
from src.retrieval.retriever import KnowledgeBaseRetriever
from src.generation.generator import AnswerGenerator


# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


class TestEndToEnd:
    """End-to-end tests using real components"""

    @pytest.fixture(scope="class")
    def vector_db_path(self):
        """Path to vector database"""
        base_path = Path(__file__).parent.parent
        db_path = base_path / "data" / "vector_db"

        if not db_path.exists():
            pytest.skip("Vector database not found. Run notebooks to create it.")

        return str(db_path)

    @pytest.fixture(scope="class")
    def evaluation_set(self):
        """Load evaluation dataset"""
        base_path = Path(__file__).parent.parent
        eval_path = base_path / "data" / "processed" / "evaluation_set.json"

        if not eval_path.exists():
            pytest.skip("Evaluation set not found.")

        with open(eval_path, 'r') as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def api_key_available(self):
        """Check if OpenAI API key is available"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OPENAI_API_KEY not set. Skipping E2E tests.")
        return True

    @pytest.fixture
    def pipeline(self, vector_db_path, api_key_available):
        """Create RAG pipeline instance"""
        return RAGPipeline(vector_db_path, model="gpt-4o-mini")

    @pytest.fixture
    def retriever(self, vector_db_path, api_key_available):
        """Create retriever instance"""
        return KnowledgeBaseRetriever(vector_db_path, use_reranking=True)

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly"""
        assert pipeline is not None
        assert pipeline.retriever is not None
        assert pipeline.generator is not None
        assert pipeline.conversation_history == []

    def test_retriever_initialization(self, retriever):
        """Test that retriever initializes correctly"""
        assert retriever is not None
        assert retriever.collection is not None
        assert retriever.use_reranking is True

    def test_simple_query(self, pipeline):
        """Test a simple query end-to-end"""
        response = pipeline.query(
            "How do I reset my password?",
            n_results=3
        )

        # Verify response structure
        assert 'question' in response
        assert 'answer' in response
        assert 'sources' in response
        assert 'metadata' in response

        # Verify content
        assert len(response['answer']) > 0
        assert len(response['sources']) == 3
        assert all('category' in s for s in response['sources'])
        assert all('similarity' in s for s in response['sources'])

    def test_retrieval_quality(self, retriever):
        """Test that retrieval returns relevant documents"""
        query = "My card was declined"
        results = retriever.retrieve(query, n_results=5)

        assert 'documents' in results
        assert 'metadatas' in results
        assert len(results['documents']) == 5

        # Check that categories are related to card/payment issues
        categories = [meta['category'] for meta in results['metadatas']]
        assert any('card' in cat.lower() or 'payment' in cat.lower()
                   for cat in categories)

    def test_reranking_improves_results(self, vector_db_path, api_key_available):
        """Test that reranking provides better ordering"""
        # Create two retrievers - with and without reranking
        retriever_no_rerank = KnowledgeBaseRetriever(
            vector_db_path,
            use_reranking=False
        )
        retriever_with_rerank = KnowledgeBaseRetriever(
            vector_db_path,
            use_reranking=True
        )

        query = "I can't transfer money"

        results_no_rerank = retriever_no_rerank.retrieve(query, n_results=3)
        results_with_rerank = retriever_with_rerank.retrieve(query, n_results=3)

        # Both should return results
        assert len(results_no_rerank['documents']) == 3
        assert len(results_with_rerank['documents']) == 3

        # Reranked results should have rerank scores
        assert results_with_rerank['reranked'] is True
        assert 'rerank_scores' in results_with_rerank
        assert len(results_with_rerank['rerank_scores']) == 3

    def test_conversation_context(self, pipeline):
        """Test multi-turn conversation"""
        # First query
        response1 = pipeline.query("What are transfer fees?")
        assert len(response1['answer']) > 0

        # Follow-up query (should use context)
        response2 = pipeline.query("How much is it?")
        assert len(response2['answer']) > 0

        # Verify conversation history
        history = pipeline.get_conversation_history()
        assert len(history) == 4  # 2 Q&A pairs
        assert history[0]['content'] == "What are transfer fees?"
        assert history[2]['content'] == "How much is it?"

    def test_streaming_query(self, pipeline):
        """Test streaming response"""
        stream = pipeline.query_stream(
            "How do I activate my card?",
            n_results=3
        )

        chunks = list(stream)

        # Should have multiple chunks
        assert len(chunks) > 1

        # First chunk should have metadata
        assert 'sources' in chunks[0]
        assert chunks[0]['done'] is False

        # Last chunk should signal completion
        assert chunks[-1]['done'] is True
        assert 'full_answer' in chunks[-1]
        assert len(chunks[-1]['full_answer']) > 0

    @pytest.mark.parametrize("test_case_idx", [0, 1, 2, 3, 4])
    def test_evaluation_samples(self, pipeline, evaluation_set, test_case_idx):
        """Test against specific evaluation samples"""
        if test_case_idx >= len(evaluation_set):
            pytest.skip(f"Test case {test_case_idx} not in evaluation set")

        test_case = evaluation_set[test_case_idx]
        query = test_case['query']
        expected_category = test_case['expected_category']

        response = pipeline.query(query, n_results=5)

        # Verify response structure
        assert 'answer' in response
        assert len(response['answer']) > 0

        # Check if expected category is in top results
        categories = [s['category'] for s in response['sources']]
        assert expected_category in categories, \
            f"Expected category '{expected_category}' not found in top results for query: '{query}'"

    def test_negation_queries(self, pipeline, evaluation_set):
        """Test queries with negation"""
        negation_queries = [
            test for test in evaluation_set
            if test.get('has_negation', False)
        ]

        if not negation_queries:
            pytest.skip("No negation queries in evaluation set")

        # Test first negation query
        test_case = negation_queries[0]
        response = pipeline.query(test_case['query'], n_results=5)

        assert 'answer' in response
        assert len(response['answer']) > 0

    def test_short_queries(self, pipeline, evaluation_set):
        """Test short queries (2-4 words)"""
        short_queries = [
            test for test in evaluation_set
            if test.get('test_type') == 'short'
        ]

        if not short_queries:
            pytest.skip("No short queries in evaluation set")

        # Test first short query
        test_case = short_queries[0]
        response = pipeline.query(test_case['query'], n_results=5)

        assert 'answer' in response
        assert len(response['answer']) > 0

    def test_complex_queries(self, pipeline, evaluation_set):
        """Test complex multi-part queries"""
        complex_queries = [
            test for test in evaluation_set
            if test.get('test_type') == 'complex'
        ]

        if not complex_queries:
            pytest.skip("No complex queries in evaluation set")

        # Test first complex query
        test_case = complex_queries[0]
        response = pipeline.query(test_case['query'], n_results=5)

        assert 'answer' in response
        assert len(response['answer']) > 0

    def test_category_accuracy_top_1(self, pipeline, evaluation_set):
        """Test accuracy of top-1 category retrieval"""
        correct = 0
        total = min(10, len(evaluation_set))  # Test first 10 samples

        for test_case in evaluation_set[:total]:
            response = pipeline.query(test_case['query'], n_results=5)

            if response['sources'][0]['category'] == test_case['expected_category']:
                correct += 1

        accuracy = correct / total
        # Should achieve at least 70% accuracy (adjust based on your system)
        assert accuracy >= 0.7, f"Top-1 accuracy too low: {accuracy:.2%}"

    def test_category_accuracy_top_3(self, pipeline, evaluation_set):
        """Test accuracy of top-3 category retrieval"""
        correct = 0
        total = min(10, len(evaluation_set))

        for test_case in evaluation_set[:total]:
            response = pipeline.query(test_case['query'], n_results=5)
            top_3_categories = [s['category'] for s in response['sources'][:3]]

            if test_case['expected_category'] in top_3_categories:
                correct += 1

        accuracy = correct / total
        # Should achieve at least 85% accuracy in top-3
        assert accuracy >= 0.85, f"Top-3 accuracy too low: {accuracy:.2%}"

    def test_answer_relevance(self, pipeline):
        """Test that answers are relevant to questions"""
        test_queries = [
            "How do I reset my password?",
            "My card was declined",
            "How long does a transfer take?"
        ]

        for query in test_queries:
            response = pipeline.query(query, n_results=3)
            answer = response['answer'].lower()

            # Answer should contain relevant keywords
            assert len(answer) > 20, "Answer too short"
            assert len(answer) < 1000, "Answer too long"

    def test_context_formatting(self, retriever):
        """Test that context is formatted correctly for LLM"""
        results = retriever.retrieve("test query", n_results=3)
        context = retriever.format_context(results)

        # Check formatting
        assert '[Document 1]' in context
        assert '[Document 2]' in context
        assert '[Document 3]' in context
        assert 'Category:' in context
        assert 'Relevance:' in context

    def test_similarity_scores(self, retriever):
        """Test that similarity scores are reasonable"""
        results = retriever.retrieve("How do I reset my password?", n_results=5)

        for dist in results['distances']:
            # Distances should be between 0 and 2 for cosine distance
            assert 0 <= dist <= 2

            # Similarity (1 - dist) should be positive for good matches
            similarity = 1 - dist
            assert similarity > 0

    def test_short_query_handling(self, pipeline):
        """Test handling of very short queries"""
        # Very short query
        response = pipeline.query("help", n_results=3)
        assert 'answer' in response
        assert len(response['answer']) > 0

    def test_very_long_query(self, pipeline):
        """Test handling of very long queries"""
        long_query = "I need help with " + "my banking issue " * 50
        response = pipeline.query(long_query, n_results=3)

        assert 'answer' in response
        assert len(response['answer']) > 0

    def test_special_characters_in_query(self, pipeline):
        """Test queries with special characters"""
        queries = [
            "What's the fee for transfers?",
            "I can't access my account!",
            "Card #1234 - declined?",
            "Transfer $500 to account"
        ]

        for query in queries:
            response = pipeline.query(query, n_results=3)
            assert 'answer' in response
            assert len(response['answer']) > 0

    def test_reset_conversation(self, pipeline):
        """Test conversation reset"""
        # Build up conversation
        pipeline.query("First question")
        pipeline.query("Second question")
        assert len(pipeline.get_conversation_history()) > 0

        # Reset
        pipeline.reset_conversation()
        assert len(pipeline.get_conversation_history()) == 0

        # Should still work after reset
        response = pipeline.query("New question")
        assert 'answer' in response
