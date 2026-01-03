"""
Performance and benchmark tests for the RAG system
"""
import pytest
import time
from pathlib import Path
from src.rag_pipeline import RAGPipeline


# Mark all tests as slow and e2e
pytestmark = [pytest.mark.slow, pytest.mark.e2e]


class TestPerformance:
    """Performance benchmarks for the RAG system"""

    @pytest.fixture(scope="class")
    def pipeline(self, skip_if_no_api_key, skip_if_no_vector_db, project_root):
        """Create pipeline instance"""
        db_path = project_root / "data" / "vector_db"
        return RAGPipeline(str(db_path), model="gpt-4o-mini")

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for benchmarking"""
        return [
            "How do I reset my password?",
            "My card was declined",
            "How long does a transfer take?",
            "What are the fees for international transfers?",
            "How do I activate my card?",
            "I can't login to my account",
            "How do I top up my account?",
            "My payment is pending",
            "How do I get a virtual card?",
            "What is the transfer limit?"
        ]

    def test_retrieval_latency(self, pipeline, performance_threshold):
        """Test retrieval latency"""
        query = "How do I reset my password?"

        start = time.time()
        results = pipeline.retriever.retrieve(query, n_results=5)
        elapsed = time.time() - start

        assert elapsed < performance_threshold['retrieval_time'], \
            f"Retrieval took {elapsed:.2f}s, expected < {performance_threshold['retrieval_time']}s"

        # Log for monitoring
        print(f"\nRetrieval latency: {elapsed:.3f}s")

    def test_generation_latency(self, pipeline, performance_threshold):
        """Test answer generation latency"""
        query = "How do I reset my password?"
        context = "[Document 1] To reset password, go to settings."

        start = time.time()
        answer = pipeline.generator.generate(query, context, [])
        elapsed = time.time() - start

        assert elapsed < performance_threshold['generation_time'], \
            f"Generation took {elapsed:.2f}s, expected < {performance_threshold['generation_time']}s"

        print(f"Generation latency: {elapsed:.3f}s")

    def test_end_to_end_latency(self, pipeline, performance_threshold):
        """Test full pipeline latency"""
        query = "How do I reset my password?"

        start = time.time()
        response = pipeline.query(query, n_results=3)
        elapsed = time.time() - start

        assert elapsed < performance_threshold['total_query_time'], \
            f"Total query took {elapsed:.2f}s, expected < {performance_threshold['total_query_time']}s"

        assert 'answer' in response
        print(f"End-to-end latency: {elapsed:.3f}s")

    def test_streaming_first_chunk_latency(self, pipeline, performance_threshold):
        """Test time to first chunk in streaming mode"""
        query = "How do I reset my password?"

        start = time.time()
        stream = pipeline.query_stream(query, n_results=3)

        # Get first chunk (metadata)
        first_chunk = next(stream)
        first_chunk_time = time.time() - start

        # Get second chunk (first content chunk)
        second_chunk = next(stream)
        first_content_time = time.time() - start

        # Consume rest of stream
        list(stream)

        assert first_content_time < performance_threshold['streaming_first_chunk'], \
            f"First content chunk took {first_content_time:.2f}s"

        print(f"Time to first chunk: {first_chunk_time:.3f}s")
        print(f"Time to first content: {first_content_time:.3f}s")

    def test_batch_query_performance(self, pipeline, sample_queries):
        """Test performance across multiple queries"""
        latencies = []

        for query in sample_queries:
            start = time.time()
            response = pipeline.query(query, n_results=3)
            elapsed = time.time() - start
            latencies.append(elapsed)

            assert 'answer' in response

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print(f"\nBatch performance (n={len(sample_queries)}):")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Min: {min_latency:.3f}s")
        print(f"  Max: {max_latency:.3f}s")

        # Average should be reasonable
        assert avg_latency < 10.0, f"Average latency too high: {avg_latency:.2f}s"

    def test_reranking_overhead(self, project_root, skip_if_no_api_key,
                               skip_if_no_vector_db):
        """Test performance impact of reranking"""
        from src.retrieval.retriever import KnowledgeBaseRetriever

        db_path = project_root / "data" / "vector_db"
        query = "How do I reset my password?"

        # Without reranking
        retriever_no_rerank = KnowledgeBaseRetriever(
            str(db_path),
            use_reranking=False
        )
        start = time.time()
        results_no_rerank = retriever_no_rerank.retrieve(query, n_results=5)
        time_no_rerank = time.time() - start

        # With reranking
        retriever_with_rerank = KnowledgeBaseRetriever(
            str(db_path),
            use_reranking=True
        )
        start = time.time()
        results_with_rerank = retriever_with_rerank.retrieve(query, n_results=5)
        time_with_rerank = time.time() - start

        overhead = time_with_rerank - time_no_rerank
        overhead_pct = (overhead / time_no_rerank) * 100 if time_no_rerank > 0 else 0

        print(f"\nReranking overhead:")
        print(f"  Without reranking: {time_no_rerank:.3f}s")
        print(f"  With reranking: {time_with_rerank:.3f}s")
        print(f"  Overhead: {overhead:.3f}s ({overhead_pct:.1f}%)")

        # Reranking should add less than 1 second overhead
        assert overhead < 1.0, f"Reranking overhead too high: {overhead:.2f}s"

    def test_conversation_history_impact(self, pipeline):
        """Test performance impact of conversation history"""
        query = "How much is it?"

        # Without history
        start = time.time()
        response1 = pipeline.query(query, n_results=3)
        time_no_history = time.time() - start

        # Build up history
        pipeline.query("What are transfer fees?")
        pipeline.query("How long does it take?")

        # With history
        start = time.time()
        response2 = pipeline.query(query, n_results=3)
        time_with_history = time.time() - start

        print(f"\nConversation history impact:")
        print(f"  Without history: {time_no_history:.3f}s")
        print(f"  With history: {time_with_history:.3f}s")
        print(f"  Difference: {abs(time_with_history - time_no_history):.3f}s")

        # Both should complete in reasonable time
        assert time_with_history < 10.0

    def test_concurrent_queries(self, pipeline, sample_queries):
        """Test handling of sequential queries (simulating concurrent use)"""
        import concurrent.futures

        # Reset conversation for clean state
        pipeline.reset_conversation()

        def run_query(query):
            start = time.time()
            response = pipeline.query(query, n_results=3)
            return time.time() - start, response

        # Run queries sequentially (to avoid API rate limits)
        results = []
        for query in sample_queries[:5]:  # Test with 5 queries
            latency, response = run_query(query)
            results.append((latency, response))
            assert 'answer' in response

        latencies = [r[0] for r in results]
        avg = sum(latencies) / len(latencies)

        print(f"\nSequential queries performance:")
        print(f"  Queries: {len(results)}")
        print(f"  Average latency: {avg:.3f}s")

    def test_memory_efficiency(self, pipeline, sample_queries):
        """Test that conversation history doesn't grow unbounded"""
        # Make many queries
        for i in range(10):
            pipeline.query(sample_queries[i % len(sample_queries)])

        # History should be limited to 6 messages
        history = pipeline.get_conversation_history()
        assert len(history) <= 6, \
            f"History should be limited to 6 messages, got {len(history)}"

        print(f"\nMemory test - conversation history size: {len(history)}")

    @pytest.mark.parametrize("n_results", [1, 3, 5, 10])
    def test_retrieval_scaling(self, pipeline, n_results):
        """Test how retrieval performance scales with number of results"""
        query = "How do I reset my password?"

        start = time.time()
        results = pipeline.retriever.retrieve(query, n_results=n_results)
        elapsed = time.time() - start

        assert len(results['documents']) <= n_results

        print(f"n_results={n_results}: {elapsed:.3f}s")

        # Should complete in reasonable time regardless of n_results
        assert elapsed < 3.0

    def test_embedding_caching_effect(self, pipeline):
        """Test if repeated queries benefit from any caching"""
        query = "How do I reset my password?"

        # First query (cold)
        start = time.time()
        response1 = pipeline.query(query, n_results=3)
        time_first = time.time() - start

        # Immediate repeat (potentially cached)
        start = time.time()
        response2 = pipeline.query(query, n_results=3)
        time_second = time.time() - start

        print(f"\nCaching effect test:")
        print(f"  First query: {time_first:.3f}s")
        print(f"  Second query: {time_second:.3f}s")
        print(f"  Ratio: {time_second/time_first:.2f}x")

        # Both should return valid responses
        assert 'answer' in response1
        assert 'answer' in response2
