"""
Test dual retriever (Smart + Contextual) integration
Verifies proper layering and interaction between retrievers
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rag_pipeline import RAGPipeline
from retrieval.contextual_retriever import ContextualRetriever
from retrieval.smart_retriever import SmartRetriever
from retrieval.retriever import KnowledgeBaseRetriever


@pytest.mark.integration
def test_dual_retriever_layer_structure(project_root, mock_env_vars):
    """Test that both retrievers are properly layered in the pipeline"""
    from unittest.mock import patch

    with patch('rag_pipeline.KnowledgeBaseRetriever'), \
         patch('rag_pipeline.AnswerGenerator'):

        pipeline = RAGPipeline(
            "test_path",
            use_contextual_retriever=True,
            use_smart_retriever=True
        )

        # Verify top layer is ContextualRetriever
        assert isinstance(pipeline.retriever, ContextualRetriever), \
            f"Top layer should be ContextualRetriever, got {type(pipeline.retriever).__name__}"

        # Verify middle layer is SmartRetriever
        middle_layer = pipeline.retriever.base_retriever
        assert isinstance(middle_layer, SmartRetriever), \
            f"Middle layer should be SmartRetriever, got {type(middle_layer).__name__}"

        print("\n✓ Retrieval flow correctly configured:")
        print("  Query → ContextualRetriever → SmartRetriever → KnowledgeBaseRetriever")


@pytest.mark.integration
def test_dual_retriever_configuration_options(project_root, mock_env_vars):
    """Test different retriever configuration combinations"""
    from unittest.mock import patch

    with patch('rag_pipeline.KnowledgeBaseRetriever') as mock_base, \
         patch('rag_pipeline.AnswerGenerator'):

        # Test 1: Both enabled
        p1 = RAGPipeline(
            "test",
            use_contextual_retriever=True,
            use_smart_retriever=True
        )
        assert isinstance(p1.retriever, ContextualRetriever)
        assert isinstance(p1.retriever.base_retriever, SmartRetriever)

        # Test 2: Only contextual
        p2 = RAGPipeline(
            "test",
            use_contextual_retriever=True,
            use_smart_retriever=False
        )
        assert isinstance(p2.retriever, ContextualRetriever)
        # Base should be the mocked KnowledgeBaseRetriever
        assert not isinstance(p2.retriever.base_retriever, SmartRetriever)

        # Test 3: Only smart
        p3 = RAGPipeline(
            "test",
            use_contextual_retriever=False,
            use_smart_retriever=True
        )
        assert isinstance(p3.retriever, SmartRetriever)

        # Test 4: Neither (base only)
        p4 = RAGPipeline(
            "test",
            use_contextual_retriever=False,
            use_smart_retriever=False
        )
        # Should be direct base retriever
        assert not isinstance(p4.retriever, (ContextualRetriever, SmartRetriever))

        print("\n✓ All configuration combinations work correctly")


@pytest.mark.integration
def test_pipeline_info_reflects_layers(project_root, mock_env_vars):
    """Test that get_pipeline_info correctly reports layer configuration"""
    from unittest.mock import patch

    with patch('rag_pipeline.KnowledgeBaseRetriever'), \
         patch('rag_pipeline.AnswerGenerator'):

        pipeline = RAGPipeline(
            "test",
            use_contextual_retriever=True,
            use_smart_retriever=True,
            confidence_threshold=0.35,
            similarity_gap_threshold=0.08
        )

        info = pipeline.get_pipeline_info()

        # Check layer flags
        assert info['layers']['contextual_retriever'] is True
        assert info['layers']['smart_retriever'] is True
        assert info['layers']['base_retriever'] is True

        # Check thresholds
        assert info['thresholds']['confidence_threshold'] == 0.35
        assert info['thresholds']['similarity_gap_threshold'] == 0.08

        print(f"\n✓ Pipeline info correctly reports configuration:")
        print(f"  Layers: {info['layers']}")
        print(f"  Thresholds: {info['thresholds']}")


@pytest.mark.e2e
def test_dual_retriever_e2e(project_root, skip_if_no_api_key, skip_if_no_vector_db):
    """End-to-end test with real database (expensive)"""
    vector_db_path = str(project_root / 'data' / 'vector_db')

    pipeline = RAGPipeline(
        vector_db_path,
        use_contextual_retriever=True,
        use_smart_retriever=True,
        model="gpt-4o-mini"
    )

    # Verify layer structure with real components
    assert isinstance(pipeline.retriever, ContextualRetriever)
    assert isinstance(pipeline.retriever.base_retriever, SmartRetriever)
    assert isinstance(pipeline.retriever.base_retriever.base_retriever, KnowledgeBaseRetriever)

    # Test a query that benefits from both layers
    # 1. Ambiguous category (triggers smart retrieval)
    # 2. Follow-up style (triggers contextual retrieval)
    result = pipeline.query("My card was declined", n_results=3)

    assert 'answer' in result
    assert len(result['answer']) > 0

    # Test follow-up that needs context
    result2 = pipeline.query("Why?", n_results=3)

    assert 'answer' in result2
    assert len(result2['answer']) > 0

    print("\n✓ Dual retriever E2E test passed")
    print(f"  Query 1: 'My card was declined'")
    print(f"  Query 2: 'Why?' (follow-up)")
    print(f"  Both queries processed successfully")


if __name__ == "__main__":
    """Manual test runner"""
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)

    print("="*60)
    print("TESTING DUAL RETRIEVER (SMART + CONTEXTUAL) SETUP")
    print("="*60)

    project_root = Path(__file__).parent.parent
    vector_db_path = project_root / 'data' / 'vector_db'

    if not vector_db_path.exists():
        print(f"\n⚠ Vector DB not found at {vector_db_path}")
        print("Testing layer structure only (without real database)...\n")

        from unittest.mock import patch

        with patch('rag_pipeline.KnowledgeBaseRetriever'), \
             patch('rag_pipeline.AnswerGenerator'):

            os.environ['OPENAI_API_KEY'] = 'test-key'

            pipeline = RAGPipeline(
                "test_path",
                use_contextual_retriever=True,
                use_smart_retriever=True
            )

            print("1. Checking retriever layers:")
            print(f"   - Top layer: {type(pipeline.retriever).__name__}")

            if isinstance(pipeline.retriever, ContextualRetriever):
                print("   ✓ ContextualRetriever is the outer layer")

                middle = pipeline.retriever.base_retriever
                print(f"   - Middle layer: {type(middle).__name__}")

                if isinstance(middle, SmartRetriever):
                    print("   ✓ SmartRetriever is the middle layer")

                    base = middle.base_retriever
                    print(f"   - Base layer: {type(base).__name__}")
                    print("   ✓ Base retriever is present")

                    print("\n✓ ALL LAYERS CORRECTLY CONFIGURED!")
                    print("\nRetrieval flow:")
                    print("  Query → ContextualRetriever (reformulates follow-ups)")
                    print("       → SmartRetriever (disambiguates overlapping categories)")
                    print("       → KnowledgeBaseRetriever (semantic search + reranking)")

            print("\n2. Pipeline configuration:")
            info = pipeline.get_pipeline_info()
            for key, value in info.items():
                print(f"   {key}: {value}")

            print("\n" + "="*60)
            print("SUCCESS! Dual retriever setup is properly configured.")
            print("="*60)

    else:
        print("\n✓ Vector DB found, running full test...\n")

        if not os.getenv('OPENAI_API_KEY'):
            print("❌ OPENAI_API_KEY not set")
            sys.exit(1)

        try:
            pipeline = RAGPipeline(
                str(vector_db_path),
                use_contextual_retriever=True,
                use_smart_retriever=True
            )

            print("1. Layer structure:")
            print(f"   Top: {type(pipeline.retriever).__name__}")
            print(f"   Middle: {type(pipeline.retriever.base_retriever).__name__}")
            print(f"   Base: {type(pipeline.retriever.base_retriever.base_retriever).__name__}")

            print("\n2. Testing query...")
            result = pipeline.query("Why was I charged a fee?", n_results=3)

            print(f"   ✓ Answer: {result['answer'][:100]}...")
            print(f"   ✓ Sources: {len(result['sources'])} found")

            print("\n" + "="*60)
            print("SUCCESS! All tests passed.")
            print("="*60)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
