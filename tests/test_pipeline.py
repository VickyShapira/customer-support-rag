"""
Quick integration test for RAG pipeline with real database
Run this manually to verify the pipeline works end-to-end
"""
import pytest
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rag_pipeline import RAGPipeline


@pytest.mark.e2e
def test_pipeline_with_real_database(project_root, skip_if_no_api_key, skip_if_no_vector_db):
    """Test RAG pipeline with the full 10,003 entry database"""
    vector_db_path = str(project_root / 'data' / 'vector_db')

    # Initialize pipeline with all layers
    rag = RAGPipeline(
        vector_db_path=vector_db_path,
        model="gpt-4o-mini",
        use_contextual_retriever=True,
        use_smart_retriever=True
    )

    # Test basic query
    test_query = "Why was I charged a fee?"
    result = rag.query(test_query, n_results=3)

    # Assertions
    assert 'answer' in result
    assert 'question' in result
    assert 'sources' in result
    assert len(result['answer']) > 0
    assert result['question'] == test_query
    assert len(result['sources']) == 3

    # Check source structure
    for source in result['sources']:
        assert 'category' in source
        assert 'similarity' in source
        assert 0 <= source['similarity'] <= 1

    print(f"\n✅ Query: {test_query}")
    print(f"✅ Answer: {result['answer'][:200]}...")
    print(f"✅ Top category: {result['sources'][0]['category']}")


@pytest.mark.e2e
def test_pipeline_conversation_flow(project_root, skip_if_no_api_key, skip_if_no_vector_db):
    """Test multi-turn conversation with contextual retrieval"""
    vector_db_path = str(project_root / 'data' / 'vector_db')

    rag = RAGPipeline(
        vector_db_path=vector_db_path,
        model="gpt-4o-mini",
        use_contextual_retriever=True,
        use_smart_retriever=True
    )

    # First query
    result1 = rag.query("What are bank transfer fees?", n_results=3)
    assert 'answer' in result1

    # Follow-up query (should be reformulated by contextual retriever)
    result2 = rag.query("How much?", n_results=3)
    assert 'answer' in result2

    # Verify conversation history is maintained
    history = rag.get_conversation_history()
    assert len(history) == 4  # 2 user + 2 assistant messages

    print(f"\n✅ First query successful")
    print(f"✅ Follow-up query successful")
    print(f"✅ Conversation history: {len(history)} messages")


@pytest.mark.e2e
def test_pipeline_streaming(project_root, skip_if_no_api_key, skip_if_no_vector_db):
    """Test streaming response generation"""
    vector_db_path = str(project_root / 'data' / 'vector_db')

    rag = RAGPipeline(
        vector_db_path=vector_db_path,
        model="gpt-4o-mini"
    )

    # Stream query
    chunks = []
    for chunk_data in rag.query_stream("How do I reset my password?", n_results=3):
        chunks.append(chunk_data)

    # Verify streaming worked
    assert len(chunks) > 0

    # First chunk should have metadata
    assert 'sources' in chunks[0]

    # Last chunk should have done=True
    assert chunks[-1]['done'] is True
    assert 'full_answer' in chunks[-1]

    print(f"\n✅ Streaming successful: {len(chunks)} chunks received")


if __name__ == "__main__":
    """Run manually for quick testing"""
    print("="*70)
    print("TESTING RAG PIPELINE WITH 10,003 ENTRY DATABASE")
    print("="*70)

    project_root = Path(__file__).parent.parent
    vector_db_path = str(project_root / 'data' / 'vector_db')

    # Check prerequisites
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    if not (project_root / 'data' / 'vector_db').exists():
        print("❌ Vector database not found")
        sys.exit(1)

    try:
        print(f"\n1. Initializing RAG pipeline...")
        print(f"   Database path: {vector_db_path}")

        rag = RAGPipeline(
            vector_db_path=vector_db_path,
            model="gpt-4o-mini",
            use_contextual_retriever=True,
            use_smart_retriever=True
        )

        print("   ✅ Pipeline initialized successfully!")

        # Test query
        print(f"\n2. Testing retrieval with sample query...")
        test_query = "Why was I charged a fee?"

        result = rag.query(test_query, n_results=3)

        print(f"   ✅ Query successful!")
        print(f"\n   Query: {test_query}")
        print(f"   Answer: {result['answer'][:200]}...")

        if 'sources' in result:
            print(f"\n   Top sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"      {i}. {source['category']} (similarity: {source['similarity']:.3f})")

        print("\n" + "="*70)
        print("✅ SUCCESS! Your pipeline is using the database with 10,003 entries!")
        print("="*70)

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        sys.exit(1)
