"""
Quick script to check your ChromaDB vector database
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import chromadb
from collections import Counter

# Paths
project_root = Path(__file__).parent
vector_db_path = project_root / "data" / "vector_db"

print("=" * 80)
print("VECTOR DATABASE INSPECTION")
print("=" * 80)

try:
    # Connect to ChromaDB
    print(f"\nConnecting to: {vector_db_path}")
    client = chromadb.PersistentClient(path=str(vector_db_path))

    # Get collection
    collection = client.get_collection(name="banking_support")

    # Get count
    try:
        count = collection.count()
        print(f"✓ Total documents: {count:,}")
    except Exception as e:
        print(f"⚠️ Cannot get count (ChromaDB version issue): {e}")
        print("   Trying alternative method...")
        # Alternative: get all and count
        all_data = collection.get()
        count = len(all_data['ids'])
        print(f"✓ Total documents: {count:,}")

    # Get sample data
    print("\n" + "-" * 80)
    print("SAMPLE DATA (first 10 documents)")
    print("-" * 80)

    sample = collection.get(limit=10)

    for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas']), 1):
        category = meta.get('category', 'unknown')
        doc_preview = doc[:80] + "..." if len(doc) > 80 else doc
        print(f"\n{i}. Category: {category}")
        print(f"   Text: {doc_preview}")

    # Get all categories
    print("\n" + "-" * 80)
    print("CATEGORY DISTRIBUTION")
    print("-" * 80)

    # Fetch all metadata
    all_data = collection.get()
    categories = [meta['category'] for meta in all_data['metadatas']]
    category_counts = Counter(categories)

    print(f"\nUnique categories: {len(category_counts)}")
    print(f"\nTop 20 categories by document count:")
    for category, count in category_counts.most_common(20):
        print(f"  {category:40s}: {count:4d} docs")

    if len(category_counts) > 20:
        print(f"\n  ... and {len(category_counts) - 20} more categories")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total documents:     {count:,}")
    print(f"Unique categories:   {len(category_counts)}")
    print(f"Avg docs/category:   {count / len(category_counts):.1f}")
    print(f"Min docs/category:   {min(category_counts.values())}")
    print(f"Max docs/category:   {max(category_counts.values())}")

    # Compare with knowledge base files
    print("\n" + "-" * 80)
    print("KNOWLEDGE BASE COMPARISON")
    print("-" * 80)

    import pandas as pd

    kb_path = project_root / "data" / "processed" / "knowledge_base.csv"
    kb_v2_path = project_root / "data" / "processed" / "knowledge_base_v2.csv"

    if kb_path.exists():
        kb_df = pd.read_csv(kb_path)
        print(f"knowledge_base.csv:     {len(kb_df):,} rows")

    if kb_v2_path.exists():
        kb_v2_df = pd.read_csv(kb_v2_path)
        print(f"knowledge_base_v2.csv:  {len(kb_v2_df):,} rows")

    print(f"\nVector database:        {count:,} documents")

    if count == len(kb_df):
        print("\n✓ Vector DB matches knowledge_base.csv (100 docs - SMALL VERSION)")
        print("  → Consider rebuilding with knowledge_base_v2.csv for better accuracy")
    elif count == len(kb_v2_df):
        print("\n✓ Vector DB matches knowledge_base_v2.csv (FULL VERSION)")
    else:
        print(f"\n⚠️ Vector DB size ({count}) doesn't match either knowledge base file")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
