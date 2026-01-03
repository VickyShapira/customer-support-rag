"""Quick script to verify ChromaDB has 10003 entries"""
import chromadb

# Connect to the vector database
client = chromadb.PersistentClient(path=r'c:\Users\victo\customer-support-rag\data\vector_db')

try:
    collection = client.get_collection(name='banking_support')
    count = collection.count()
    print(f"SUCCESS: Collection 'banking_support' found!")
    print(f"Total entries: {count}")

    if count == 10003:
        print("\nCONFIRMED: Database contains exactly 10,003 entries as expected!")
    else:
        print(f"\nWARNING: Expected 10,003 entries but found {count}")

    # Show sample metadata
    sample = collection.peek(limit=1)
    if sample['metadatas']:
        print(f"\nSample entry metadata:")
        print(f"  Category: {sample['metadatas'][0].get('category')}")
        print(f"  Has negation: {sample['metadatas'][0].get('has_negation')}")
        print(f"  Question: {sample['metadatas'][0].get('question', 'N/A')[:60]}...")

except Exception as e:
    print(f"ERROR: {e}")
    print("\nThis might be a ChromaDB version incompatibility issue.")
