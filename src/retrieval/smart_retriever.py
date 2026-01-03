"""
Smart Retriever with LLM Disambiguation
Handles overlapping categories by using GPT-4o-mini for final classification
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import openai

load_dotenv(override=True)

class SmartRetriever:
    """
    Retriever that uses LLM disambiguation for overlapping categories
    """
    
    # Known overlapping category groups from confusion matrix analysis
    OVERLAPPING_GROUPS = [
        # Card payment issues
        {
            'declined_card_payment',
            'card_not_working',
            'reverted_card_payment?',
            'card_payment_wrong_exchange_rate'
        },
        # Transfer/Balance issues
        {
            'balance_not_updated_after_bank_transfer',
            'pending_transfer',
            'transfer_timing',
            'transfer_not_received_by_recipient'
        },
        # Payment recognition
        {
            'card_payment_not_recognised',
            'direct_debit_payment_not_recognised',
            'cash_withdrawal_not_recognised'
        },
        # Top-up issues
        {
            'top_up_failed',
            'top_up_reverted',
            'pending_top_up'
        },
        # Fee/charge categories (often overlapping)
        {
            'card_payment_fee_charged',
            'cash_withdrawal_charge',
            'transfer_fee_charged',
            'exchange_charge',
            'atm_fee_in_foreign_currency'
        }
    ]
    
    def __init__(self, base_retriever, confidence_threshold=0.38, similarity_gap_threshold=0.10):
        """
        Args:
            base_retriever: Your existing KnowledgeBaseRetriever
            confidence_threshold: If top similarity < this, use LLM (default: 0.38)
            similarity_gap_threshold: If gap between top 2 < this, use LLM (default: 0.10)
        """
        self.base_retriever = base_retriever
        self.confidence_threshold = confidence_threshold
        self.similarity_gap_threshold = similarity_gap_threshold
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def retrieve(self, query: str, n_results: int = 5, **kwargs):
        """
        Smart retrieval with automatic disambiguation

        Args:
            query: User query
            n_results: Number of results to return
            **kwargs: Additional arguments (e.g., conversation_history) passed to base retriever
        """
        # Fetch top 20 results for better category diversity
        results = self.base_retriever.retrieve(query, n_results=20, **kwargs)

        # Calculate confidence metrics
        top_similarity = 1 - results['distances'][0]
        second_similarity = 1 - results['distances'][1]
        similarity_gap = top_similarity - second_similarity

        top_category = results['metadatas'][0]['category']
        second_category = results['metadatas'][1]['category']

        # Check if we need disambiguation
        needs_disambiguation = (
            top_similarity < self.confidence_threshold or  # Low confidence
            similarity_gap < self.similarity_gap_threshold or  # Very close scores
            self._are_overlapping(top_category, second_category)  # Known overlap
        )

        if needs_disambiguation:
            print(f"[SMART RETRIEVAL] Low confidence detected (sim={top_similarity:.3f}, gap={similarity_gap:.3f})")
            print(f"[SMART RETRIEVAL] Using LLM to disambiguate between: {top_category}, {second_category}")

            # Get unique categories from top 20 results (increased from 5 to 8 candidates)
            candidate_categories = self._get_unique_categories(results, max_candidates=8)

            # Smart fallback: Force add exchange_charge for fee-related queries
            query_lower = query.lower().strip()
            fee_keywords = ['fee', 'fees', 'charge', 'charged', 'cost', 'costs',
                          'international transaction', 'exchange rate', 'currency']
            is_fee_query = any(keyword in query_lower for keyword in fee_keywords)

            print(f"[DEBUG] Query received by SmartRetriever: '{query}'")
            print(f"[DEBUG] Is fee query: {is_fee_query} (checking: {fee_keywords})")
            print(f"[DEBUG] Candidates before force-add: {candidate_categories}")

            if is_fee_query:
                # ALWAYS include ALL fee categories as candidates for fee queries
                # This ensures the LLM can choose the most specific fee type
                fee_categories_to_add = [
                    'exchange_charge',  # Must be first priority for international/currency fees
                    'transfer_fee_charged',
                    'cash_withdrawal_charge',
                    'card_payment_fee_charged',
                    'atm_fee_in_foreign_currency'
                ]

                for fee_cat in fee_categories_to_add:
                    if fee_cat not in candidate_categories:
                        candidate_categories.append(fee_cat)
                        print(f"[SMART RETRIEVAL] ✓ Force-added {fee_cat}")

                print(f"[SMART RETRIEVAL] 💡 Fee query detected - added all fee categories")

            print(f"[SMART RETRIEVAL] Final candidates ({len(candidate_categories)}): {candidate_categories}")

            # Use LLM to select best category
            best_category = self._llm_classify(query, candidate_categories, results)

            # Filter results to only include best category
            filtered_results = self._filter_by_category(results, best_category, n_results)

            # If no results found (category not in top 20), do a fresh targeted search
            if len(filtered_results['metadatas']) == 0:
                print(f"[SMART RETRIEVAL] ⚠️ No {best_category} docs in top 20, doing targeted retrieval...")
                filtered_results = self._retrieve_by_category(query, best_category, n_results, fallback_results=results)

            return filtered_results

        # High confidence, return as-is (truncate to requested n_results)
        return {
            'distances': results['distances'][:n_results],
            'metadatas': results['metadatas'][:n_results],
            'documents': results['documents'][:n_results],
            'query': results.get('query', query),
            'reranked': results.get('reranked', False)
        }
    
    def _are_overlapping(self, cat1: str, cat2: str) -> bool:
        """Check if two categories are known to overlap"""
        for group in self.OVERLAPPING_GROUPS:
            if cat1 in group and cat2 in group:
                return True
        return False
    
    def _get_unique_categories(self, results, max_candidates=8) -> List[str]:
        """
        Extract unique categories from results
        
        Args:
            results: Retrieval results with metadatas
            max_candidates: Maximum number of unique categories to return
            
        Returns:
            List of unique category names
        """
        seen = set()
        categories = []
        
        # Iterate through all results (up to 20) to find diverse categories
        for meta in results['metadatas']:
            cat = meta['category']
            if cat not in seen:
                categories.append(cat)
                seen.add(cat)
            if len(categories) >= max_candidates:
                break
        
        # Warning if we couldn't find enough diversity
        if len(categories) < 3:
            print(f"[WARNING] Only found {len(categories)} unique categories in top results")
        
        return categories
    
    def _llm_classify(self, query: str, candidate_categories: List[str], results) -> str:
        """
        Use GPT-4o-mini to classify query into best category
        
        Args:
            query: User's query
            candidate_categories: List of candidate category names
            results: Full retrieval results for context
            
        Returns:
            Selected category name
        """
        
        # Build examples for each category from retrieved results
        category_examples = {}
        for meta in results['metadatas'][:20]:  # Look through top 20
            cat = meta['category']
            if cat in candidate_categories and cat not in category_examples:
                category_examples[cat] = meta.get('question', '')
        
        # Create prompt with specific guidance
        examples_text = "\n".join([
            f"- **{cat}**: Example: \"{category_examples.get(cat, 'N/A')[:80]}...\""
            for cat in candidate_categories
        ])
        
        prompt = f"""You are a banking customer service classifier. Classify this query into the MOST SPECIFIC category.

Query: "{query}"

Candidate categories:
{examples_text}

Category definitions:

**Card Payment Issues:**
- **declined_card_payment**: Payment was REJECTED/DENIED at merchant (e.g., "payment declined", "card rejected at store")
- **card_not_working**: Physical/technical CARD DEFECT (e.g., "card won't swipe", "card is damaged", "card broken")
- **reverted_card_payment?**: Completed payment was REVERSED/REFUNDED
- **card_payment_wrong_exchange_rate**: Wrong exchange rate applied to card payment

**Transfer/Balance Issues:**
- **balance_not_updated_after_bank_transfer**: Transfer COMPLETED but balance wrong
- **pending_transfer**: Transfer IN PROGRESS, waiting
- **transfer_timing**: Asking HOW LONG transfers take
- **transfer_not_received_by_recipient**: Recipient hasn't received money

**Payment Recognition:**
- **card_payment_not_recognised**: Unrecognized card payment charge
- **direct_debit_payment_not_recognised**: Unrecognized direct debit
- **cash_withdrawal_not_recognised**: Unrecognized ATM withdrawal

**Fee/Charge Categories (PAY SPECIAL ATTENTION):**
- **exchange_charge**: Currency conversion fees, exchange rate fees, "international transaction fees" IN GENERAL, foreign exchange costs
- **card_payment_fee_charged**: Fees on card purchases (online/in-store payments)
- **cash_withdrawal_charge**: Fees on ATM withdrawals (domestic or foreign)
- **transfer_fee_charged**: Fees on bank transfers (sending money between accounts)
- **atm_fee_in_foreign_currency**: ATM fees specifically in foreign countries

**Top-up Issues:**
- **top_up_failed**: Top-up attempt failed
- **top_up_reverted**: Top-up was reversed/refunded
- **pending_top_up**: Top-up is in progress

Critical Rules:
1. "declined" or "rejected" = declined_card_payment (NOT card_not_working)
2. Physical card problems (broken, damaged, won't swipe) = card_not_working
3. "How long" questions = transfer_timing or similar timing category
4. Completed action with wrong result = balance_not_updated or similar
5. In-progress/pending = pending_transfer or similar

**For Fee Questions (MOST IMPORTANT):**
6. "international transaction fees" (GENERAL) = exchange_charge
7. "exchange rate fees" or "currency conversion" = exchange_charge
8. "ATM fees" or "withdrawal fees" = cash_withdrawal_charge or atm_fee_in_foreign_currency
9. "transfer fees" or "sending money fees" = transfer_fee_charged
10. "card payment fees" or "purchase fees" = card_payment_fee_charged

Decision Process:
1. What is the user's PRIMARY concern?
2. Is this about: REJECTION, DEFECT, TIMING, or FEES?
3. If FEES: Is it about currency/exchange (→ exchange_charge) or a specific transaction type?
4. Which category is MOST SPECIFIC to the user's question?

Answer with ONLY the category name (exactly as shown above)."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            
            selected = response.choices[0].message.content.strip()
            
            # Validate selection is in candidate list
            if selected in candidate_categories:
                print(f"[LLM] Selected: {selected}")
                return selected
            else:
                print(f"[LLM] Invalid selection '{selected}', falling back to top result")
                return candidate_categories[0]
        
        except Exception as e:
            print(f"[LLM] Classification failed: {e}, falling back to top result")
            return candidate_categories[0]
    
    def _filter_by_category(self, results, category: str, n_results: int):
        """
        Filter results to only include specific category

        Args:
            results: Full retrieval results
            category: Category to filter by
            n_results: Number of results to return

        Returns:
            Filtered results dictionary
        """
        filtered_distances = []
        filtered_metadatas = []
        filtered_documents = []

        for i, meta in enumerate(results['metadatas']):
            if meta['category'] == category:
                filtered_distances.append(results['distances'][i])
                filtered_metadatas.append(meta)
                filtered_documents.append(results['documents'][i])

                if len(filtered_metadatas) >= n_results:
                    break

        return {
            'distances': filtered_distances,
            'metadatas': filtered_metadatas,
            'documents': filtered_documents,
            'query': results.get('query', ''),
            'reranked': results.get('reranked', False)
        }

    def _retrieve_by_category(self, query: str, category: str, n_results: int, fallback_results=None):
        """
        Do a fresh retrieval filtered by specific category

        This is used when the LLM selects a category that doesn't appear in the
        initial top 20 results. We query the vector DB with a category filter.

        Args:
            query: User query
            category: Category to filter by
            n_results: Number of results to return
            fallback_results: Original results to use if category-filtered query returns nothing

        Returns:
            Filtered results dictionary
        """
        # Access the underlying ChromaDB collection directly
        collection = self.base_retriever.collection
        query_embedding = self.base_retriever.get_embedding(query)

        # Query with category filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"category": category}  # Filter by category
        )

        # Handle case where no results are found for this category
        if not results['distances'] or len(results['distances'][0]) == 0:
            print(f"[SMART RETRIEVAL] ⚠️ No documents found for category '{category}' in database!")
            if fallback_results:
                print(f"[SMART RETRIEVAL] Falling back to top result from original query")
                # Return top n_results from original results
                return {
                    'distances': fallback_results['distances'][:n_results],
                    'metadatas': fallback_results['metadatas'][:n_results],
                    'documents': fallback_results['documents'][:n_results],
                    'query': fallback_results.get('query', query),
                    'reranked': fallback_results.get('reranked', False)
                }
            else:
                print(f"[SMART RETRIEVAL] No fallback available, returning empty results")
                return {
                    'distances': [],
                    'metadatas': [],
                    'documents': [],
                    'query': query,
                    'reranked': False
                }

        return {
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0],
            'documents': results['documents'][0],
            'query': query,
            'reranked': False
        }