"""
Conversation Test Scenarios for Contextual RAG Pipeline

These scenarios test multi-turn conversations that require context understanding,
which single-turn evaluation datasets cannot properly assess.
"""

# Test scenarios based on realistic customer support conversations
CONVERSATION_TEST_SCENARIOS = [
    {
        "id": "scenario_1",
        "name": "International Transfer Follow-ups",
        "description": "Customer asks about international transfers with ambiguous follow-ups",
        "turns": [
            {
                "query": "How do I send money abroad?",
                "expected_categories": ["transfer_into_account", "transfer_timing", "country_support"],
                "context_independent": True
            },
            {
                "query": "How long will it take?",
                "expected_categories": ["transfer_timing", "transfer_into_account"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to international transfer"
            },
            {
                "query": "Are there any fees?",
                "expected_categories": ["transfer_fee_charged", "country_support"],
                "context_dependent": True,
                "requires_context": "Should understand asking about international transfer fees"
            },
            {
                "query": "What if it's stuck as pending?",
                "expected_categories": ["pending_transfer", "transfer_timing"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to international transfer"
            }
        ],
        "success_criteria": "All follow-ups should be correctly understood in context"
    },

    {
        "id": "scenario_2",
        "name": "Card Payment Issues",
        "description": "Customer reports card decline and asks follow-ups",
        "turns": [
            {
                "query": "My card payment was declined at a store",
                "expected_categories": ["declined_card_payment", "card_acceptance"],
                "context_independent": True
            },
            {
                "query": "Why did this happen?",
                "expected_categories": ["declined_card_payment"],
                "context_dependent": True,
                "requires_context": "Should understand asking about reason for card decline"
            },
            {
                "query": "Can I fix it?",
                "expected_categories": ["declined_card_payment", "contactless_not_working"],
                "context_dependent": True,
                "requires_context": "Should understand asking how to resolve card decline"
            }
        ],
        "success_criteria": "System should maintain card payment context throughout"
    },

    {
        "id": "scenario_3",
        "name": "Transfer Not Showing",
        "description": "Customer reports missing transfer with vague follow-ups",
        "turns": [
            {
                "query": "I transferred money but it's not showing in my account",
                "expected_categories": ["transfer_into_account", "balance_not_updated_after_bank_transfer"],
                "context_independent": True
            },
            {
                "query": "When will it appear?",
                "expected_categories": ["transfer_timing", "transfer_into_account"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to the transfer"
            },
            {
                "query": "What if it still doesn't show tomorrow?",
                "expected_categories": ["balance_not_updated_after_bank_transfer", "pending_transfer"],
                "context_dependent": True,
                "requires_context": "Should maintain transfer context"
            }
        ],
        "success_criteria": "All pronouns and implicit references correctly resolved"
    },

    {
        "id": "scenario_4",
        "name": "ATM Withdrawal Problems",
        "description": "ATM issue with card-related follow-ups",
        "turns": [
            {
                "query": "The ATM didn't give me money but charged my account",
                "expected_categories": ["atm_support", "failed_transfer", "wrong_amount_of_cash_received"],
                "context_independent": True
            },
            {
                "query": "Should I cancel my card?",
                "expected_categories": ["lost_or_stolen_card", "cancel_transfer", "compromised_card"],
                "context_dependent": True,
                "requires_context": "Card cancellation in context of ATM issue"
            },
            {
                "query": "How do I get my money back?",
                "expected_categories": ["failed_transfer", "atm_support", "request_refund"],
                "context_dependent": True,
                "requires_context": "Should understand referring to ATM withdrawal"
            }
        ],
        "success_criteria": "Context switches from ATM to card should be handled"
    },

    {
        "id": "scenario_5",
        "name": "Top-up Questions",
        "description": "Customer asks about adding money with various follow-ups",
        "turns": [
            {
                "query": "How do I add money to my account?",
                "expected_categories": ["top_up_by_card_charge", "verify_top_up", "transfer_into_account", "topping_up_by_card"],
                "context_independent": True
            },
            {
                "query": "Is there a limit?",
                "expected_categories": ["top_up_limits"],
                "context_dependent": True,
                "requires_context": "Should understand asking about top-up limit"
            },
            {
                "query": "What if it fails?",
                "expected_categories": ["top_up_failed"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to top-up"
            },
            {
                "query": "Are there charges?",
                "expected_categories": ["top_up_by_card_charge", "top_up_by_bank_transfer_charge"],
                "context_dependent": True,
                "requires_context": "Should understand asking about top-up charges"
            }
        ],
        "success_criteria": "All follow-ups correctly interpreted in top-up context"
    },

    {
        "id": "scenario_6",
        "name": "Lost Card Scenario",
        "description": "Customer reports lost card and asks about replacement",
        "turns": [
            {
                "query": "I lost my card and need a new one",
                "expected_categories": ["lost_or_stolen_card", "card_arrival"],
                "context_independent": True
            },
            {
                "query": "How long until it arrives?",
                "expected_categories": ["card_arrival"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to replacement card"
            },
            {
                "query": "Can I cancel my pending transaction?",
                "expected_categories": ["cancel_transfer", "pending_cash_withdrawal"],
                "context_dependent": True,
                "requires_context": "Context shift to different topic"
            },
            {
                "query": "Will the new one have contactless?",
                "expected_categories": ["contactless_not_working", "card_arrival"],
                "context_dependent": True,
                "requires_context": "Should understand 'new one' refers to replacement card"
            }
        ],
        "success_criteria": "Handle topic shift and return to card context"
    },

    {
        "id": "scenario_7",
        "name": "Direct Debit Issues",
        "description": "Customer confused about direct debit with follow-ups",
        "turns": [
            {
                "query": "There's a payment I don't recognize",
                "expected_categories": ["direct_debit_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_payment_not_recognised"],
                "context_independent": True
            },
            {
                "query": "Could it be a direct debit?",
                "expected_categories": ["direct_debit_payment_not_recognised"],
                "context_dependent": True,
                "requires_context": "Should connect to unrecognized payment"
            },
            {
                "query": "How do I check?",
                "expected_categories": ["direct_debit_payment_not_recognised"],
                "context_dependent": True,
                "requires_context": "Should understand checking direct debits"
            }
        ],
        "success_criteria": "Maintain payment investigation context"
    },

    {
        "id": "scenario_8",
        "name": "Exchange Rate Questions",
        "description": "Currency exchange with comparative questions",
        "turns": [
            {
                "query": "What's the exchange rate for USD?",
                "expected_categories": ["exchange_rate", "card_payment_wrong_exchange_rate"],
                "context_independent": True
            },
            {
                "query": "Is it better than yesterday?",
                "expected_categories": ["exchange_rate"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to USD exchange rate"
            },
            {
                "query": "What about EUR?",
                "expected_categories": ["exchange_rate"],
                "context_dependent": True,
                "requires_context": "Topic shift to different currency but same domain"
            }
        ],
        "success_criteria": "Handle currency comparisons and topic shifts"
    },

    {
        "id": "scenario_9",
        "name": "Virtual Card Setup",
        "description": "Setting up virtual card with activation questions",
        "turns": [
            {
                "query": "Can I get a virtual card?",
                "expected_categories": ["getting_virtual_card", "get_disposable_virtual_card"],
                "context_independent": True
            },
            {
                "query": "Do I need to activate it?",
                "expected_categories": ["activate_my_card", "getting_virtual_card"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to virtual card"
            },
            {
                "query": "Where is it accepted?",
                "expected_categories": ["card_acceptance", "getting_virtual_card", "country_support"],
                "context_dependent": True,
                "requires_context": "Should understand asking about virtual card acceptance"
            },
            {
                "query": "How do I verify a transaction?",
                "expected_categories": ["verify_top_up", "card_payment_wrong_exchange_rate"],
                "context_dependent": True,
                "requires_context": "Related to virtual card transactions"
            }
        ],
        "success_criteria": "Maintain virtual card context across multiple questions"
    },

    {
        "id": "scenario_10",
        "name": "Account Balance Confusion",
        "description": "Customer confused about balance discrepancy",
        "turns": [
            {
                "query": "My balance seems wrong",
                "expected_categories": ["balance_not_updated_after_bank_transfer", "wrong_amount_of_cash_received"],
                "context_independent": True
            },
            {
                "query": "Could it be the exchange rate?",
                "expected_categories": ["card_payment_wrong_exchange_rate", "exchange_rate"],
                "context_dependent": True,
                "requires_context": "Possible cause for balance discrepancy"
            },
            {
                "query": "Or maybe a pending withdrawal?",
                "expected_categories": ["pending_cash_withdrawal", "pending_transfer"],
                "context_dependent": True,
                "requires_context": "Another possible cause"
            }
        ],
        "success_criteria": "Handle hypothetical reasoning about balance issues"
    },

    {
        "id": "scenario_11",
        "name": "Mixed Topics Conversation",
        "description": "Customer switches between unrelated topics",
        "turns": [
            {
                "query": "How do I top up my account?",
                "expected_categories": ["top_up_by_card_charge", "verify_top_up", "transfer_into_account", "topping_up_by_card"],
                "context_independent": True
            },
            {
                "query": "Actually, when will my new card arrive?",
                "expected_categories": ["card_arrival"],
                "context_dependent": False,
                "requires_context": "Complete topic change - should recognize new context"
            },
            {
                "query": "And can I use it abroad?",
                "expected_categories": ["card_acceptance", "country_support", "activate_my_card"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to new card, not top-up"
            }
        ],
        "success_criteria": "Handle abrupt topic changes correctly"
    },

    {
        "id": "scenario_12",
        "name": "Pronoun-Heavy Follow-ups",
        "description": "Multiple pronouns requiring careful tracking",
        "turns": [
            {
                "query": "I want to send £100 to my friend",
                "expected_categories": ["transfer_into_account", "transfer_timing", "receiving_money"],
                "context_independent": True
            },
            {
                "query": "How long will that take?",
                "expected_categories": ["transfer_timing"],
                "context_dependent": True,
                "requires_context": "'that' refers to the £100 transfer"
            },
            {
                "query": "What if they don't receive it?",
                "expected_categories": ["failed_transfer", "transfer_into_account", "transfer_not_received_by_recipient"],
                "context_dependent": True,
                "requires_context": "'they' refers to friend, 'it' refers to £100 transfer"
            },
            {
                "query": "Can I cancel that?",
                "expected_categories": ["cancel_transfer"],
                "context_dependent": True,
                "requires_context": "'that' refers to the transfer"
            }
        ],
        "success_criteria": "All pronouns correctly resolved to their referents"
    },

    {
        "id": "scenario_13",
        "name": "Implicit Comparison",
        "description": "Customer makes implicit comparisons",
        "turns": [
            {
                "query": "How much does a transfer cost?",
                "expected_categories": ["transfer_fee_charged"],
                "context_independent": True
            },
            {
                "query": "What about international?",
                "expected_categories": ["transfer_fee_charged", "country_support"],
                "context_dependent": True,
                "requires_context": "Implicit comparison - international transfer cost vs regular"
            },
            {
                "query": "Is that more expensive?",
                "expected_categories": ["transfer_fee_charged", "country_support"],
                "context_dependent": True,
                "requires_context": "Comparing international vs domestic transfer fees"
            }
        ],
        "success_criteria": "Handle implicit comparisons and contrasts"
    },

    {
        "id": "scenario_14",
        "name": "Escalating Issue",
        "description": "Problem escalates through conversation",
        "turns": [
            {
                "query": "My contactless isn't working",
                "expected_categories": ["contactless_not_working"],
                "context_independent": True
            },
            {
                "query": "Now the whole card won't work",
                "expected_categories": ["card_acceptance", "declined_card_payment", "card_not_working"],
                "context_dependent": True,
                "requires_context": "Problem escalation from contactless to entire card"
            },
            {
                "query": "Should I report it as lost?",
                "expected_categories": ["lost_or_stolen_card"],
                "context_dependent": True,
                "requires_context": "Should understand 'it' refers to malfunctioning card"
            }
        ],
        "success_criteria": "Track problem evolution through conversation"
    },

    {
        "id": "scenario_15",
        "name": "Clarification Questions",
        "description": "Customer asks for clarification on previous answer",
        "turns": [
            {
                "query": "How do I activate my card?",
                "expected_categories": ["activate_my_card"],
                "context_independent": True
            },
            {
                "query": "What do you mean by PIN?",
                "expected_categories": ["activate_my_card", "get_physical_card", "change_pin"],
                "context_dependent": True,
                "requires_context": "Clarification question about activation process"
            },
            {
                "query": "Where do I find that?",
                "expected_categories": ["activate_my_card", "card_arrival", "get_physical_card", "verify_source_of_funds"],
                "context_dependent": True,
                "requires_context": "'that' refers to PIN mentioned in clarification"
            }
        ],
        "success_criteria": "Handle meta-questions about system responses"
    }
]


def evaluate_conversation_scenario(pipeline, scenario, verbose=False):
    """
    Evaluate a single conversation scenario

    Args:
        pipeline: RAGPipeline instance with contextual retriever
        scenario: Scenario dict from CONVERSATION_TEST_SCENARIOS
        verbose: Whether to print detailed results

    Returns:
        dict with evaluation results
    """
    pipeline.reset_conversation()

    results = {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "total_turns": len(scenario["turns"]),
        "successful_turns": 0,
        "failed_turns": [],
        "turn_results": []
    }

    for i, turn in enumerate(scenario["turns"]):
        query = turn["query"]
        expected_cats = turn["expected_categories"]
        is_context_dependent = turn.get("context_dependent", False)

        # Get response
        response = pipeline.query(query, n_results=5)
        retrieved_cats = [s["category"] for s in response["sources"]]

        # Check if any expected category is in top-3 results
        top_3_cats = retrieved_cats[:3]
        success = any(cat in top_3_cats for cat in expected_cats)

        turn_result = {
            "turn": i + 1,
            "query": query,
            "context_dependent": is_context_dependent,
            "expected_categories": expected_cats,
            "retrieved_top_3": top_3_cats,
            "success": success,
            "requires_context": turn.get("requires_context", "N/A")
        }

        results["turn_results"].append(turn_result)

        if success:
            results["successful_turns"] += 1
        else:
            results["failed_turns"].append(turn_result)

    results["success_rate"] = results["successful_turns"] / results["total_turns"]
    results["passed"] = results["success_rate"] >= 0.75  # 75% threshold

    if verbose:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*80}")
        print(f"Description: {scenario['description']}")
        print(f"\nResults: {results['successful_turns']}/{results['total_turns']} turns successful")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Overall: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")

        for turn_result in results["turn_results"]:
            status = "✓" if turn_result["success"] else "✗"
            context_flag = "🔗" if turn_result["context_dependent"] else "  "
            print(f"\n{status} {context_flag} Turn {turn_result['turn']}: {turn_result['query']}")
            if turn_result["requires_context"] != "N/A":
                print(f"   Context requirement: {turn_result['requires_context']}")
            print(f"   Expected: {', '.join(turn_result['expected_categories'])}")
            print(f"   Retrieved: {', '.join(turn_result['retrieved_top_3'])}")

    return results


def evaluate_all_scenarios(pipeline, verbose=False):
    """
    Evaluate all conversation scenarios

    Args:
        pipeline: RAGPipeline instance
        verbose: Whether to print detailed results

    Returns:
        dict with overall evaluation results
    """
    all_results = []

    for scenario in CONVERSATION_TEST_SCENARIOS:
        result = evaluate_conversation_scenario(pipeline, scenario, verbose=verbose)
        all_results.append(result)

    # Calculate overall statistics
    total_scenarios = len(all_results)
    passed_scenarios = sum(1 for r in all_results if r["passed"])
    total_turns = sum(r["total_turns"] for r in all_results)
    successful_turns = sum(r["successful_turns"] for r in all_results)

    context_dependent_turns = []
    context_independent_turns = []

    for result in all_results:
        for turn in result["turn_results"]:
            if turn["context_dependent"]:
                context_dependent_turns.append(turn["success"])
            else:
                context_independent_turns.append(turn["success"])

    overall = {
        "total_scenarios": total_scenarios,
        "passed_scenarios": passed_scenarios,
        "scenario_success_rate": passed_scenarios / total_scenarios,
        "total_turns": total_turns,
        "successful_turns": successful_turns,
        "overall_turn_success_rate": successful_turns / total_turns,
        "context_dependent_accuracy": sum(context_dependent_turns) / len(context_dependent_turns) if context_dependent_turns else 0,
        "context_independent_accuracy": sum(context_independent_turns) / len(context_independent_turns) if context_independent_turns else 0,
        "scenario_results": all_results
    }

    if verbose:
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"\nScenarios: {passed_scenarios}/{total_scenarios} passed ({overall['scenario_success_rate']:.1%})")
        print(f"Total Turns: {successful_turns}/{total_turns} successful ({overall['overall_turn_success_rate']:.1%})")
        print(f"\nContext-Dependent Turns: {sum(context_dependent_turns)}/{len(context_dependent_turns)} ({overall['context_dependent_accuracy']:.1%})")
        print(f"Context-Independent Turns: {sum(context_independent_turns)}/{len(context_independent_turns)} ({overall['context_independent_accuracy']:.1%})")

        print(f"\n{'='*80}")
        print("SCENARIO BREAKDOWN")
        print(f"{'='*80}")
        for result in all_results:
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"{status} - {result['scenario_name']}: {result['successful_turns']}/{result['total_turns']} ({result['success_rate']:.0%})")

    return overall
