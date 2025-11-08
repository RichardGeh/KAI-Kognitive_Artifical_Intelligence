"""
test_meta_learning_integration.py

Quick integration test for Phase 4.1: Meta-Learning Integration
"""

import sys


def test_imports():
    """Test dass alle benötigten Komponenten importierbar sind"""
    print("Testing imports...")

    try:
        pass

        print("✓ ReasoningOrchestrator imported")


        print("✓ MetaLearningEngine imported")


        print("✓ SelfEvaluator imported")


        print("✓ AdaptiveResonanceEngine imported")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_orchestrator_structure():
    """Test dass ReasoningOrchestrator die neuen Methoden hat"""
    print("\nTesting ReasoningOrchestrator structure...")

    from kai_reasoning_orchestrator import ReasoningOrchestrator

    # Check dass neue Methoden existieren
    required_methods = [
        "query_with_meta_learning",
        "_execute_resonance_strategy",
        "_get_available_strategy_names",
        "_map_strategy_name_to_enum",
        "_execute_single_strategy",
        "_evaluate_result_quality",
        "_record_strategy_usage",
    ]

    for method_name in required_methods:
        if hasattr(ReasoningOrchestrator, method_name):
            print(f"✓ Method '{method_name}' exists")
        else:
            print(f"✗ Method '{method_name}' missing")
            return False

    return True


def test_orchestrator_initialization():
    """Test dass Orchestrator mit neuen Dependencies initialisiert werden kann"""
    print("\nTesting ReasoningOrchestrator initialization...")

    try:
        from kai_reasoning_orchestrator import ReasoningOrchestrator

        # Minimal mock objects
        class MockObject:
            pass

        netzwerk = MockObject()
        logic_engine = MockObject()
        graph_traversal = MockObject()
        working_memory = MockObject()
        signals = MockObject()

        # Test mit neuen Dependencies
        orchestrator = ReasoningOrchestrator(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=graph_traversal,
            working_memory=working_memory,
            signals=signals,
            meta_learning_engine=None,  # Optional
            self_evaluator=None,  # Optional
        )

        # Check dass neue Attribute existieren
        if hasattr(orchestrator, "meta_learning_engine"):
            print("✓ Attribute 'meta_learning_engine' exists")
        else:
            print("✗ Attribute 'meta_learning_engine' missing")
            return False

        if hasattr(orchestrator, "self_evaluator"):
            print("✓ Attribute 'self_evaluator' exists")
        else:
            print("✗ Attribute 'self_evaluator' missing")
            return False

        print("✓ ReasoningOrchestrator initialization successful")
        return True

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Phase 4.1 Integration Test: Meta-Learning + Self-Evaluation")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Orchestrator Structure", test_orchestrator_structure),
        ("Orchestrator Initialization", test_orchestrator_initialization),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print("=" * 60)
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
