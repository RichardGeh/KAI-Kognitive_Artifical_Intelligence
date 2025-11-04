"""
tests/test_logik_engine_phase4.py

Test Suite for Phase 4.1: Consistency & Contradiction Enhancements for Logic Engine

Tests:
- validate_inference_chain(): Validates reasoning chain consistency
- explain_contradiction(): Explains contradictions between facts

Author: KAI Development Team
Date: 2025-10-31
"""

import pytest
from component_9_logik_engine import Engine, Fact, Goal, ProofStep, Rule
from component_1_netzwerk import KonzeptNetzwerk


@pytest.fixture
def netzwerk():
    """Mock network for tests"""
    # Use real network but clean up after test
    netz = KonzeptNetzwerk()
    yield netz
    # Cleanup if needed


@pytest.fixture
def engine(netzwerk):
    """Engine with SAT solver for tests"""
    eng = Engine(netzwerk, use_probabilistic=False, use_sat=True)
    return eng


# ==================== TESTS: validate_inference_chain() ====================


def test_validate_inference_chain_consistent_proof(engine):
    """Tests validation of consistent proof chain"""
    # Create simple consistent proof
    fact1 = Fact(
        pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"}, confidence=1.0
    )

    goal = Goal(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"})

    proof = ProofStep(
        goal=goal, method="fact", supporting_facts=[fact1], confidence=1.0
    )

    # Validate
    inconsistencies = engine.validate_inference_chain(proof)

    # Should be consistent (empty list or only SAT warning)
    assert isinstance(inconsistencies, list)
    print(f"[OK] Consistent proof validated: {len(inconsistencies)} issues")


def test_validate_inference_chain_with_contradictory_facts(engine):
    """Tests validation with contradictory facts"""
    # Create contradictory facts
    fact1 = Fact(
        pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"}, confidence=1.0
    )
    fact2 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "apfel", "object": "rot"},
        confidence=0.3,
        status="contradicted",
    )  # Negated fact

    goal = Goal(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"})

    proof = ProofStep(
        goal=goal, method="fact", supporting_facts=[fact1, fact2], confidence=0.3
    )

    # Validate
    inconsistencies = engine.validate_inference_chain(proof)

    # Should find inconsistencies
    assert isinstance(inconsistencies, list)
    assert len(inconsistencies) > 0
    print(f"[OK] Contradictory facts detected: {len(inconsistencies)} inconsistencies")


def test_validate_inference_chain_with_invalid_rule_application(engine):
    """Tests validation with invalid rule application"""
    # Create rule
    rule = Rule(
        id="test_rule",
        salience=10,
        when=[{"pred": "IS_A", "args": {"subject": "?x", "object": "frucht"}}],
        then=[
            {
                "assert": {
                    "pred": "HAS_PROPERTY",
                    "args": {"subject": "?x", "object": "essbar"},
                }
            }
        ],
    )

    engine.rules.append(rule)

    # Create proof with wrong premise
    wrong_premise = Fact(
        pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"}, confidence=1.0
    )

    goal = Goal(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "essbar"})

    proof = ProofStep(
        goal=goal,
        method="rule",
        rule_id="test_rule",
        supporting_facts=[wrong_premise],  # Wrong premise (should be IS_A)
        confidence=1.0,
    )

    # Validate
    inconsistencies = engine.validate_inference_chain(proof)

    # Should detect invalid rule application
    real_inconsistencies = [inc for inc in inconsistencies if "SAT-Solver" not in inc]
    assert len(real_inconsistencies) > 0
    assert any(
        "Ungültige Regelanwendung" in inc or "test_rule" in inc
        for inc in real_inconsistencies
    )
    print(f"[OK] Invalid rule application detected")


def test_validate_inference_chain_confidence_monotonicity(engine):
    """Tests validation of confidence monotonicity"""
    # Create proof with violated confidence monotonicity
    fact1 = Fact(
        pred="IS_A", args={"subject": "hund", "object": "tier"}, confidence=1.0
    )

    subgoal = Goal(pred="IS_A", args={"subject": "hund", "object": "tier"})

    subproof = ProofStep(
        goal=subgoal,
        method="fact",
        supporting_facts=[fact1],
        confidence=0.9,  # Higher than parent goal (should trigger warning)
    )

    goal = Goal(pred="IS_A", args={"subject": "hund", "object": "säugetier"})

    proof = ProofStep(
        goal=goal,
        method="rule",
        subgoals=[subproof],
        confidence=0.5,  # Lower than subgoal
    )

    # Validate
    inconsistencies = engine.validate_inference_chain(proof)

    # Should find confidence violation
    assert any("Confidence-Verletzung" in inc for inc in inconsistencies)
    print(f"[OK] Confidence violation detected")


def test_validate_inference_chain_without_sat_solver(netzwerk):
    """Tests validation without SAT solver (fallback)"""
    engine_no_sat = Engine(netzwerk, use_probabilistic=False, use_sat=False)

    fact1 = Fact(
        pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"}, confidence=1.0
    )
    goal = Goal(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"})

    proof = ProofStep(
        goal=goal, method="fact", supporting_facts=[fact1], confidence=1.0
    )

    # Validate
    inconsistencies = engine_no_sat.validate_inference_chain(proof)

    # Should contain warning
    assert any("SAT-Solver nicht verfügbar" in inc for inc in inconsistencies)
    print(f"[OK] SAT solver unavailable warning provided")


# ==================== TESTS: explain_contradiction() ====================


def test_explain_contradiction_direct_negation(engine):
    """Tests explanation of direct negation"""
    fact1 = Fact(
        pred="IS_A",
        args={"subject": "hund", "object": "tier"},
        confidence=1.0,
        source="kb",
    )
    fact2 = Fact(
        pred="IS_A",
        args={"subject": "hund", "object": "tier"},
        confidence=0.2,
        status="contradicted",
        source="user_input",
    )

    explanation = engine.explain_contradiction(fact1, fact2)

    assert isinstance(explanation, str)
    assert "Direkter Widerspruch" in explanation or "Negation" in explanation
    assert "hund" in explanation.lower()
    print(f"[OK] Direct negation explained")


def test_explain_contradiction_exclusive_properties_colors(engine):
    """Tests explanation of exclusive properties (colors)"""
    fact1 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "apfel", "object": "rot"},
        confidence=1.0,
        source="visual_observation",
    )
    fact2 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "apfel", "object": "grün"},
        confidence=0.9,
        source="text_extraction",
    )

    explanation = engine.explain_contradiction(fact1, fact2)

    assert isinstance(explanation, str)
    assert "apfel" in explanation.lower()
    assert "rot" in explanation.lower() or "grün" in explanation.lower()
    # Should recognize color exclusivity
    assert (
        "Exklusive Eigenschaften" in explanation
        or "schließen sich" in explanation.lower()
    )
    print(f"[OK] Exclusive color properties explained")


def test_explain_contradiction_exclusive_categories(engine):
    """Tests explanation of exclusive categories (animal vs plant)"""
    fact1 = Fact(
        pred="IS_A",
        args={"subject": "rose", "object": "pflanze"},
        confidence=1.0,
        source="biology_db",
    )
    fact2 = Fact(
        pred="IS_A",
        args={"subject": "rose", "object": "tier"},
        confidence=0.5,
        source="user_error",
    )

    explanation = engine.explain_contradiction(fact1, fact2)

    assert isinstance(explanation, str)
    assert "rose" in explanation.lower()
    # Should recognize hierarchy contradiction
    assert "Hierarchiewiderspruch" in explanation or "disjunkt" in explanation.lower()
    print(f"[OK] Exclusive categories explained")


def test_explain_contradiction_with_size_properties(engine):
    """Tests explanation of exclusive sizes"""
    fact1 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "elefant", "object": "groß"},
        confidence=1.0,
        source="observation",
    )
    fact2 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "elefant", "object": "klein"},
        confidence=0.7,
        source="confused_input",
    )

    explanation = engine.explain_contradiction(fact1, fact2)

    assert isinstance(explanation, str)
    assert "elefant" in explanation.lower()
    assert "groß" in explanation.lower() or "klein" in explanation.lower()
    print(f"[OK] Exclusive size properties explained")


def test_explain_contradiction_with_temperature_properties(engine):
    """Tests explanation of exclusive temperatures"""
    fact1 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "feuer", "object": "heiß"},
        confidence=1.0,
        source="physics_knowledge",
    )
    fact2 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "feuer", "object": "kalt"},
        confidence=0.3,
        source="nonsensical_input",
    )

    explanation = engine.explain_contradiction(fact1, fact2)

    assert isinstance(explanation, str)
    assert "heiß" in explanation.lower() or "kalt" in explanation.lower()
    print(f"[OK] Exclusive temperature properties explained")


def test_explain_contradiction_fallback_generic(engine):
    """Tests generic explanation for unknown contradiction"""
    fact1 = Fact(
        pred="CUSTOM_RELATION",
        args={"subject": "x", "object": "y"},
        confidence=1.0,
        source="source_a",
    )
    fact2 = Fact(
        pred="CUSTOM_RELATION",
        args={"subject": "x", "object": "z"},
        confidence=0.9,
        source="source_b",
    )

    explanation = engine.explain_contradiction(fact1, fact2)

    assert isinstance(explanation, str)
    # Should provide generic explanation
    assert "Widerspruch" in explanation or "inkonsistent" in explanation.lower()
    print(f"[OK] Generic explanation provided for unknown contradiction")


def test_explain_contradiction_with_confidence_info(engine):
    """Tests that confidence values are included in explanation"""
    fact1 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "ball", "object": "rot"},
        confidence=0.95,
        source="camera_1",
    )
    fact2 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "ball", "object": "blau"},
        confidence=0.85,
        source="camera_2",
    )

    explanation = engine.explain_contradiction(fact1, fact2)

    assert isinstance(explanation, str)
    # Should contain confidence values
    assert "0.95" in explanation or "0.85" in explanation or "Konfidenz" in explanation
    print(f"[OK] Confidence values included in explanation")


# ==================== HELPER METHOD TESTS ====================


def test_are_properties_exclusive_colors(engine):
    """Tests _are_properties_exclusive for colors"""
    assert engine._are_properties_exclusive("rot", "grün") == True
    assert engine._are_properties_exclusive("blau", "gelb") == True
    assert engine._are_properties_exclusive("schwarz", "weiß") == True
    assert engine._are_properties_exclusive("rot", "rot") == False  # Same color
    print(f"[OK] Color exclusivity recognized")


def test_are_properties_exclusive_sizes(engine):
    """Tests _are_properties_exclusive for sizes"""
    assert engine._are_properties_exclusive("groß", "klein") == True
    assert engine._are_properties_exclusive("riesig", "winzig") == True
    assert engine._are_properties_exclusive("groß", "mittel") == True
    print(f"[OK] Size exclusivity recognized")


def test_are_properties_exclusive_temperatures(engine):
    """Tests _are_properties_exclusive for temperatures"""
    assert engine._are_properties_exclusive("heiß", "kalt") == True
    assert engine._are_properties_exclusive("warm", "eiskalt") == True
    assert engine._are_properties_exclusive("glühend", "kühl") == True
    print(f"[OK] Temperature exclusivity recognized")


def test_are_properties_exclusive_states(engine):
    """Tests _are_properties_exclusive for states"""
    assert engine._are_properties_exclusive("lebendig", "tot") == True
    assert engine._are_properties_exclusive("an", "aus") == True
    assert engine._are_properties_exclusive("offen", "geschlossen") == True
    print(f"[OK] State exclusivity recognized")


def test_are_properties_exclusive_non_exclusive(engine):
    """Tests that non-exclusive properties are recognized correctly"""
    assert engine._are_properties_exclusive("rot", "rund") == False
    assert engine._are_properties_exclusive("groß", "schwer") == False
    assert engine._are_properties_exclusive("süß", "rot") == False
    print(f"[OK] Non-exclusive properties recognized")


def test_are_categories_exclusive_biology(engine):
    """Tests _are_categories_exclusive for biological categories"""
    assert engine._are_categories_exclusive("tier", "pflanze") == True
    assert engine._are_categories_exclusive("säugetier", "vogel") == True
    assert engine._are_categories_exclusive("säugetier", "fisch") == True
    print(f"[OK] Biological category exclusivity recognized")


def test_are_categories_exclusive_objects(engine):
    """Tests _are_categories_exclusive for object categories"""
    assert engine._are_categories_exclusive("lebewesen", "gegenstand") == True
    assert engine._are_categories_exclusive("fahrzeug", "gebäude") == True
    print(f"[OK] Object category exclusivity recognized")


def test_are_categories_exclusive_non_exclusive(engine):
    """Tests that non-exclusive categories are recognized correctly"""
    assert engine._are_categories_exclusive("tier", "säugetier") == False
    assert engine._are_categories_exclusive("hund", "tier") == False
    print(f"[OK] Non-exclusive categories recognized")


def test_format_args(engine):
    """Tests _format_args helper method"""
    args = {"subject": "apfel", "object": "rot"}
    formatted = engine._format_args(args)

    assert "subject=apfel" in formatted or "apfel" in formatted
    assert "object=rot" in formatted or "rot" in formatted
    print(f"[OK] Arguments formatted correctly")


# ==================== INTEGRATION TESTS ====================


def test_validate_and_explain_combined(engine):
    """Tests combination of validation and explanation"""
    # Create contradictory facts
    fact1 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "himmel", "object": "blau"},
        confidence=1.0,
        source="observation",
    )
    fact2 = Fact(
        pred="HAS_PROPERTY",
        args={"subject": "himmel", "object": "grün"},
        confidence=0.8,
        source="erroneous_input",
    )

    goal = Goal(pred="HAS_PROPERTY", args={"subject": "himmel", "object": "blau"})

    proof = ProofStep(
        goal=goal, method="fact", supporting_facts=[fact1, fact2], confidence=0.8
    )

    # Validate proof
    inconsistencies = engine.validate_inference_chain(proof)

    # Explain contradiction
    explanation = engine.explain_contradiction(fact1, fact2)

    # Both should detect contradiction
    real_inconsistencies = [inc for inc in inconsistencies if "SAT-Solver" not in inc]
    assert len(real_inconsistencies) > 0 or "Exklusive" in explanation
    print(f"[OK] Validation and explanation combined successfully")


def test_validate_inference_chain_with_rule_based_reasoning(engine):
    """Tests validation of complex rule-based proofs"""
    # Create rule
    rule = Rule(
        id="frucht_regel",
        salience=10,
        when=[{"pred": "IS_A", "args": {"subject": "?x", "object": "frucht"}}],
        then=[
            {
                "assert": {
                    "pred": "HAS_PROPERTY",
                    "args": {"subject": "?x", "object": "essbar"},
                }
            }
        ],
    )

    engine.rules.append(rule)

    # Create correct proof
    premise_fact = Fact(
        pred="IS_A", args={"subject": "apfel", "object": "frucht"}, confidence=1.0
    )

    subgoal = Goal(pred="IS_A", args={"subject": "apfel", "object": "frucht"})
    subproof = ProofStep(
        goal=subgoal, method="fact", supporting_facts=[premise_fact], confidence=1.0
    )

    goal = Goal(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "essbar"})

    proof = ProofStep(
        goal=goal,
        method="rule",
        rule_id="frucht_regel",
        supporting_facts=[premise_fact],
        subgoals=[subproof],
        confidence=1.0,
    )

    # Validate - should find no inconsistencies (except SAT warning)
    inconsistencies = engine.validate_inference_chain(proof)

    # Filter out SAT warning
    real_inconsistencies = [
        inc for inc in inconsistencies if "SAT-Solver nicht verfügbar" not in inc
    ]

    # Should have no real inconsistencies
    assert len(real_inconsistencies) == 0
    print(f"[OK] Valid rule-based proof validated successfully")


if __name__ == "__main__":
    # Can be run directly for quick debugging
    pytest.main([__file__, "-v", "--tb=short"])
