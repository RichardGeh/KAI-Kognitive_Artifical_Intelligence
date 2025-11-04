"""
Tests für Semantic Contradiction Detection mit SAT-Solver und Ontologie-Constraints.

Testet die Integration zwischen component_9_logik_engine (SAT-Solver),
component_30_sat_solver (DPLL) und component_31_ontology_constraints (Constraint-Generator).

Fokus:
- IS_A Exklusivität für Geschwister-Konzepte (z.B. tier vs. pflanze)
- Property-Konflikte (z.B. rot vs. grün, groß vs. klein)
- Location-Konflikte (z.B. deutschland vs. frankreich)
- Integration zwischen Engine-Fakten und Graph-Fakten

Author: KAI Development Team
Date: 2025-10-31
"""

import pytest
from unittest.mock import Mock, MagicMock

from component_9_logik_engine import Engine, Fact
from component_1_netzwerk import KonzeptNetzwerk


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_netzwerk():
    """
    Mock Netzwerk mit vordefinierter Ontologie-Hierarchie.
    """
    mock = Mock(spec=KonzeptNetzwerk)
    mock.driver = MagicMock()

    # Create proper mock for driver.session() context manager
    mock_session = MagicMock()
    mock.driver.session.return_value.__enter__.return_value = mock_session
    mock.driver.session.return_value.__exit__.return_value = None

    # Mock query_graph_for_facts für verschiedene Entitäten
    def query_facts(concept):
        knowledge_base = {
            "hund": {
                "IS_A": ["säugetier", "tier"],
                "HAS_PROPERTY": ["groß", "braun"],
                "LOCATED_IN": ["deutschland"],
            },
            "katze": {
                "IS_A": ["säugetier", "tier"],
                "HAS_PROPERTY": ["klein", "schwarz"],
            },
            "apfel": {
                "IS_A": ["frucht", "obst", "pflanze"],
                "HAS_PROPERTY": ["rot", "süß"],
            },
            "rose": {"IS_A": ["blume", "pflanze"], "HAS_PROPERTY": ["rot", "duftend"]},
            "säugetier": {"IS_A": ["tier", "lebewesen"]},
            "tier": {"IS_A": ["lebewesen"]},
            "pflanze": {"IS_A": ["lebewesen"]},
            "berlin": {"PART_OF": ["deutschland"]},
            "münchen": {"PART_OF": ["deutschland"]},
        }
        return knowledge_base.get(concept.lower(), {})

    mock.query_graph_for_facts.side_effect = query_facts

    # Mock session.run für Ontologie-Queries
    def mock_run(query, **params):
        result_mock = MagicMock()

        # IS_A Hierarchie-Query (für Geschwister-Konzepte)
        if "MATCH (parent:Konzept)<-[:IS_A]-(child:Konzept)" in query:
            # Simuliere IS_A Hierarchie
            records = [
                {"parent": "lebewesen", "children": ["tier", "pflanze", "pilz"]},
                {"parent": "tier", "children": ["säugetier", "vogel", "fisch"]},
                {"parent": "pflanze", "children": ["blume", "baum", "frucht"]},
            ]
            result_mock.__iter__.return_value = iter(
                [{"parent": r["parent"], "children": r["children"]} for r in records]
            )

        # PART_OF Hierarchie-Query (für Locations)
        elif "MATCH (parent:Konzept)<-[:PART_OF]-(child:Konzept)" in query:
            records = [
                {"parent": "deutschland", "children": ["berlin", "münchen", "hamburg"]},
                {"parent": "frankreich", "children": ["paris", "lyon", "marseille"]},
            ]
            result_mock.__iter__.return_value = iter(
                [{"parent": r["parent"], "children": r["children"]} for r in records]
            )

        # Location Hierarchie-Check (für spezifische Location-Paare)
        elif "MATCH path = (a:Konzept" in query and "PART_OF" in query:
            loc1 = params.get("loc1", "")
            loc2 = params.get("loc2", "")

            # berlin und münchen sind beide in deutschland (hierarchisch verwandt)
            if (loc1 == "berlin" and loc2 == "deutschland") or (
                loc1 == "deutschland" and loc2 == "berlin"
            ):
                result_mock.single.return_value = {"count": 1}
            elif (loc1 == "münchen" and loc2 == "deutschland") or (
                loc1 == "deutschland" and loc2 == "münchen"
            ):
                result_mock.single.return_value = {"count": 1}
            else:
                result_mock.single.return_value = {"count": 0}

        else:
            result_mock.__iter__.return_value = iter([])
            result_mock.single.return_value = None

        return result_mock

    mock_session.run.side_effect = mock_run

    return mock


@pytest.fixture
def engine(mock_netzwerk):
    """
    Engine mit SAT-Solver und Ontologie-Constraints.
    """
    return Engine(netzwerk=mock_netzwerk, use_sat=True)


# ============================================================================
# TESTS: IS_A Exclusivity (Semantic Contradictions)
# ============================================================================


def test_detects_is_a_contradiction_tier_vs_pflanze(engine):
    """
    Test: SAT-Solver erkennt semantischen Widerspruch zwischen 'tier' und 'pflanze'.

    Dies ist der Haupttest für die neue Funktionalität.
    Vorher: SAT-Solver erkannte dies NICHT (keine semantischen Constraints)
    Nachher: SAT-Solver erkennt dies (Ontologie-Constraints aktiv)
    """
    facts = [
        Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
        Fact(pred="IS_A", args={"subject": "hund", "object": "pflanze"}),
    ]

    # check_consistency sollte False zurückgeben (Widerspruch erkannt)
    is_consistent = engine.check_consistency(facts)

    assert is_consistent == False, (
        "SAT-Solver sollte erkennen, dass 'tier' und 'pflanze' sich ausschließen "
        "(Geschwister unter 'lebewesen')"
    )
    print("[OK] IS_A Widerspruch erkannt: tier != pflanze")


def test_detects_is_a_contradiction_säugetier_vs_vogel(engine):
    """
    Test: Widerspruch zwischen 'säugetier' und 'vogel' (beide Kinder von 'tier').
    """
    facts = [
        Fact(pred="IS_A", args={"subject": "entity", "object": "säugetier"}),
        Fact(pred="IS_A", args={"subject": "entity", "object": "vogel"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert (
        is_consistent == False
    ), "SAT-Solver sollte erkennen, dass 'säugetier' und 'vogel' sich ausschließen"
    print("[OK] IS_A Widerspruch erkannt: saeugetier != vogel")


def test_allows_hierarchical_is_a(engine):
    """
    Test: Hierarchische IS_A-Beziehungen sind erlaubt.

    hund IS_A säugetier IS_A tier IS_A lebewesen → Konsistent
    """
    facts = [
        Fact(pred="IS_A", args={"subject": "hund", "object": "säugetier"}),
        Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
        Fact(pred="IS_A", args={"subject": "hund", "object": "lebewesen"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert is_consistent == True, "Hierarchische IS_A-Beziehungen sollten erlaubt sein"
    print(
        "[OK] Hierarchische IS_A-Beziehungen erlaubt: hund -> saeugetier -> tier -> lebewesen"
    )


# ============================================================================
# TESTS: Property Conflicts
# ============================================================================


def test_detects_property_conflict_colors(engine):
    """
    Test: Widersprüchliche Farben (rot vs. grün) werden erkannt.
    """
    facts = [
        Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"}),
        Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "grün"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert (
        is_consistent == False
    ), "SAT-Solver sollte erkennen, dass 'rot' und 'grün' sich ausschließen"
    print("[OK] Property Conflict erkannt: rot != gruen")


def test_detects_property_conflict_sizes(engine):
    """
    Test: Widersprüchliche Größen (groß vs. klein) werden erkannt.
    """
    facts = [
        Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "groß"}),
        Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "klein"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert (
        is_consistent == False
    ), "SAT-Solver sollte erkennen, dass 'groß' und 'klein' sich ausschließen"
    print("[OK] Property Conflict erkannt: gross != klein")


def test_allows_compatible_properties(engine):
    """
    Test: Kompatible Properties sind erlaubt.
    """
    facts = [
        Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "groß"}),
        Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "braun"}),
        Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "freundlich"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert is_consistent == True, "Kompatible Properties sollten erlaubt sein"
    print("[OK] Kompatible Properties erlaubt: groß + braun + freundlich")


# ============================================================================
# TESTS: Location Conflicts
# ============================================================================


def test_detects_location_conflict_deutschland_vs_frankreich(engine):
    """
    Test: Widersprüchliche Locations (deutschland vs. frankreich) werden erkannt.
    """
    facts = [
        Fact(pred="LOCATED_IN", args={"subject": "person", "object": "deutschland"}),
        Fact(pred="LOCATED_IN", args={"subject": "person", "object": "frankreich"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert (
        is_consistent == False
    ), "SAT-Solver sollte erkennen, dass 'deutschland' und 'frankreich' sich ausschließen"
    print("[OK] Location Conflict erkannt: deutschland != frankreich")


def test_allows_hierarchical_locations(engine):
    """
    Test: Hierarchische Locations (berlin → deutschland) sind erlaubt.
    """
    facts = [
        Fact(pred="LOCATED_IN", args={"subject": "person", "object": "berlin"}),
        Fact(pred="LOCATED_IN", args={"subject": "person", "object": "deutschland"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert (
        is_consistent == True
    ), "Hierarchische Locations sollten erlaubt sein (berlin ist in deutschland)"
    print("[OK] Hierarchische Locations erlaubt: berlin -> deutschland")


# ============================================================================
# TESTS: Graph Facts Integration
# ============================================================================


def test_loads_graph_facts_automatically(engine):
    """
    Test: Engine lädt automatisch relevante Fakten aus dem Graph.

    Wenn User "hund IS_A pflanze" behauptet, sollte Engine automatisch
    "hund IS_A tier" aus Graph laden und Widerspruch erkennen.
    """
    # Nur das widersprüchliche Fact hinzufügen (nicht das existierende "hund IS_A tier")
    facts = [Fact(pred="IS_A", args={"subject": "hund", "object": "pflanze"})]

    # check_consistency sollte automatisch "hund IS_A tier" aus Graph laden
    is_consistent = engine.check_consistency(facts, include_graph_facts=True)

    assert is_consistent == False, (
        "Engine sollte automatisch 'hund IS_A tier' aus Graph laden und "
        "Widerspruch mit 'hund IS_A pflanze' erkennen"
    )
    print("[OK] Graph Facts automatisch geladen und Widerspruch erkannt")


def test_without_graph_facts_no_contradiction(engine):
    """
    Test: Ohne Graph-Facts wird kein Widerspruch erkannt (Baseline).

    Dies zeigt den Unterschied zur neuen Funktionalität.
    """
    facts = [Fact(pred="IS_A", args={"subject": "unbekannt", "object": "pflanze"})]

    # Ohne Graph-Facts: Nur das eine Fact, keine Constraints
    is_consistent = engine.check_consistency(facts, include_graph_facts=False)

    assert is_consistent == True, (
        "Ohne Graph-Facts sollte kein Widerspruch erkannt werden "
        "(nur ein einzelnes Fact, keine Constraints)"
    )
    print("[OK] Ohne Graph-Facts: Kein Widerspruch (wie erwartet)")


# ============================================================================
# TESTS: Complex Multi-Fact Scenarios
# ============================================================================


def test_complex_scenario_multiple_contradictions(engine):
    """
    Test: Komplexes Szenario mit mehreren Widersprüchen.

    Kombiniert IS_A-, Property- und Location-Konflikte.
    """
    facts = [
        # IS_A Konflikt: hund ist tier UND pflanze
        Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
        Fact(pred="IS_A", args={"subject": "hund", "object": "pflanze"}),
        # Property Konflikt: apfel ist rot UND grün
        Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"}),
        Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "grün"}),
        # Location Konflikt: person in deutschland UND frankreich
        Fact(pred="LOCATED_IN", args={"subject": "person", "object": "deutschland"}),
        Fact(pred="LOCATED_IN", args={"subject": "person", "object": "frankreich"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert (
        is_consistent == False
    ), "SAT-Solver sollte mindestens einen der Widersprüche erkennen"
    print("[OK] Komplexes Szenario: Mehrere Widersprüche erkannt")


def test_complex_scenario_all_consistent(engine):
    """
    Test: Komplexes konsistentes Szenario (kein Widerspruch).
    """
    facts = [
        # Hierarchische IS_A
        Fact(pred="IS_A", args={"subject": "hund", "object": "säugetier"}),
        Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
        # Kompatible Properties
        Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "groß"}),
        Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "braun"}),
        # Hierarchische Location
        Fact(pred="LOCATED_IN", args={"subject": "hund", "object": "berlin"}),
        Fact(pred="LOCATED_IN", args={"subject": "hund", "object": "deutschland"}),
    ]

    is_consistent = engine.check_consistency(facts)

    assert (
        is_consistent == True
    ), "Konsistentes Szenario sollte keine Widersprüche haben"
    print("[OK] Komplexes konsistentes Szenario: Keine Widersprüche")


# ============================================================================
# EDGE CASES
# ============================================================================


def test_empty_facts_list(engine):
    """
    Test: Leere Faktenliste ist konsistent.
    """
    facts = []

    is_consistent = engine.check_consistency(facts)

    assert is_consistent == True, "Leere Faktenliste sollte konsistent sein"
    print("[OK] Leere Faktenliste ist konsistent")


def test_single_fact(engine):
    """
    Test: Einzelnes Fact ist konsistent.
    """
    facts = [Fact(pred="IS_A", args={"subject": "hund", "object": "tier"})]

    is_consistent = engine.check_consistency(facts)

    assert is_consistent == True, "Einzelnes Fact sollte konsistent sein"
    print("[OK] Einzelnes Fact ist konsistent")


def test_handles_unknown_concepts(engine):
    """
    Test: Engine handhabt unbekannte Konzepte graceful.
    """
    facts = [
        Fact(pred="IS_A", args={"subject": "alien", "object": "außerirdischer"}),
        Fact(pred="IS_A", args={"subject": "alien", "object": "lebewesen"}),
    ]

    # Sollte nicht crashen, sondern konsistent sein (keine bekannten Constraints)
    is_consistent = engine.check_consistency(facts)

    assert isinstance(is_consistent, bool), "Sollte Boolean zurückgeben"
    print("[OK] Unbekannte Konzepte werden graceful behandelt")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_sat_solver_with_ontology_constraints_enabled(engine):
    """
    Test: Ontologie-Constraints sind aktiviert.
    """
    assert engine.use_sat == True, "SAT-Solver sollte aktiviert sein"
    assert (
        engine.use_ontology_constraints == True
    ), "Ontologie-Constraints sollten aktiviert sein"
    assert (
        engine.ontology_generator is not None
    ), "OntologyConstraintGenerator sollte initialisiert sein"
    print("[OK] SAT-Solver und Ontologie-Constraints sind aktiviert")


def test_ontology_generator_finds_siblings(engine):
    """
    Test: OntologyConstraintGenerator findet Geschwister-Konzepte.
    """
    generator = engine.ontology_generator

    # tier und pflanze sind Geschwister (beide Kinder von lebewesen)
    assert (
        generator.are_concepts_mutually_exclusive("tier", "pflanze") == True
    ), "tier und pflanze sollten als Geschwister erkannt werden"

    # tier und säugetier sind NICHT Geschwister (hierarchisch)
    assert (
        generator.are_concepts_mutually_exclusive("tier", "säugetier") == False
    ), "tier und säugetier sollten NICHT als Geschwister erkannt werden"

    print("[OK] OntologyConstraintGenerator findet Geschwister korrekt")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
