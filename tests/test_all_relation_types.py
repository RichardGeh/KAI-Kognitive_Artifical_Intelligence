"""
KAI Test Suite - All Relation Types Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestAllRelationTypes:
    """Umfassende Tests für alle unterstützten Relationstypen mit Edge Cases."""

    def test_is_a_relation_comprehensive(self, netzwerk_session, clean_test_concepts):
        """Testet IS_A Relation mit verschiedenen Edge Cases."""
        # Standard-Fall
        subject1 = f"{clean_test_concepts}hund_isa"
        object1 = f"{clean_test_concepts}tier_isa"
        netzwerk_session.assert_relation(subject1, "IS_A", object1, "Standard IS_A")

        facts = netzwerk_session.query_graph_for_facts(subject1)
        assert "IS_A" in facts
        assert object1 in facts["IS_A"]

        # Edge Case 1: Mehrere IS_A Relationen (Taxonomie-Kette)
        object2 = f"{clean_test_concepts}lebewesen_isa"
        netzwerk_session.assert_relation(subject1, "IS_A", object2, "Zweite IS_A")
        facts = netzwerk_session.query_graph_for_facts(subject1)
        assert len(facts["IS_A"]) == 2
        assert object1 in facts["IS_A"] and object2 in facts["IS_A"]

        # Edge Case 2: Transitivität (hund -> tier -> lebewesen)
        netzwerk_session.assert_relation(object1, "IS_A", object2, "Transitive IS_A")
        facts_transitive = netzwerk_session.query_graph_for_facts(object1)
        assert object2 in facts_transitive["IS_A"]

        logger.info("[SUCCESS] IS_A Relation: Standard, Multiple, Transitive Cases")

    def test_has_property_relation_comprehensive(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet HAS_PROPERTY Relation mit Edge Cases."""
        subject = f"{clean_test_concepts}apfel_prop"

        # Standard-Fall: Einzelne Eigenschaft
        prop1 = f"{clean_test_concepts}rot"
        netzwerk_session.assert_relation(subject, "HAS_PROPERTY", prop1, "Farbe")
        facts = netzwerk_session.query_graph_for_facts(subject)
        assert "HAS_PROPERTY" in facts
        assert prop1 in facts["HAS_PROPERTY"]

        # Edge Case 1: Mehrere Eigenschaften
        properties = [
            f"{clean_test_concepts}süß",
            f"{clean_test_concepts}rund",
            f"{clean_test_concepts}saftig",
        ]
        for prop in properties:
            netzwerk_session.assert_relation(
                subject, "HAS_PROPERTY", prop, f"Property {prop}"
            )

        facts = netzwerk_session.query_graph_for_facts(subject)
        assert len(facts["HAS_PROPERTY"]) == 4  # rot + 3 neue
        for prop in properties:
            assert prop in facts["HAS_PROPERTY"]

        # Edge Case 2: Widersprüchliche Eigenschaften (sollten beide erlaubt sein)
        prop_green = f"{clean_test_concepts}grün"
        netzwerk_session.assert_relation(
            subject, "HAS_PROPERTY", prop_green, "Grün (unreif)"
        )
        facts = netzwerk_session.query_graph_for_facts(subject)
        # System sollte BEIDE Farben erlauben (Kontext-abhängig: rot=reif, grün=unreif)
        assert prop1 in facts["HAS_PROPERTY"] and prop_green in facts["HAS_PROPERTY"]

        logger.info(
            "[SUCCESS] HAS_PROPERTY: Single, Multiple, Contradictory Properties"
        )

    def test_capable_of_relation_comprehensive(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet CAPABLE_OF Relation mit Edge Cases."""
        subject = f"{clean_test_concepts}vogel_cap"

        # Standard-Fall
        ability1 = f"{clean_test_concepts}fliegen"
        netzwerk_session.assert_relation(subject, "CAPABLE_OF", ability1, "Fliegen")
        facts = netzwerk_session.query_graph_for_facts(subject)
        assert "CAPABLE_OF" in facts
        assert ability1 in facts["CAPABLE_OF"]

        # Edge Case 1: Mehrere Fähigkeiten
        abilities = [
            f"{clean_test_concepts}singen",
            f"{clean_test_concepts}eier_legen",
            f"{clean_test_concepts}nestbauen",
        ]
        for ability in abilities:
            netzwerk_session.assert_relation(
                subject, "CAPABLE_OF", ability, f"Ability {ability}"
            )

        facts = netzwerk_session.query_graph_for_facts(subject)
        assert len(facts["CAPABLE_OF"]) == 4

        # Edge Case 2: Negative Fähigkeit (kann NICHT X) - über INCAPABLE_OF
        # Dies könnte eine zukünftige Erweiterung sein, aber für jetzt dokumentieren wir es
        logger.info(
            "[INFO] Negative abilities (INCAPABLE_OF) sind nicht implementiert - Future Feature"
        )

        # Edge Case 3: Vererbte Fähigkeiten (Vogel IST Tier, Tiere können atmen)
        tier_concept = f"{clean_test_concepts}tier_cap"
        netzwerk_session.assert_relation(subject, "IS_A", tier_concept, "Taxonomie")
        netzwerk_session.assert_relation(
            tier_concept, "CAPABLE_OF", f"{clean_test_concepts}atmen", "Basis-Fähigkeit"
        )

        # Prüfe ob beide Fähigkeiten abrufbar sind (direkte + vererbte)
        facts_subject = netzwerk_session.query_graph_for_facts(subject)
        facts_parent = netzwerk_session.query_graph_for_facts(tier_concept)
        assert ability1 in facts_subject["CAPABLE_OF"]  # Eigene Fähigkeit
        assert (
            f"{clean_test_concepts}atmen" in facts_parent["CAPABLE_OF"]
        )  # Parent-Fähigkeit

        logger.info("[SUCCESS] CAPABLE_OF: Single, Multiple, Inheritance Cases")

    def test_part_of_relation_comprehensive(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet PART_OF Relation mit Edge Cases."""
        # Standard-Fall: Rad ist Teil von Auto
        part = f"{clean_test_concepts}rad_part"
        whole = f"{clean_test_concepts}auto_part"
        netzwerk_session.assert_relation(part, "PART_OF", whole, "Teil-Ganzes")
        facts = netzwerk_session.query_graph_for_facts(part)
        assert "PART_OF" in facts
        assert whole in facts["PART_OF"]

        # Edge Case 1: Ein Teil gehört zu mehreren Ganzen (z.B. Schraube in Auto UND Fahrrad)
        whole2 = f"{clean_test_concepts}fahrrad_part"
        netzwerk_session.assert_relation(
            part, "PART_OF", whole2, "Auch Teil von Fahrrad"
        )
        facts = netzwerk_session.query_graph_for_facts(part)
        assert len(facts["PART_OF"]) == 2
        assert whole in facts["PART_OF"] and whole2 in facts["PART_OF"]

        # Edge Case 2: Komposition-Kette (Rad -> Auto -> Verkehrssystem)
        super_whole = f"{clean_test_concepts}verkehrssystem"
        netzwerk_session.assert_relation(
            whole, "PART_OF", super_whole, "Auto Teil von Verkehrssystem"
        )
        facts_transitive = netzwerk_session.query_graph_for_facts(whole)
        assert super_whole in facts_transitive["PART_OF"]

        # Edge Case 3: Synonym-Nutzung für PART_OF (Synonym statt echter Teil)
        # HINWEIS: Laut CLAUDE.md wird PART_OF auch für Synonyme verwendet
        synonym_test1 = f"{clean_test_concepts}pkw_synonym"
        synonym_test2 = f"{clean_test_concepts}wagen_synonym"
        netzwerk_session.assert_relation(
            synonym_test1, "PART_OF", synonym_test2, "Synonym via PART_OF"
        )
        facts_synonym = netzwerk_session.query_graph_for_facts(synonym_test1)
        assert synonym_test2 in facts_synonym["PART_OF"]

        logger.info("[SUCCESS] PART_OF: Single, Multiple, Transitive, Synonym Cases")

    def test_located_in_relation_comprehensive(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet LOCATED_IN Relation mit Edge Cases."""
        # Standard-Fall: Berlin in Deutschland
        entity = f"{clean_test_concepts}berlin_loc"
        location = f"{clean_test_concepts}deutschland_loc"
        netzwerk_session.assert_relation(
            entity, "LOCATED_IN", location, "Stadt in Land"
        )
        facts = netzwerk_session.query_graph_for_facts(entity)
        assert "LOCATED_IN" in facts
        assert location in facts["LOCATED_IN"]

        # Edge Case 1: Hierarchische Orte (Berlin -> Deutschland -> Europa)
        super_location = f"{clean_test_concepts}europa_loc"
        netzwerk_session.assert_relation(
            location, "LOCATED_IN", super_location, "Land in Kontinent"
        )
        facts_transitive = netzwerk_session.query_graph_for_facts(location)
        assert super_location in facts_transitive["LOCATED_IN"]

        # Edge Case 2: Mehrere Orte für bewegliche Objekte (z.B. Person in Berlin UND München)
        movable = f"{clean_test_concepts}person_movable"
        loc1 = f"{clean_test_concepts}berlin_multi"
        loc2 = f"{clean_test_concepts}münchen_multi"
        netzwerk_session.assert_relation(movable, "LOCATED_IN", loc1, "Wohnt in Berlin")
        netzwerk_session.assert_relation(
            movable, "LOCATED_IN", loc2, "Arbeitet in München"
        )
        facts_multi = netzwerk_session.query_graph_for_facts(movable)
        assert len(facts_multi["LOCATED_IN"]) == 2

        # Edge Case 3: Virtuelle/Abstrakte Orte
        abstract_entity = f"{clean_test_concepts}gedanke"
        abstract_loc = f"{clean_test_concepts}gehirn"
        netzwerk_session.assert_relation(
            abstract_entity, "LOCATED_IN", abstract_loc, "Abstrakte Location"
        )
        facts_abstract = netzwerk_session.query_graph_for_facts(abstract_entity)
        assert abstract_loc in facts_abstract["LOCATED_IN"]

        logger.info(
            "[SUCCESS] LOCATED_IN: Single, Hierarchical, Multiple, Abstract Cases"
        )

    def test_relation_type_combination_scenarios(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet Kombinationen verschiedener Relationstypen für realistisches Wissen."""
        # Realistisches Beispiel: Umfassendes Wissen über "Hund"
        dog = f"{clean_test_concepts}hund_comprehensive"

        # IS_A: Taxonomie
        netzwerk_session.assert_relation(
            dog, "IS_A", f"{clean_test_concepts}säugetier", "Taxonomie"
        )
        netzwerk_session.assert_relation(
            dog, "IS_A", f"{clean_test_concepts}haustier", "Rolle"
        )

        # HAS_PROPERTY: Eigenschaften
        netzwerk_session.assert_relation(
            dog, "HAS_PROPERTY", f"{clean_test_concepts}treu", "Charakter"
        )
        netzwerk_session.assert_relation(
            dog, "HAS_PROPERTY", f"{clean_test_concepts}fellbedeckt", "Aussehen"
        )

        # CAPABLE_OF: Fähigkeiten
        netzwerk_session.assert_relation(
            dog, "CAPABLE_OF", f"{clean_test_concepts}bellen", "Laut"
        )
        netzwerk_session.assert_relation(
            dog, "CAPABLE_OF", f"{clean_test_concepts}laufen", "Bewegung"
        )

        # PART_OF: Körperteile als separate Konzepte
        schwanz = f"{clean_test_concepts}schwanz"
        netzwerk_session.assert_relation(schwanz, "PART_OF", dog, "Körperteil")

        # LOCATED_IN: Lebensraum
        netzwerk_session.assert_relation(
            dog, "LOCATED_IN", f"{clean_test_concepts}haushalt", "Wohnort"
        )

        # VERIFIKATION: Prüfe dass alle Relationstypen gespeichert wurden
        facts = netzwerk_session.query_graph_for_facts(dog)

        assert "IS_A" in facts and len(facts["IS_A"]) == 2
        assert "HAS_PROPERTY" in facts and len(facts["HAS_PROPERTY"]) == 2
        assert "CAPABLE_OF" in facts and len(facts["CAPABLE_OF"]) == 2
        assert "LOCATED_IN" in facts and len(facts["LOCATED_IN"]) == 1

        # Prüfe invertierte Relation (PART_OF)
        facts_schwanz = netzwerk_session.query_graph_for_facts(schwanz)
        assert "PART_OF" in facts_schwanz and dog in facts_schwanz["PART_OF"]

        logger.info(
            f"[SUCCESS] Kombiniertes Wissen: {len(facts)} Relationstypen für '{dog}'"
        )

    def test_relation_idempotence_across_types(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet Idempotenz für alle Relationstypen."""
        subject = f"{clean_test_concepts}idempotence_test"
        obj = f"{clean_test_concepts}target"

        relation_types = ["IS_A", "HAS_PROPERTY", "CAPABLE_OF", "PART_OF", "LOCATED_IN"]

        for rel_type in relation_types:
            # Erstelle Relation zweimal
            created1 = netzwerk_session.assert_relation(
                subject, rel_type, obj, f"First {rel_type}"
            )
            created2 = netzwerk_session.assert_relation(
                subject, rel_type, obj, f"Second {rel_type}"
            )

            # Erste sollte erstellt sein, zweite nicht
            assert created1 is True, f"{rel_type}: Erste Erstellung sollte True sein"
            assert (
                created2 is False
            ), f"{rel_type}: Zweite Erstellung sollte False sein (Duplikat)"

            # Verifiziere, dass nur eine Relation existiert
            with netzwerk_session.driver.session(database="neo4j") as session:
                result = session.run(
                    f"""
                    MATCH (s:Konzept {{name: $subject}})-[r:{rel_type}]->(o:Konzept {{name: $obj}})
                    RETURN count(r) AS count
                """,
                    subject=subject,
                    obj=obj,
                )
                count = result.single()["count"]
                assert (
                    count == 1
                ), f"{rel_type}: Sollte genau 1 Relation haben, hat aber {count}"

        logger.info(
            f"[SUCCESS] Idempotenz für alle {len(relation_types)} Relationstypen verifiziert"
        )

    @pytest.mark.slow
    def test_relation_query_performance_with_many_relations(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet Korrektheit beim Abfragen vieler Relationen verschiedener Typen.
        Vollständigkeit ist wichtiger als Geschwindigkeit.
        """
        import time

        subject = f"{clean_test_concepts}perf_many_rel"
        num_relations_per_type = 20

        # Erstelle viele Relationen verschiedener Typen
        relation_types = ["IS_A", "HAS_PROPERTY", "CAPABLE_OF", "PART_OF", "LOCATED_IN"]

        for rel_type in relation_types:
            for i in range(num_relations_per_type):
                obj = f"{clean_test_concepts}{rel_type.lower()}_obj_{i}"
                netzwerk_session.assert_relation(
                    subject, rel_type, obj, f"Batch {rel_type} {i}"
                )

        # HAUPTTEST: Vollständigkeit und Korrektheit
        start = time.time()
        facts = netzwerk_session.query_graph_for_facts(subject)
        elapsed = time.time() - start

        # Verifiziere Vollständigkeit (wichtigster Test!)
        for rel_type in relation_types:
            assert rel_type in facts, f"{rel_type} sollte in Ergebnissen vorhanden sein"
            assert (
                len(facts[rel_type]) == num_relations_per_type
            ), f"{rel_type} sollte {num_relations_per_type} haben, hat aber {len(facts[rel_type])}"

        # Performance-Info (informativ, kein Hard-Limit)
        logger.info(
            f"[SUCCESS] Query mit 100 Relationen (5 Typen): Vollständig und korrekt in {elapsed:.4f}s"
        )
        if elapsed > 2.0:
            logger.info(
                f"[INFO] Query dauerte {elapsed:.4f}s - Vollständigkeit erreicht!"
            )

    def test_custom_relation_types_extensibility(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet ob System mit benutzerdefinierten Relationstypen umgehen kann."""
        subject = f"{clean_test_concepts}custom_rel_subj"
        obj = f"{clean_test_concepts}custom_rel_obj"

        # Erstelle benutzerdefinierte Relationstypen
        custom_relations = ["CAUSED_BY", "SIMILAR_TO", "OPPOSITE_OF", "REQUIRES"]

        for custom_rel in custom_relations:
            # System sollte beliebige Relationstypen akzeptieren (Neo4j ist schema-less)
            created = netzwerk_session.assert_relation(
                subject, custom_rel, obj, f"Custom {custom_rel}"
            )
            assert created is True

            # Verifiziere im Graphen
            netzwerk_session.query_graph_for_facts(subject)
            # query_graph_for_facts() gibt nur bekannte Typen zurück - teste direkte Cypher-Query
            with netzwerk_session.driver.session(database="neo4j") as session:
                result = session.run(
                    f"""
                    MATCH (s:Konzept {{name: $subject}})-[r:{custom_rel}]->(o:Konzept {{name: $obj}})
                    RETURN count(r) AS count
                """,
                    subject=subject,
                    obj=obj,
                )
                count = result.single()["count"]
                assert count == 1, f"Custom Relation {custom_rel} sollte existieren"

        logger.info(
            f"[SUCCESS] System unterstützt {len(custom_relations)} benutzerdefinierte Relationstypen"
        )


# ============================================================================
# TESTS FÜR GRENZWERTE UND PERFORMANCE
# ============================================================================
