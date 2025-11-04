"""
Cheryl's Birthday Puzzle - Als Problem-Definition (KEINE puzzle-spezifische Logik!)

Das berühmte Logik-Puzzle als Anwendungsfall für generische epistemische Reasoning-Komponenten.

WICHTIG: Dieser Test verwendet NUR generische Komponenten (component_37, component_38).
Keine puzzle-spezifische Logik in den Komponenten selbst!

Puzzle-Beschreibung:
--------------------
Cheryl gibt Albert und Bernard 10 mögliche Geburtstage:
May 15, May 16, May 19
June 17, June 18
July 14, July 16
August 14, August 15, August 17

Albert kennt nur den MONAT.
Bernard kennt nur den TAG.

Dann folgt dieser Dialog:
1. Albert: "Ich weiß, dass Bernard es nicht weiß."
2. Bernard: "Anfangs wusste ich es nicht, aber jetzt weiß ich es."
3. Albert: "Dann weiß ich es jetzt auch."

Lösung: July 16

Reasoning:
----------
1. Albert sagt "Ich weiß, dass Bernard es nicht weiß"
   → Albert's Monat darf KEINEN eindeutigen Tag haben
   → Eliminiert May (hat 19, eindeutig) und June (hat 18, eindeutig)
   → Verbleiben: July 14, July 16, August 14, August 15, August 17

2. Bernard sagt "Jetzt weiß ich es"
   → Bernard's Tag muss nach Elimination eindeutig sein
   → Eliminiert 14 (kommt in July und August vor)
   → Verbleiben: July 16, August 15, August 17

3. Albert sagt "Dann weiß ich es jetzt auch"
   → Albert's Monat muss jetzt eindeutig sein
   → July hat nur 16, August hat 15 und 17
   → Lösung: July 16
"""

import pytest
from component_37_partial_observation import (
    WorldObject,
    PartialObserver,
    PartialObservationReasoner,
)
from component_38_elimination_reasoning import (
    StatementType,
    AgentStatement,
    EliminationReasoner,
)
from component_35_epistemic_engine import EpistemicEngine
from component_1_netzwerk import KonzeptNetzwerk


def create_cheryls_birthday_problem():
    """
    Erstelle Cheryl's Birthday Problem als WorldObject-Liste.

    WICHTIG: Dies ist reine Daten-Definition, keine Logik!
    """
    dates = [
        # May
        WorldObject("may_15", {"month": "May", "day": 15}),
        WorldObject("may_16", {"month": "May", "day": 16}),
        WorldObject("may_19", {"month": "May", "day": 19}),
        # June
        WorldObject("june_17", {"month": "June", "day": 17}),
        WorldObject("june_18", {"month": "June", "day": 18}),
        # July
        WorldObject("july_14", {"month": "July", "day": 14}),
        WorldObject("july_16", {"month": "July", "day": 16}),
        # August
        WorldObject("august_14", {"month": "August", "day": 14}),
        WorldObject("august_15", {"month": "August", "day": 15}),
        WorldObject("august_17", {"month": "August", "day": 17}),
    ]
    return dates


def create_cheryls_birthday_statements():
    """
    Erstelle Dialog als AgentStatement-Liste.

    WICHTIG: Dies ist reine Daten-Definition, keine Logik!
    """
    statements = [
        # 1. Albert: "Ich weiß, dass Bernard es nicht weiß"
        AgentStatement(
            speaker="albert",
            statement_type=StatementType.I_KNOW_OTHER_DOESNT_KNOW,
            about_agent="bernard",
            turn=1,
            metadata={"original_text": "I know that Bernard doesn't know"},
        ),
        # 2. Bernard: "Jetzt weiß ich es"
        AgentStatement(
            speaker="bernard",
            statement_type=StatementType.NOW_I_KNOW,
            turn=2,
            metadata={"original_text": "At first I didn't know, but now I know"},
        ),
        # 3. Albert: "Dann weiß ich es jetzt auch"
        AgentStatement(
            speaker="albert",
            statement_type=StatementType.NOW_I_KNOW,
            turn=3,
            metadata={"original_text": "Then I also know"},
        ),
    ]
    return statements


class TestCheryls_Birthday:
    """Test Cheryl's Birthday Puzzle using generic components"""

    def test_solve_cheryls_birthday(self):
        """
        Solve Cheryl's Birthday using ONLY generic reasoning components.

        Expected solution: july_16
        """
        # Setup infrastructure
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        # Create problem (pure data)
        dates = create_cheryls_birthday_problem()
        reasoner.add_objects(dates)

        # Create observers (pure data)
        albert = PartialObserver("albert", observable_properties=["month"])
        bernard = PartialObserver("bernard", observable_properties=["day"])

        reasoner.add_observer(albert)
        reasoner.add_observer(bernard)

        # Create agents in epistemic engine
        engine.create_agent("albert", "Albert")
        engine.create_agent("bernard", "Bernard")

        # Create elimination reasoner with GENERIC rules
        elim_reasoner = EliminationReasoner(reasoner)
        elim_reasoner.create_standard_rules()

        # Create statements (pure data)
        statements = create_cheryls_birthday_statements()

        # SOLVE using generic reasoning!
        solution, proof_tree = elim_reasoner.process_statements(dates, statements)

        # Verify solution
        assert solution is not None, "No solution found!"
        assert (
            solution.object_id == "july_16"
        ), f"Expected july_16, got {solution.object_id}"

        print(f"\n[SUCCESS] Cheryl's Birthday solved: {solution.object_id}")
        print(f"  Month: {solution.get_property('month')}")
        print(f"  Day: {solution.get_property('day')}")
        print(f"  Deductive steps: {len(elim_reasoner.deductive_chain.steps)}")
        print(f"  Proof tree steps: {len(proof_tree.root_steps)}")

    def test_step_by_step_elimination(self):
        """
        Test each elimination step individually to verify reasoning.
        """
        # Setup
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        dates = create_cheryls_birthday_problem()
        reasoner.add_objects(dates)

        albert = PartialObserver("albert", observable_properties=["month"])
        bernard = PartialObserver("bernard", observable_properties=["day"])

        reasoner.add_observer(albert)
        reasoner.add_observer(bernard)

        engine.create_agent("albert", "Albert")
        engine.create_agent("bernard", "Bernard")

        elim_reasoner = EliminationReasoner(reasoner)
        elim_reasoner.create_standard_rules()

        # Step 1: Albert says "I know Bernard doesn't know"
        print("\n=== Step 1: Albert says 'I know Bernard doesn't know' ===")

        candidates = dates.copy()

        stmt1 = AgentStatement(
            speaker="albert",
            statement_type=StatementType.I_KNOW_OTHER_DOESNT_KNOW,
            about_agent="bernard",
            turn=1,
        )

        solution, _ = elim_reasoner.process_statements(candidates, [stmt1])

        step1_candidates = elim_reasoner.deductive_chain.current_candidates

        print(f"Candidates after step 1: {len(step1_candidates)}")
        for obj in step1_candidates:
            print(f"  - {obj.object_id}: {obj.properties}")

        # Should eliminate May and June (have unique days 19 and 18)
        remaining_months = set(obj.get_property("month") for obj in step1_candidates)
        assert "May" not in remaining_months, "May should be eliminated"
        assert "June" not in remaining_months, "June should be eliminated"
        assert "July" in remaining_months, "July should remain"
        assert "August" in remaining_months, "August should remain"

        # Step 2: Bernard says "Now I know"
        print("\n=== Step 2: Bernard says 'Now I know' ===")

        # Reset and re-run with both statements
        elim_reasoner = EliminationReasoner(reasoner)
        elim_reasoner.create_standard_rules()

        stmt2 = AgentStatement(
            speaker="bernard", statement_type=StatementType.NOW_I_KNOW, turn=2
        )

        solution, _ = elim_reasoner.process_statements(candidates, [stmt1, stmt2])

        step2_candidates = elim_reasoner.deductive_chain.current_candidates

        print(f"Candidates after step 2: {len(step2_candidates)}")
        for obj in step2_candidates:
            print(f"  - {obj.object_id}: {obj.properties}")

        # Should eliminate day 14 (appears in both July and August)
        remaining_days = set(obj.get_property("day") for obj in step2_candidates)
        assert 14 not in remaining_days, "Day 14 should be eliminated"

        # Step 3: Albert says "Now I also know"
        print("\n=== Step 3: Albert says 'Now I also know' ===")

        # Reset and re-run with all statements
        elim_reasoner = EliminationReasoner(reasoner)
        elim_reasoner.create_standard_rules()

        stmt3 = AgentStatement(
            speaker="albert", statement_type=StatementType.NOW_I_KNOW, turn=3
        )

        solution, proof_tree = elim_reasoner.process_statements(
            candidates, [stmt1, stmt2, stmt3]
        )

        print(f"\nFinal solution: {solution.object_id if solution else 'None'}")

        assert solution is not None
        assert solution.object_id == "july_16"

    def test_uniqueness_analysis(self):
        """Test uniqueness analysis on Cheryl's Birthday dates"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        dates = create_cheryls_birthday_problem()
        reasoner.add_objects(dates)

        # Test unique days
        unique_days = reasoner.uniqueness.get_unique_properties("day")

        print(f"\nUnique days: {unique_days}")

        assert 18 in unique_days, "Day 18 should be unique (only June 18)"
        assert 19 in unique_days, "Day 19 should be unique (only May 19)"
        assert 14 not in unique_days, "Day 14 is NOT unique (July 14, August 14)"
        assert 15 not in unique_days, "Day 15 is NOT unique (May 15, August 15)"

        # Test unique months
        unique_months = reasoner.uniqueness.get_unique_properties("month")

        print(f"Unique months: {unique_months}")

        # No month is unique (all have multiple dates)
        assert len(unique_months) == 0

    def test_partition_analysis(self):
        """Test partition analysis on Cheryl's Birthday dates"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        dates = create_cheryls_birthday_problem()
        reasoner.add_objects(dates)

        # Partition by month
        month_partitions = reasoner.partitions.partition_by_property("month")

        print("\nPartitions by month:")
        for month, objs in month_partitions.items():
            print(f"  {month}: {len(objs)} dates")
            for obj in objs:
                print(f"    - {obj.object_id}")

        assert len(month_partitions["May"]) == 3
        assert len(month_partitions["June"]) == 2
        assert len(month_partitions["July"]) == 2
        assert len(month_partitions["August"]) == 3

        # Check which partitions have unique identifiers
        may_has_unique = reasoner.partitions.partition_has_unique_identifier(
            "month", "May", "day"
        )
        june_has_unique = reasoner.partitions.partition_has_unique_identifier(
            "month", "June", "day"
        )
        july_has_unique = reasoner.partitions.partition_has_unique_identifier(
            "month", "July", "day"
        )

        print(f"\nMay has unique day: {may_has_unique}")
        print(f"June has unique day: {june_has_unique}")
        print(f"July has unique day: {july_has_unique}")

        # May has day 19 (unique) and June has day 18 (unique)
        assert may_has_unique is True
        assert june_has_unique is True
        # July has no unique day
        assert july_has_unique is False

    def test_proof_tree_generation(self):
        """Test that proof tree is correctly generated"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        dates = create_cheryls_birthday_problem()
        reasoner.add_objects(dates)

        albert = PartialObserver("albert", observable_properties=["month"])
        bernard = PartialObserver("bernard", observable_properties=["day"])

        reasoner.add_observer(albert)
        reasoner.add_observer(bernard)

        engine.create_agent("albert", "Albert")
        engine.create_agent("bernard", "Bernard")

        elim_reasoner = EliminationReasoner(reasoner)
        elim_reasoner.create_standard_rules()

        statements = create_cheryls_birthday_statements()

        solution, proof_tree = elim_reasoner.process_statements(dates, statements)

        print(f"\nProof tree:")
        print(f"  Query: {proof_tree.query}")
        print(f"  Root steps: {len(proof_tree.root_steps)}")

        for i, step in enumerate(proof_tree.root_steps):
            print(f"  Step {i+1}: {step.step_type.value} - {step.explanation_text}")

        # Should have initial + 3 eliminations + conclusion
        assert len(proof_tree.root_steps) >= 4

        # Last step should be conclusion
        assert proof_tree.root_steps[-1].step_type.value == "conclusion"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
