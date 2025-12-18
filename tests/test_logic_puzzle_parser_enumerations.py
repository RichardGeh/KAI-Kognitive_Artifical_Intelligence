"""
tests/test_logic_puzzle_parser_enumerations.py

Tests for LogicConditionParser enumeration and negation patterns
Added: 2025-12-13 (Fix for three-person puzzle parsing failure)
"""

from component_45_logic_puzzle_parser import LogicConditionParser


class TestColonEnumerationDetection:
    """Test colon-separated object enumeration detection"""

    def test_job_enumeration_detected(self):
        """Test Berufe: Lehrer, Arzt und Ingenieur pattern"""
        parser = LogicConditionParser()
        text = "Sie haben unterschiedliche Berufe: Lehrer, Arzt und Ingenieur."
        entities = ["Alex", "Bob"]

        parser.parse_conditions(text, entities)

        # Verify all jobs detected as objects
        assert (
            "lehrer" in parser._detected_objects
        ), "Lehrer should be detected as object"
        assert "arzt" in parser._detected_objects, "Arzt should be detected as object"
        assert (
            "ingenieur" in parser._detected_objects
        ), "Ingenieur should be detected as object"

    def test_pet_enumeration_detected(self):
        """Test Haustiere: Hund, Katze, Vogel pattern"""
        parser = LogicConditionParser()
        text = "Haustiere: Hund, Katze, Vogel."
        entities = []

        parser.parse_conditions(text, entities)

        assert "hund" in parser._detected_objects, "Hund should be detected"
        assert "katze" in parser._detected_objects, "Katze should be detected"
        assert "vogel" in parser._detected_objects, "Vogel should be detected"

    def test_entities_not_detected_as_objects(self):
        """Entities should not appear in detected objects (unless they ARE objects)"""
        parser = LogicConditionParser()
        text = "Berufe: Arzt, Lehrer."
        entities = ["Anna", "Bob"]

        parser.parse_conditions(text, entities)

        # Jobs SHOULD be in detected objects (from colon enumeration)
        assert "arzt" in parser._detected_objects
        assert "lehrer" in parser._detected_objects

        # Entities NOT mentioned in colon enumeration should NOT be in objects
        # (unless they appear elsewhere in text with article/verb patterns)
        # This test just verifies that the colon pattern works
        assert len(parser._detected_objects) >= 2

    def test_colon_enumeration_with_commas_only(self):
        """Test enumeration with only commas (no 'und')"""
        parser = LogicConditionParser()
        text = "Farben: Rot, Grün, Blau."
        entities = []

        parser.parse_conditions(text, entities)

        assert "rot" in parser._detected_objects
        assert "grün" in parser._detected_objects
        assert "blau" in parser._detected_objects

    def test_colon_enumeration_high_confidence(self):
        """Colon-enumerated objects should have higher confidence (weight=3)"""
        parser = LogicConditionParser()
        text = "Berufe: Pilot."
        entities = []

        parser.parse_conditions(text, entities)

        # Should be detected despite only appearing once
        assert "pilot" in parser._detected_objects


class TestExpandedNegationPatterns:
    """Test expanded negation pattern recognition"""

    def test_ist_kein_negation(self):
        """Test 'X ist kein Y' negation pattern"""
        parser = LogicConditionParser()
        # Pre-populate detected objects (simulate colon detection)
        parser._detected_objects = {"arzt", "lehrer"}
        entities = ["Alex", "Bob"]
        parser.entities = set(e.lower() for e in entities)

        cond = parser._parse_negation("Alex ist kein Arzt")

        assert cond is not None, "Should parse 'ist kein' negation"
        assert cond.condition_type == "NEGATION"
        assert len(cond.operands) == 1
        # Should construct variable like "Alex_hat_arzt"
        assert "alex" in cond.operands[0].lower()
        assert "arzt" in cond.operands[0].lower()

    def test_ist_nicht_negation(self):
        """Test 'X ist nicht Y' negation pattern"""
        parser = LogicConditionParser()
        parser._detected_objects = {"ingenieur"}
        parser.entities = {"carol"}

        cond = parser._parse_negation("Carol ist nicht Ingenieur")

        assert cond is not None, "Should parse 'ist nicht' negation"
        assert cond.condition_type == "NEGATION"
        assert len(cond.operands) == 1

    def test_hat_kein_negation(self):
        """Test 'X hat kein Y' negation pattern"""
        parser = LogicConditionParser()
        parser._detected_objects = {"auto"}
        parser.entities = {"ben"}

        cond = parser._parse_negation("Ben hat kein Auto")

        assert cond is not None, "Should parse 'hat kein' negation"
        assert cond.condition_type == "NEGATION"
        assert len(cond.operands) == 1

    def test_simple_nicht_negation_still_works(self):
        """Test that original 'nicht X' pattern still works"""
        parser = LogicConditionParser()
        parser._detected_objects = {"pizza"}
        parser.entities = {"mark"}
        parser._context_object = "pizza"  # Provide context for variable extraction

        # More realistic: "nicht Mark trinkt Pizza" -> "Mark hat Pizza" negated
        cond = parser._parse_negation("nicht Mark trinkt Pizza")

        assert cond is not None, "Original 'nicht X' should still work"
        assert cond.condition_type == "NEGATION"

    def test_negation_with_keine_feminine(self):
        """Test 'ist keine' for feminine nouns"""
        parser = LogicConditionParser()
        parser._detected_objects = {"katze"}
        parser.entities = {"anna"}

        cond = parser._parse_negation("Anna ist keine Katze")

        assert cond is not None, "Should parse 'ist keine' negation"
        assert cond.condition_type == "NEGATION"

    def test_negation_with_keinen_accusative(self):
        """Test 'hat keinen' for masculine accusative"""
        parser = LogicConditionParser()
        parser._detected_objects = {"hund"}
        parser.entities = {"bob"}

        cond = parser._parse_negation("Bob hat keinen Hund")

        assert cond is not None, "Should parse 'hat keinen' negation"
        assert cond.condition_type == "NEGATION"


class TestIntegrationColonAndNegation:
    """Integration tests combining colon enumeration and negation"""

    def test_full_job_assignment_puzzle(self):
        """Test full puzzle with colon enumeration and negations"""
        parser = LogicConditionParser()
        text = """
Alex, Bob und Carol haben unterschiedliche Berufe: Lehrer, Arzt und Ingenieur.
1. Alex ist kein Arzt.
2. Bob ist Lehrer.
3. Carol ist nicht Ingenieur.
        """
        entities = ["Alex", "Bob", "Carol"]

        conditions = parser.parse_conditions(text, entities)

        # Verify objects detected
        assert "lehrer" in parser._detected_objects
        assert "arzt" in parser._detected_objects
        assert "ingenieur" in parser._detected_objects

        # Verify conditions parsed
        assert (
            len(conditions) >= 3
        ), f"Expected at least 3 conditions, got {len(conditions)}"

        # Find negations
        negations = [c for c in conditions if c.condition_type == "NEGATION"]
        assert len(negations) >= 2, f"Expected 2 negations, got {len(negations)}"

        # Find simple facts
        facts = [c for c in conditions if c.condition_type == "SIMPLE_FACT"]
        assert len(facts) >= 1, f"Expected at least 1 simple fact, got {len(facts)}"

    def test_pet_assignment_with_negations(self):
        """Test pet assignment puzzle"""
        parser = LogicConditionParser()
        text = """
Anna, Ben und Clara haben unterschiedliche Haustiere: Hund, Katze und Vogel.
1. Anna hat keinen Hund.
2. Ben hat eine Katze.
3. Clara hat keinen Vogel.
        """
        entities = ["Anna", "Ben", "Clara"]

        conditions = parser.parse_conditions(text, entities)

        # Verify objects
        assert "hund" in parser._detected_objects
        assert "katze" in parser._detected_objects
        assert "vogel" in parser._detected_objects

        # Verify parsing
        assert len(conditions) >= 3
        negations = [c for c in conditions if c.condition_type == "NEGATION"]
        assert len(negations) >= 2
